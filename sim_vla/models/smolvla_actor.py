"""The pretrained SmolVLA expert, conditioned on world-model latents.

Read against lerobot 0.4.4 and 0.6.1, whose ``VLAFlowMatching`` exposes
``embed_prefix`` / ``embed_suffix`` / ``denoise_step`` on ``policy.model``. The
flow methods are not on the policy: an earlier version of this file called
``policy.denoise_step`` and there is no such method.

Nor does the policy own a tokenizer. Tokenization normally happens in a
processor pipeline before a batch reaches the policy, so the tokenizer itself
hangs off ``model.vlm_with_expert.processor``. sim_vla tokenizes one fixed
instruction per task, so it resolves that object directly -- see
:data:`TOKENIZER_PATHS`.

What the real path is, and why it is not a one-line call:

``denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)`` needs a
prefix that has already been run through the VLM to produce a key/value cache.
So conditioning is two passes. The prefix is built once per observation --
tokenized instruction plus one state token -- and run through the VLM to get
the cache; every integration step of the sampler then reuses it. Building the
prefix inside the integration loop would be correct and ten times slower.

**The state token replaces the raw-state path, not the vision path twice
over.** SmolVLA's prefix is images, language, then one state token projected by
``state_proj``. Here the observation has already been through the world model,
so no images are supplied and the adapter produces the state slot directly.
``embed_prefix`` always calls ``state_proj(state)`` -- there is no path through
it that takes an already-projected embedding -- so ``state_token_mode`` decides
how that is handled:

``embedding`` (default) the adapter emits ``vlm_hidden_size`` and
    ``state_proj`` is swapped for an identity **for the duration of the
    call**. The prefix is still built by ``embed_prefix``, so the token
    ordering, the padding masks and the position ids are the pretrained ones;
    only the projection is stepped over. Worth the small trick: ``state_proj``
    takes 32 inputs, and routing a 4000-wide world-model feature through it
    would put a 32-dimensional bottleneck between the world model and the
    policy, which is most of what the world model is for.
``state_proj`` the adapter emits ``max_state_dim`` and the frozen projection
    runs normally. No swap, at that bottleneck. The conservative choice, and
    the one to fall back to if the swap ever misbehaves.

Actions are padded to ``max_action_dim`` (32 in the released checkpoint) before
``action_in_proj`` and sliced back to the task's width after
``action_out_proj``. The loss and the sampler both mask the padding, so the
policy is never scored on dimensions the robot does not have.

Frozen means ``requires_grad=False``, never ``detach()``: gradients flow
*through* the frozen transformer back into the adapter, which is the only
reason conditioning it works.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .pretrained import PretrainedError

# What the integration needs from VLAFlowMatching. Checked at construction so
# a version mismatch is reported here rather than inside a transformer.
REQUIRED_MODEL_METHODS = ("embed_prefix", "embed_suffix", "denoise_step")
REQUIRED_MODEL_MODULES = ("state_proj", "action_in_proj", "action_out_proj")

# Where the text tokenizer lives. In lerobot 0.4.4 and 0.6.1 the policy does
# not own one -- tokenization is normally done by a processor pipeline before
# the batch reaches the policy, and the tokenizer itself hangs off the VLM's
# processor. sim_vla tokenizes its own fixed instruction, so it needs the
# object rather than the pipeline.
TOKENIZER_PATHS = (
    "model.vlm_with_expert.processor.tokenizer",
    "vlm_with_expert.processor.tokenizer",
    "language_tokenizer",
    "model.vlm_with_expert.processor",
)

# How far a shortened chunk's velocities may move from the full chunk's first
# positions, relative to their size. Causal attention makes them equal up to
# kernel rounding; a mask that let actions see later ones would move them by
# the size of the velocity itself.
SHRINK_TOLERANCE = 1e-2


def freeze(module: nn.Module) -> int:
    """Stop a module's own parameters training, without cutting the graph."""
    count = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            count += 1
    return count


def resolve_attr(root: Any, paths: Sequence[str], what: str) -> tuple:
    """First dotted attribute path that exists, or an error naming all of them.

    Not restricted to ``nn.Module``: a tokenizer is not one, and requiring it
    to be was how the first attempt at this failed.
    """
    for path in paths:
        node: Any = root
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return node, path
    raise PretrainedError(
        f"could not find the {what} on this checkpoint. Tried: {list(paths)}. "
        f"Submodules present: {module_tree(root)[:30]}")


def module_tree(root: nn.Module, depth: int = 3) -> List[str]:
    return sorted(name for name, _ in root.named_modules()
                  if name and name.count(".") < depth)


class SmolVLAActor(nn.Module):
    """Pretrained SmolVLA driven by one world-model conditioning token."""

    def __init__(self, loaded, adapter: nn.Module, *, action_dim: int,
                 chunk_size: Optional[int] = None,
                 flow_steps: Optional[int] = None,
                 instruction: str = "",
                 state_token_mode: str = "embedding",
                 freeze_vlm: bool = True, train_expert: bool = True):
        super().__init__()
        self.loaded = loaded
        self.policy = loaded.policy
        self.model = loaded.model
        self.adapter = adapter
        self.action_dim = int(action_dim)
        self.state_token_mode = str(state_token_mode)

        missing = [name for name in REQUIRED_MODEL_METHODS
                   if not callable(getattr(self.model, name, None))]
        missing += [name for name in REQUIRED_MODEL_MODULES
                    if getattr(self.model, name, None) is None]
        if missing:
            raise PretrainedError(
                f"this VLAFlowMatching lacks {missing}; the integration targets "
                f"lerobot {loaded.lerobot_version}. Present: "
                f"{module_tree(self.model)[:30]}")

        from .pretrained import model_facts

        self.facts = model_facts(loaded)
        # The pretrained stack reads its *own* config.chunk_size when it builds
        # the action attention mask and slices the expert's output. Setting
        # only the wrapper's copy gave a policy that was supervised on one
        # horizon and predicted another, with no error anywhere: the loss
        # descends, the chunks just do not line up. So the override is applied
        # to the config the model actually reads, and then verified.
        self.chunk_size = int(chunk_size or self.facts["chunk_size"])
        self._synchronise_chunk(self.chunk_size)
        self.flow_steps = int(flow_steps or self.facts["num_steps"])
        self.max_action_dim = int(self.facts["max_action_dim"])
        self.vlm_hidden = int(self.facts["vlm_hidden_size"])
        if self.action_dim > self.max_action_dim:
            raise PretrainedError(
                f"the task has {self.action_dim} action dimensions and the "
                f"checkpoint pads to {self.max_action_dim}")

        expected = (self.vlm_hidden if self.state_token_mode == "embedding"
                    else int(self.facts["max_state_dim"]))
        if int(getattr(adapter, "token_dim", -1)) != expected:
            raise PretrainedError(
                f"the adapter emits {getattr(adapter, 'token_dim', None)}-wide "
                f"tokens; state_token_mode={self.state_token_mode!r} on this "
                f"checkpoint needs {expected}")

        # The pretrained stack decides the device. It arrives already placed
        # -- from_pretrained does not necessarily leave it on CPU -- while a
        # freshly built adapter is wherever torch defaults to, and the two
        # meet inside an embedding lookup that reports "index is on cpu,
        # different from other tensors on cuda". Aligning here means a caller
        # never has to know which way round it was.
        self.adapter.to(self.device)

        self.instruction = str(instruction)
        self.tokenizer_path = ""
        self._lang_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.frozen: Dict[str, int] = {}
        if freeze_vlm:
            # The whole pretrained stack is frozen, then the expert is thawed
            # if it is meant to train. Freezing by name would depend on a
            # module layout that moves between versions.
            self.frozen["model"] = freeze(self.model)
            if train_expert:
                self.frozen["thawed_expert"] = -self._thaw_expert()

    def _synchronise_chunk(self, chunk: int) -> None:
        """Push the requested chunk into every config the stack reads.

        ``VLAFlowMatching`` and the policy wrapping it both hold a config, and
        which of them is consulted differs by call path. Both are set, and then
        read back: a config object that silently rejects the assignment (a
        frozen dataclass, a property without a setter) would otherwise leave
        the wrapper and the model disagreeing.
        """
        chunk = int(chunk)
        configs = []
        for owner in (self.model, self.policy, self.loaded):
            config = getattr(owner, "config", None)
            if config is not None and config not in configs:
                configs.append(config)
        for config in configs:
            if getattr(config, "chunk_size", None) is None:
                continue
            if int(config.chunk_size) == chunk:
                continue
            try:
                config.chunk_size = chunk
            except Exception as exc:                       # noqa: BLE001
                raise PretrainedError(
                    f"cannot set chunk_size={chunk} on {type(config).__name__}: "
                    f"{exc}. The pretrained checkpoint predicts "
                    f"{self.facts['chunk_size']} actions per call and uses that "
                    "value for its own attention mask and output slicing, so "
                    "an unsynchronised override is not supported. Leave "
                    "actor.chunk_size at 0 to take the checkpoint's value."
                ) from exc
            # n_action_steps is how many of the chunk the policy would execute.
            # It can never exceed the chunk.
            steps = getattr(config, "n_action_steps", None)
            if steps is not None and int(steps) > chunk:
                config.n_action_steps = chunk

        disagree = {type(c).__name__: int(c.chunk_size) for c in configs
                    if getattr(c, "chunk_size", None) is not None
                    and int(c.chunk_size) != chunk}
        if disagree:
            raise PretrainedError(
                f"chunk_size={chunk} was requested but {disagree} still "
                "disagree after assignment; the pretrained stack would build "
                "its attention mask for a different horizon than the one this "
                "policy is supervised on.")

    def shrink_chunk(self, chunk: int) -> float:
        """Generate only the first ``chunk`` actions from here on.

        Lossless because SmolVLA's action tokens attend causally:
        ``embed_suffix`` makes every action its own attention block, so action
        ``i`` reads the prefix and actions ``1..i`` at every denoising step and
        never a later one. A shorter chunk therefore yields the actions a full
        chunk would have started with, and weights imitated at the full length
        serve a shorter one unchanged. That is a property of the pinned
        lerobot's mask, not of flow matching, so it is measured here on this
        expert before it is relied on. Returns the largest velocity difference.

        Growing is refused: positions past the imitated length were never
        supervised.
        """
        chunk, current = int(chunk), int(self.chunk_size)
        if not 1 <= chunk <= current:
            raise PretrainedError(
                f"cannot shrink a {current}-action chunk to {chunk}; only a "
                "shorter chunk keeps the actions the policy was trained on")
        if chunk == current:
            return 0.0
        # A private generator: the run's seeded streams are not advanced.
        generator = torch.Generator().manual_seed(0)
        feat = torch.randn(2, int(self.adapter.feature_dim),
                           generator=generator).to(self.device)
        noisy = torch.randn(2, current, self.action_dim,
                            generator=generator).to(self.device)
        times = torch.rand(2, generator=generator).to(self.device)
        with torch.no_grad():
            cond = self.condition(feat)
            full = self.expert_velocity(noisy, times, cond)[:, :chunk]
            self._synchronise_chunk(chunk)
            try:
                short = self.expert_velocity(noisy[:, :chunk], times, cond)
            except BaseException:
                self._synchronise_chunk(current)
                raise
        difference = float((full - short).abs().max())
        scale = max(float(full.abs().max()), 1.0)
        if not difference <= SHRINK_TOLERANCE * scale:
            self._synchronise_chunk(current)
            raise PretrainedError(
                f"shrinking the chunk from {current} to {chunk} moved the first "
                f"{chunk} velocities by {difference:.3g} (scale {scale:.3g}). "
                "This expert's action tokens read later ones, so a shorter "
                "chunk is a different policy; leave online.chunk_size null.")
        self.chunk_size = chunk
        return difference

    @property
    def device(self) -> torch.device:
        """Where the pretrained weights live. Everything else follows them."""
        return next(self.model.parameters()).device

    # ------------------------------------------------------------- expert set
    def _expert_parameters(self) -> List[nn.Parameter]:
        """The action expert and the projections that read or write actions.

        Taken from the modules the flow path actually uses rather than from a
        name match, so a renamed submodule changes nothing here.
        """
        parameters: List[nn.Parameter] = []
        for name in ("action_in_proj", "action_out_proj", "action_time_mlp_in",
                     "action_time_mlp_out"):
            module = getattr(self.model, name, None)
            if module is not None:
                parameters += list(module.parameters())
        expert = getattr(getattr(self.model, "vlm_with_expert", None),
                         "lm_expert", None)
        if expert is not None:
            parameters += list(expert.parameters())
        return parameters

    def _thaw_expert(self) -> int:
        count = 0
        for parameter in self._expert_parameters():
            if not parameter.requires_grad:
                parameter.requires_grad_(True)
                count += 1
        if count == 0:
            raise PretrainedError(
                "no action-expert parameters were made trainable; the expert "
                "modules were not found on this checkpoint")
        return count

    # ------------------------------------------------------------- language
    def language(self, text: str, batch: int, device=None) -> Tuple[Any, Any]:
        """Tokenize the fixed instruction once and reuse it.

        The instruction does not change within a task, so tokenizing it per
        forward pass is pure overhead; the cache is keyed by the text so a
        second task in the same process cannot inherit the first one's tokens.
        """
        device = self.device
        key = f"{text}|{device}"
        if key not in self._lang_cache:
            tokenizer, path = resolve_attr(
                self.policy, TOKENIZER_PATHS, "text tokenizer")
            self.tokenizer_path = path
            max_length = int(self.facts.get("tokenizer_max_length") or 48)
            encoded = tokenizer(text, padding="max_length", truncation=True,
                                max_length=max_length, return_tensors="pt")
            self._lang_cache[key] = (
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device).bool())
        tokens, mask = self._lang_cache[key]
        return tokens.expand(batch, -1), mask.expand(batch, -1)

    # ---------------------------------------------------------- conditioning
    def condition(self, features: torch.Tensor,
                  instruction: Optional[str] = None) -> Dict[str, Any]:
        """Build the prefix once and cache its keys and values.

        Returned rather than stored on the module: an imagined rollout holds
        several conditionings at once, and a cache on ``self`` would have them
        overwrite each other.
        """
        features = features.to(self.device)
        token = self.adapter(features)                     # (B, 1, token_dim)
        batch = token.shape[0]
        lang_tokens, lang_masks = self.language(
            self.instruction if instruction is None else str(instruction),
            batch)

        # Handed over un-projected either way: in "state_proj" mode
        # embed_prefix's own projection consumes it, and in "embedding" mode
        # that projection is an identity for the call.
        state = token.squeeze(-2)
        embs, pad_masks, att_masks = self._embed_prefix(
            lang_tokens, lang_masks, state)
        past_key_values = self._prefix_cache(embs, pad_masks, att_masks)
        return {"prefix_pad_masks": pad_masks, "past_key_values": past_key_values,
                "state_token": token, "batch": batch}

    @contextlib.contextmanager
    def _state_projection(self):
        """Step over ``state_proj`` when the adapter already emits its output.

        ``embed_prefix`` projects unconditionally, so a token that is already
        ``vlm_hidden_size`` wide has to meet an identity there or it hits a
        ``mat1 and mat2 shapes cannot be multiplied (2x960 and 32x960)``.
        Swapped only for the call, and restored in a finally: the module is
        frozen, so nothing about it changes except what runs during the prefix.
        """
        if self.state_token_mode != "embedding":
            yield
            return
        original = self.model.state_proj
        self.model.state_proj = nn.Identity()
        try:
            yield
        finally:
            self.model.state_proj = original

    def _embed_prefix(self, lang_tokens, lang_masks, state
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Language embeddings plus the state token, with SmolVLA's masks.

        ``embed_prefix`` is called with no images: the observation reached here
        through the world model, and re-encoding pixels with the frozen vision
        tower would be encoding the same frame twice.
        """
        try:
            with self._state_projection():
                return self.model.embed_prefix(
                    images=[], img_masks=[], lang_tokens=lang_tokens,
                    lang_masks=lang_masks, state=state)
        except TypeError as exc:
            raise PretrainedError(
                "embed_prefix rejected the imageless call "
                f"({exc}); lerobot {self.loaded.lerobot_version} may order or "
                "name its arguments differently") from exc

    def _prefix_cache(self, embs, pad_masks, att_masks):
        """Run the prefix through the VLM and keep its key/value cache."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        _prefix_out, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[embs, None],
            use_cache=True, fill_kv_cache=True)
        return past_key_values

    # ------------------------------------------------------------- velocity
    def velocity_fn(self) -> Callable:
        def velocity(x_t: torch.Tensor, t: torch.Tensor, cond: Dict[str, Any]
                     ) -> torch.Tensor:
            return self.expert_velocity(x_t, t, cond)
        return velocity

    def pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Widen a task action chunk to the checkpoint's padded width."""
        if actions.shape[-1] == self.max_action_dim:
            return actions
        pad = self.max_action_dim - actions.shape[-1]
        return torch.cat([actions, actions.new_zeros(*actions.shape[:-1], pad)],
                         dim=-1)

    def action_dim_mask(self, device=None) -> torch.Tensor:
        """One per padded dimension: true where the robot actually acts."""
        mask = torch.zeros(self.max_action_dim, dtype=torch.bool, device=device)
        mask[: self.action_dim] = True
        return mask

    def expert_velocity(self, x_t: torch.Tensor, t: torch.Tensor,
                        cond: Dict[str, Any]) -> torch.Tensor:
        """One denoising step through the real expert.

        ``x_t`` arrives at the task's action width and comes back at it; the
        padding to ``max_action_dim`` exists only between these two lines.
        """
        padded = self.pad_actions(x_t)
        try:
            velocity = self.model.denoise_step(
                cond["prefix_pad_masks"], cond["past_key_values"], padded, t)
        except TypeError as exc:
            raise PretrainedError(
                f"denoise_step rejected its arguments ({exc}); this "
                "integration targets lerobot "
                f"{self.loaded.lerobot_version} where the signature is "
                "(prefix_pad_masks, past_key_values, x_t, timestep)") from exc
        return velocity[..., : x_t.shape[-1]]

    # ----------------------------------------------------------------- report
    def trainable_report(self) -> Dict[str, Any]:
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        return {
            "repo_id": self.loaded.repo_id,
            "revision": self.loaded.revision,
            "lerobot_version": self.loaded.lerobot_version,
            "state_token_mode": self.state_token_mode,
            "tokenizer_path": self.tokenizer_path,
            "chunk_size": self.chunk_size,
            "flow_steps": self.flow_steps,
            "action_dim": self.action_dim,
            "max_action_dim": self.max_action_dim,
            "vlm_hidden_size": self.vlm_hidden,
            "frozen": self.frozen,
            "parameters_total": sum(p.numel() for p in self.parameters()),
            "parameters_trainable": sum(p.numel() for p in self.parameters()
                                        if p.requires_grad),
            "trainable_prefixes": sorted({n.split(".")[0] for n in trainable}),
        }
