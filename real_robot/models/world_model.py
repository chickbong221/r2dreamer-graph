"""The offline world model: the repository's modules, composed rather than inherited.

``Dreamer`` optimises the world model and the imagined actor-critic in one
backward pass. Offline, the world model is trained first and alone, so this
wrapper instantiates the same modules --

    MultiEncoder, GraphEncoder, the semantic RSSM, MultiDecoder,
    SimpleGraphDecoder, the reward head and the continuation head

-- and computes only the world-model objectives, with the same terms, the same
loss scales and the same gradient boundaries (pixels read the semantic state
but do not shape it). No actor, critic or imagination objective is built.

What changes is only what recorded data requires:

* Encoder inputs come from an explicit allowlist, so rewards, outcomes and
  annotation metadata cannot reach the encoder by name.
* Graph conditioning is masked by explicit observation validity. The online
  path masks on ``is_last`` because a simulator auto-reset makes the last frame
  stale; a final recorded frame is a real observation.
* Every loss is a mean over valid positions: padding after an episode's end,
  frames after a terminal state and burn-in positions are never supervised.
* Recurrent state is rebuilt from a burn-in segment rather than stored.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import networks
import rssm as rssm_module
from graph import GraphEncoder, SimpleGraphDecoder, graph_from
from optim import LaProp, clip_grad_agc_
from tools import rpad

from ..common import IdentityError, file_sha256, repo_path, stable_hash

BOOL_KEYS = ("is_first", "is_terminal", "obs_valid", "graph_valid", "cont_valid", "learn", "reward_in_valid")


def compose_model_config(wm_cfg: Mapping[str, Any], manifest, device: str):
    """The repository presets, overrides, and every size read from the dataset."""
    from omegaconf import OmegaConf

    base = OmegaConf.load(repo_path(wm_cfg["model"]["base"]))
    preset = OmegaConf.load(repo_path(wm_cfg["model"]["preset"]))
    for node in (base, preset):
        if "defaults" in node:
            del node["defaults"]
    model = OmegaConf.merge(base, preset, OmegaConf.create(dict(wm_cfg["model"]["overrides"])))
    root = OmegaConf.create({
        "device": str(device),
        "env": {"encoder": {"mlp_keys": "^state$", "cnn_keys": "^image_"},
                "decoder": {"mlp_keys": "^state$", "cnn_keys": "^image_"},
                "progress_mode": "ee_target"},
        "model": model,
    })
    OmegaConf.resolve(root)
    model = root.model
    graph = manifest.data["graph"]
    sizes = graph["vocab_sizes"]
    model.graph.enabled = True
    model.graph.entity_vocab = int(sizes["entity_vocab"])
    model.graph.n_rel = int(sizes["n_rel"])
    model.graph.n_abs = int(sizes["n_abs"])
    model.graph.n_temp = int(sizes["n_temp"])
    model.graph.n_cams = int(graph["n_cams"])
    model.graph.n_max = int(graph["n_max"])
    model.graph.e_max = int(graph["e_max"])
    model.graph.centroid_origin = [float(v) for v in graph["centroid_origin"]]
    model.graph.centroid_scale = graph["centroid_scale"]
    if model.rep_loss != "dreamer":
        raise ValueError("the graph branch is a DreamerV3 extension; model.rep_loss must be dreamer")
    return model


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (values.float() * mask).sum() / mask.sum().clamp_min(1.0)


def mode_onehot(logit: torch.Tensor) -> torch.Tensor:
    return F.one_hot(logit.argmax(-1), logit.shape[-1]).to(torch.float32)


def to_torch(batch: Mapping[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    """Host arrays to device tensors; images stay uint8 until the model reads them."""
    out = {}
    for key, value in batch.items():
        tensor = torch.as_tensor(np.asarray(value))
        if key in BOOL_KEYS:
            tensor = tensor.bool()
        elif tensor.dtype == torch.float64:
            tensor = tensor.float()
        out[key] = tensor.to(device, non_blocking=True)
    return out


class OfflineWorldModel(nn.Module):
    def __init__(self, config, manifest, observation_keys: Mapping[str, Any]):
        super().__init__()
        self.config = config
        self.image_keys = list(observation_keys["images"])
        self.state_key = str(observation_keys["state"])
        if self.image_keys != manifest.image_keys:
            raise ValueError(f"observation_keys.images {self.image_keys} != dataset cameras {manifest.image_keys}")
        height, width = (int(v) for v in manifest.data["model_inputs"]["image_size"])
        # The only observation shapes that exist for the encoder and decoder.
        shapes = {key: (height, width, 3) for key in self.image_keys}
        shapes["state"] = (manifest.state_dim,)
        self.act_dim = manifest.action_dim

        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.graph_encoder = GraphEncoder(config.graph)
        self.graph_dim = int(config.graph.semantic_dim)
        self.rssm = rssm_module.RSSM(config.rssm, self.encoder.out_dim, self.act_dim, semantic=True,
                                     graph_token_size=int(self.graph_encoder.units), graph_dim=self.graph_dim)
        self.decoder = networks.MultiDecoder(config.decoder, self.rssm._deter, self.rssm.flat_stoch, dict(shapes),
                                             flat_sem=self.rssm.flat_sem, detach_sem_cnn=True)
        self.graph_decoder = SimpleGraphDecoder(config.graph, self.graph_dim)
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        scales = dict(config.loss_scales)
        recon = scales.pop("recon")
        scales.update({key: recon for key in self.decoder.all_keys})
        self.loss_scales = {key: float(value) for key, value in scales.items()}
        self.kl_free = float(config.kl_free)
        self.graph_amplitude = float(self.loss_scales.get("graphamp", 0.0)) != 0.0
        amp = str(config.amp_dtype)
        self.amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp]

    # ------------------------------------------------------------ inputs
    def observation(self, part: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Allowlisted encoder/decoder inputs: images in [0, 1] and the state."""
        out = {key: part[key].to(torch.float32) / 255.0 for key in self.image_keys}
        out["state"] = part[self.state_key].to(torch.float32)
        return out

    @property
    def feat_size(self) -> int:
        return int(self.rssm.feat_size)

    def split_feat(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inverse of ``RSSM.get_feat``: ``[stoch, sem, deter]``."""
        r = self.rssm
        stoch = feat[..., : r.flat_stoch].reshape(*feat.shape[:-1], r._stoch, r._discrete)
        sem = feat[..., r.flat_stoch: r.flat_stoch + r.flat_sem]
        deter = feat[..., r.flat_stoch + r.flat_sem:]
        return stoch, sem, deter

    # ------------------------------------------------------------ losses
    def compute_losses(self, data: Mapping[str, torch.Tensor], burn_in: int
                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        B, L = data["is_first"].shape
        ctx = int(burn_in)

        def part(start: int, end: int) -> Dict[str, torch.Tensor]:
            return {key: value[:, start:end] for key, value in data.items()}

        initial = self.rssm.initial(B)
        if ctx > 0:
            head = part(0, ctx)
            with torch.no_grad():
                embed = self.encoder(self.observation(head))
                token = self.graph_encoder(graph_from(head)).token
                token = token * head["graph_valid"].unsqueeze(-1).to(token.dtype)
                observed = self.rssm.observe(embed, head["prev_action"].float(), initial, head["is_first"], token)
            initial = tuple(v[:, -1].detach() for v in (observed[0], observed[1], observed[3]))
        learn = part(ctx, L)
        obs = learn["obs_valid"]
        graph_valid = learn["graph_valid"] & obs

        embed = self.encoder(self.observation(learn))
        encoding = self.graph_encoder(graph_from(learn))
        token = encoding.token * graph_valid.unsqueeze(-1).to(encoding.token.dtype)
        post_stoch, post_deter, post_logit, post_sem = self.rssm.observe(
            embed, learn["prev_action"].float(), initial, learn["is_first"], token)

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        _, prior_logit = self.rssm.prior(post_deter, post_sem)
        dyn, rep = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = masked_mean(dyn, obs)
        losses["rep"] = masked_mean(rep, obs)

        prior_sem = self.rssm.semantic_prior_seq(post_deter)
        sem_dyn, sem_rep = self.rssm.semantic_align_loss(post_sem, prior_sem)
        losses["graphdyn"] = masked_mean(sem_dyn, graph_valid)
        losses["graphrep"] = masked_mean(sem_rep, graph_valid)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.graph_amplitude):
            amp_error, prior_rms, post_rms = self.rssm.semantic_amplitude_loss(post_sem, prior_sem)
            amp_loss = masked_mean(amp_error, graph_valid)
        if self.graph_amplitude:
            losses["graphamp"] = amp_loss
        graph_losses, graph_metrics = self.graph_decoder(post_sem, encoding.compact, graph_valid)
        losses.update(graph_losses)
        metrics.update({f"graph/{k}": v for k, v in graph_metrics.items()})

        feat = self.rssm.get_feat(post_stoch, post_deter, post_sem)
        targets = self.observation(learn)
        for key, dist in self.decoder(post_stoch, post_deter, post_sem).items():
            losses[key] = masked_mean(-dist.log_prob(targets[key]), obs)

        reward_mask = learn["reward_in_valid"] & obs
        reward_dist = self.reward(feat)
        losses["rew"] = masked_mean(-reward_dist.log_prob(learn["reward_in"].float().unsqueeze(-1)), reward_mask)
        cont_mask = learn["cont_valid"] & obs
        cont_target = 1.0 - learn["is_terminal"].float()
        cont_dist = self.cont(feat)
        losses["con"] = masked_mean(-cont_dist.log_prob(cont_target.unsqueeze(-1)), cont_mask)

        total = sum(self.loss_scales[key] * value for key, value in losses.items())
        with torch.no_grad():
            predicted = reward_dist.mode().squeeze(-1)
            metrics["reward/mae"] = masked_mean((predicted - learn["reward_in"].float()).abs(), reward_mask)
            probability = cont_dist.mean.squeeze(-1)
            metrics["cont/accuracy"] = masked_mean(((probability > 0.5).float() == cont_target).float(), cont_mask)
            metrics["cont/terminal_frames"] = (learn["is_terminal"] & cont_mask).float().sum()
            metrics["kl/dyn_entropy"] = masked_mean(self.rssm.get_dist(prior_logit).entropy(), obs)
            metrics["kl/rep_entropy"] = masked_mean(self.rssm.get_dist(post_logit).entropy(), obs)
            metrics["graph/sem_prior_rms"] = masked_mean(prior_rms, graph_valid)
            metrics["graph/sem_post_rms"] = masked_mean(post_rms, graph_valid)
            metrics["graph/valid_fraction"] = graph_valid.float().mean()
            metrics["data/obs_valid_fraction"] = obs.float().mean()
        metrics.update({f"loss/{key}": value.detach() for key, value in losses.items()})
        metrics["loss/model"] = total.detach()
        posterior = {"stoch": post_stoch.detach(), "deter": post_deter.detach(), "sem": post_sem.detach()}
        return total, metrics, posterior

    # --------------------------------------------------------- inference
    def posterior_step(self, stoch, deter, sem, prev_action, embed, graph_token, reset, sample: bool = False):
        """``RSSM.obs_step`` with a choice of how ``z`` is drawn.

        The latent cache and the robot wrapper must agree on one convention;
        ``sample=False`` takes the posterior mode, which makes both
        deterministic given the observations.
        """
        r = self.rssm
        reset = reset.bool()

        def clear(x):
            return torch.where(rpad(reset, x.dim() - int(reset.dim())), torch.zeros_like(x), x)

        stoch, deter, sem, prev_action = clear(stoch), clear(deter), clear(sem), clear(prev_action)
        deter = r._deter_net(stoch, deter, prev_action, sem)
        sem = r._sem_obs(torch.cat([deter, graph_token], -1))
        logit = r._obs_net(torch.cat([deter, embed], -1))
        stoch = r.get_dist(logit).rsample() if sample else mode_onehot(logit)
        return stoch, deter, sem

    def prior_step(self, stoch, deter, sem, action, sample: bool = False):
        """``RSSM.img_step`` with the same choice for ``z``."""
        r = self.rssm
        deter = r._deter_net(stoch, deter, action, sem)
        sem = r._sem_img(deter)
        logit = r._img_net(deter)
        stoch = r.get_dist(logit).rsample() if sample else mode_onehot(logit)
        return stoch, deter, sem

    @torch.no_grad()
    def encode_sequence(self, data: Mapping[str, torch.Tensor], sample: bool = False) -> Dict[str, torch.Tensor]:
        """Chronological posterior over ``(1, T)`` or ``(B, T)`` batches, in float32."""
        B, T = data["is_first"].shape
        embed = self.encoder(self.observation(data))
        token = self.graph_encoder(graph_from(data)).token
        token = token * (data["graph_valid"] & data["obs_valid"]).unsqueeze(-1).to(token.dtype)
        stoch, deter, sem = self.rssm.initial(B)
        feats = []
        for t in range(T):
            stoch, deter, sem = self.posterior_step(stoch, deter, sem, data["prev_action"][:, t].float(),
                                                    embed[:, t], token[:, t], data["is_first"][:, t], sample)
            feats.append(self.rssm.get_feat(stoch, deter, sem))
        feat = torch.stack(feats, 1)
        return {"feat": feat, "reward": self.reward(feat).mode().squeeze(-1),
                "cont": self.cont(feat).mean.squeeze(-1)}

    @torch.no_grad()
    def imagine(self, feat: torch.Tensor, action: torch.Tensor, sample: bool = False) -> Dict[str, torch.Tensor]:
        """One prior step from cached features; reward and continuation of the arrival."""
        stoch, sem, deter = self.split_feat(feat.float())
        stoch, deter, sem = self.prior_step(stoch, deter, sem, action.float(), sample)
        nxt = self.rssm.get_feat(stoch, deter, sem)
        return {"feat": nxt, "reward": self.reward(nxt).mode().squeeze(-1),
                "cont": self.cont(nxt).mean.squeeze(-1)}

    @torch.no_grad()
    def decode(self, feat: torch.Tensor) -> Dict[str, Any]:
        """Observation distributions for ``(B, F)`` features, with a time axis of one: ``(B, 1, ...)``."""
        stoch, sem, deter = self.split_feat(feat.float())
        return self.decoder(stoch.unsqueeze(1), deter.unsqueeze(1), sem.unsqueeze(1))


def make_optimizer(model: OfflineWorldModel, config):
    """LaProp with adaptive gradient clipping and linear warm-up, as ``Dreamer``."""
    from torch.optim.lr_scheduler import LambdaLR

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = LaProp(params, lr=float(config.lr), betas=(float(config.beta1), float(config.beta2)),
                       eps=float(config.eps))
    warmup = int(config.warmup)
    scheduler = LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)

    def clip():
        clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

    return optimizer, scheduler, clip


def world_model_identity(manifest, model_config, wm_cfg: Mapping[str, Any], partial_dataset: bool = False
                         ) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(model_config, resolve=True)
    container.pop("device", None)
    for section in container.values():
        if isinstance(section, dict):
            section.pop("device", None)
    identity = manifest.identity
    return {
        "dataset": manifest.dataset_key(),
        "model": stable_hash(container),
        "observation_keys": dict(wm_cfg["observation_keys"]),
        "vocab": identity["vocab"],
        "reward": identity["reward"],
        "action_mapping": identity["action_mapping"],
        "action_transform": identity["action_transform"],
        "annotation_mode": identity["annotation_mode"],
        "selection": identity["selection"],
        "training_episodes": stable_hash(manifest.training_episodes()),
        "partial_dataset": bool(partial_dataset),
    }


def weights_digest(state: Mapping[str, Any]) -> str:
    """SHA-256 over every tensor's name, dtype, shape and bytes, in name order.

    Two checkpoints with this digest hold the same weights whatever else their
    files contain; any change to any weight changes it.
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().to("cpu").contiguous()
            digest.update(f"{tensor.dtype}{tuple(tensor.shape)}".encode("utf-8"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def freeze(model: nn.Module) -> nn.Module:
    """No gradients and evaluation mode: nothing downstream can train it."""
    return model.requires_grad_(False).eval()


def _with_device(node: Any, device: str) -> Any:
    """A saved config with every ``device`` field pointed at this process's device."""
    if isinstance(node, dict):
        return {key: (device if key == "device" else _with_device(value, device)) for key, value in node.items()}
    if isinstance(node, list):
        return [_with_device(value, device) for value in node]
    return node


def load_world_model(checkpoint_path: str, device, expected_dataset: Optional[Mapping[str, str]] = None,
                     expected_weights: Optional[Mapping[str, str]] = None):
    """``(model, payload, manifest)`` from a world-model checkpoint, identity checked, frozen.

    ``expected_weights`` (``{"sha256": ..., "weights": ...}``) pins the exact
    checkpoint: a latent cache or a rollout set built from other weights -- even
    with the same architecture and configuration -- is refused.
    """
    from omegaconf import OmegaConf
    from ..data.manifest import DatasetManifest

    if expected_weights is not None:
        found = file_sha256(checkpoint_path)
        if found != expected_weights["sha256"]:
            raise IdentityError(
                f"{checkpoint_path} is not the checkpoint this artifact was built from "
                f"(file sha256 {found[:12]} vs {str(expected_weights['sha256'])[:12]}); "
                "re-encode the latent cache and regenerate imagined transitions"
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if expected_weights is not None:
        digest = weights_digest(payload["state"]["model"])
        if digest != expected_weights["weights"]:
            raise IdentityError(f"{checkpoint_path}: weights digest {digest[:12]} differs from "
                                f"{str(expected_weights['weights'])[:12]}")
    meta = payload["meta"]
    manifest = DatasetManifest.load(meta["dataset_root"])
    if manifest.dataset_key() != payload["identity"]["dataset"]:
        raise IdentityError(
            f"{checkpoint_path} was trained on dataset {payload['identity']['dataset']}, but "
            f"{meta['dataset_root']} is now {manifest.dataset_key()} (rebuilt or changed since)"
        )
    if expected_dataset is not None and expected_dataset != manifest.dataset_key():
        raise IdentityError(f"{checkpoint_path} does not belong to dataset {expected_dataset}")
    config = OmegaConf.create(_with_device(meta["model_config"], str(device)))
    model = OfflineWorldModel(config, manifest, meta["observation_keys"])
    model.load_state_dict(payload["state"]["model"], strict=True)
    return freeze(model.to(device)), payload, manifest
