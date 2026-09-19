"""Building SOLD's components, from SOLD's classes and SOLD's numbers.

Nothing here is an architecture decision. The classes come out of the vendored
tree and the widths come out of ``sold/configs``; what this module does is call
them with the dataset's image size and action width in the places the config
files leave as ``'???'``.

Two ways to build the world model:

``components``   the individual modules -- SAVi, the OCVP dynamics, the reward,
                 actor and critic heads. This is what the offline stages use,
                 and it needs only torch, so it can be built and tested without
                 Lightning.
``sold_module``  the real ``SOLDModule``, for Stage 3. It is a
                 ``LightningModule`` and needs an environment at construction
                 (for the action space and the episode length), so it also
                 needs Lightning, gym and Hydra to be installed.

The included ``checkpoints/`` in the SOLD tree are SAVi and SOLD models for
``reach_red`` and ``push_red`` -- multi-object-fetch tasks, at 7 slots and a
different action space. They are **not** a starting point for a ManiSkill task
and nothing here loads them; Stage 1A pretrains SAVi on the target
demonstrations.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence

from ..vendor import SOLD as VENDOR


def _strip(node: Mapping[str, Any]) -> Dict[str, Any]:
    """A config block without Hydra's own keys or its unfilled placeholders.

    ``'???'`` is Hydra's "the dataset supplies this" -- the image size and the
    action width. The builders pass those explicitly, so leaving the literal
    in would either collide with the keyword or be handed to a constructor as
    a string.
    """
    return {k: v for k, v in dict(node or {}).items()
            if not k.startswith("_") and v != "???"}


def build_autoencoder(spec: Mapping[str, Any], *, image_size: Sequence[int],
                      action_dim: int):
    """SAVi, at the widths ``configs/autoencoder/savi.yaml`` states."""
    with VENDOR.active():
        from modeling.autoencoder.savi.autoencoder import SAVi
        from modeling.autoencoder.savi.corrector import Corrector
        from modeling.autoencoder.savi.decoder import SaviCnnDecoder
        from modeling.autoencoder.savi.encoder import SaviCnnEncoder
        from modeling.autoencoder.savi import initializer as initializers
        from modeling.autoencoder.savi import predictor as predictors

        size = (int(image_size[0]), int(image_size[1]))
        corrector = Corrector(**_strip(spec["corrector"]))
        encoder = SaviCnnEncoder(image_size=size, **_strip(spec["encoder"]))
        decoder = SaviCnnDecoder(image_size=size, **_strip(spec["decoder"]))

        init_name = str(spec["initializer"]["_target_"]).rsplit(".", 1)[-1]
        initializer = getattr(initializers, init_name)(
            **_strip(spec["initializer"]))

        predictor_name = str(spec["predictor"]["_target_"]).rsplit(".", 1)[-1]
        predictor_kwargs = _strip(spec["predictor"])
        predictor_kwargs["action_dim"] = int(action_dim)
        predictor = getattr(predictors, predictor_name)(**predictor_kwargs)

        return SAVi(corrector, predictor, encoder, decoder, initializer)


def build_dynamics(world: Mapping[str, Any], *, num_slots: int, slot_dim: int,
                   action_dim: int, imagination_horizon: int,
                   sequence_length: int):
    """The OCVP-Seq slot dynamics, wrapped in its autoregressive driver."""
    with VENDOR.active():
        from modeling.sold.dynamics import make_ocvp_seq_dynamics_model

        spec = _strip(world["dynamics_predictor"])
        if bool(spec.get("teacher_forcing", False)):
            raise SystemExit(
                "dynamics_predictor.teacher_forcing is on. In its batched "
                "form that feeds the *ground-truth* future slots into the "
                "rollout, so the dynamics loss is computed against a model "
                "that saw the frames it was asked to predict, and imagination "
                "at run time has no such frames. Upstream ships it off; "
                "turning it on for training speed is future-observation "
                "leakage.")
        return make_ocvp_seq_dynamics_model(
            num_slots=int(num_slots), slot_dim=int(slot_dim),
            sequence_length=int(imagination_horizon),
            action_dim=int(action_dim), input_buffer_size=int(sequence_length),
            **spec)


def build_heads(world: Mapping[str, Any], *, num_slots: int, slot_dim: int,
                action_dim: int, max_episode_steps: int,
                action_low=None, action_high=None):
    """The reward head, the Gaussian actor, the critic and its target."""
    with VENDOR.active():
        from modeling.sold import prediction

        infos = {"max_episode_steps": int(max_episode_steps),
                 "num_slots": int(num_slots), "slot_dim": int(slot_dim)}

        def make(block):
            name = str(block["_target_"]).rsplit(".", 1)[-1]
            return getattr(prediction, name), _strip(block)

        actor_cls, actor_kwargs = make(world["actor"])
        critic_cls, critic_kwargs = make(world["critic"])
        reward_cls, reward_kwargs = make(world["reward_predictor"])

        low = [-1.0] * int(action_dim) if action_low is None else list(action_low)
        high = [1.0] * int(action_dim) if action_high is None else list(action_high)
        actor = actor_cls(**infos, **actor_kwargs, output_dim=int(action_dim),
                          lower_bound=low, upper_bound=high)
        critic = critic_cls(**infos, **critic_kwargs)
        reward = reward_cls(**infos, **reward_kwargs)
        return {"actor": actor, "critic": critic,
                "critic_target": copy.deepcopy(critic), "reward": reward}


def components(world: Mapping[str, Any], *, image_size: Sequence[int],
               action_dim: int, max_episode_steps: int,
               action_low=None, action_high=None) -> Dict[str, Any]:
    """Every module SOLD trains, built without Lightning."""
    from .config import context_bounds

    autoencoder = build_autoencoder(world["autoencoder_spec"],
                                    image_size=image_size,
                                    action_dim=int(action_dim))
    _low, high_context = context_bounds(world)
    horizon = int(world.get("imagination_horizon", 15))
    dynamics = build_dynamics(world, num_slots=autoencoder.num_slots,
                              slot_dim=autoencoder.slot_dim,
                              action_dim=int(action_dim),
                              imagination_horizon=horizon,
                              sequence_length=horizon + high_context)
    heads = build_heads(world, num_slots=autoencoder.num_slots,
                        slot_dim=autoencoder.slot_dim,
                        action_dim=int(action_dim),
                        max_episode_steps=int(max_episode_steps),
                        action_low=action_low, action_high=action_high)
    return {"autoencoder": autoencoder, "dynamics": dynamics, **heads,
            "num_slots": autoencoder.num_slots,
            "slot_dim": autoencoder.slot_dim,
            "sequence_length": horizon + high_context}


def build_adapter(smolvla: Mapping[str, Any], *, num_slots: int, slot_dim: int,
                  token_dim: int, context: int,
                  max_episode_steps: Optional[int] = None):
    """The actor-side adapter over the causal slot history.

    ``max_episode_steps`` defaults to ``context`` because that is the longest
    history this head ever sees -- it reads a fixed window everywhere. The
    ALiBi mask is ``(heads, steps * (slots + 1), steps * (slots + 1))``, so
    sizing it for a 150-step episode when it will only ever see three frames
    would cost tens of megabytes to hold a mask that is never indexed.
    """
    from .adapter import SlotHistoryAdapter

    spec = dict(smolvla.get("adapter") or {})
    return SlotHistoryAdapter(
        num_slots=int(num_slots), slot_dim=int(slot_dim),
        token_dim=int(token_dim), context=int(context),
        max_episode_steps=int(max_episode_steps or context),
        head_token_dim=int(spec.get("token_dim", 256)),
        hidden_dim=int(spec.get("hidden_dim", 512)),
        num_heads=int(spec.get("num_heads", 8)),
        num_layers=int(spec.get("num_layers", 3)),
        num_mlp_layers=int(spec.get("num_mlp_layers", 1)))


def build_actor(smolvla: Mapping[str, Any], *, num_slots: int, slot_dim: int,
                action_dim: int, context: int, device="cuda"):
    """The pretrained SmolVLA with the slot-history adapter as its front end.

    The adapter is handed to ``SmolVLAActor`` as *its* adapter, so the whole
    conditioning path -- adapter, state token, prefix, key/value cache -- is
    the one code path, used in imitation, in imagination and at inference.
    """
    from ...models.pretrained import load_policy, model_facts
    from ...models.smolvla_actor import SmolVLAActor
    from ..latent_actor import LatentActor

    loaded = load_policy(str(smolvla["pretrained"]),
                         str(smolvla.get("revision") or "main"))
    loaded.policy.to(device)
    facts = model_facts(loaded)
    mode = str(smolvla.get("state_token_mode") or "embedding")
    token_dim = int(facts["vlm_hidden_size"] if mode == "embedding"
                    else facts["max_state_dim"])
    adapter = build_adapter(smolvla, num_slots=num_slots, slot_dim=slot_dim,
                            token_dim=token_dim, context=context)
    inner = SmolVLAActor(
        loaded, adapter, action_dim=int(action_dim),
        chunk_size=int(smolvla.get("chunk_size") or 0) or None,
        flow_steps=int(smolvla.get("flow_steps") or 0) or None,
        instruction=str(smolvla.get("instruction") or ""),
        state_token_mode=mode)
    return LatentActor(inner, instruction=str(smolvla.get("instruction") or ""))


def parameters(parts: Mapping[str, Any], *, actor=None) -> Dict[str, Any]:
    """Every component, counted once, with SmolVLA outside the budget."""
    from ..params import report, split

    autoencoder = parts["autoencoder"]
    listed = [
        ("autoencoder.encoder", autoencoder.encoder),
        ("autoencoder.decoder", autoencoder.decoder),
        ("autoencoder.corrector", autoencoder.corrector),
        ("autoencoder.predictor", autoencoder.predictor),
        ("autoencoder.initializer", autoencoder.initializer),
        ("autoencoder(rest)", autoencoder),
        ("dynamics", parts["dynamics"]),
        ("reward", parts["reward"]),
        ("policy_prior(gaussian actor)", parts["actor"]),
        ("critic", parts["critic"]),
        ("critic_target", parts["critic_target"]),
    ]
    if actor is not None:
        listed.append(("adapter", actor.adapter))
        listed.append(("smolvla", actor))
    full = report(listed)
    return split(full, exclude=("smolvla",) if actor is not None else ())


def sold_module(cfg: Mapping[str, Any], env, *, device="cuda"):
    """The real ``SOLDModule``, for Stage 3.

    Constructed from the same blocks the offline stages use, so the widths
    cannot drift between them. Needs Lightning, gym and Hydra, which is what
    upstream's own entry point needs.
    """
    world = dict(cfg["world_model"])
    with VENDOR.active():
        from functools import partial

        from train_sold import SOLDModule

        spec = world["autoencoder_spec"]
        autoencoder = build_autoencoder(
            spec, image_size=world["env"]["image_size"],
            action_dim=int(env.action_space.shape[0]))

        def dynamics_factory(**kwargs):
            merged = {**_strip(world["dynamics_predictor"]), **kwargs}
            from modeling.sold.dynamics import make_ocvp_seq_dynamics_model

            return make_ocvp_seq_dynamics_model(**merged)

        def head_factory(block):
            name = str(block["_target_"]).rsplit(".", 1)[-1]
            from modeling.sold import prediction

            return partial(getattr(prediction, name), **_strip(block))

        keys = ("max_steps", "num_seed", "update_freq", "num_updates",
                "eval_freq", "num_eval_episodes", "batch_size",
                "buffer_capacity", "save_replay_buffer",
                "dynamics_learning_rate", "dynamics_grad_clip",
                "actor_learning_rate", "actor_grad_clip",
                "actor_entropy_loss_weight", "actor_gradients",
                "critic_learning_rate", "critic_grad_clip",
                "reward_learning_rate", "reward_grad_clip",
                "finetune_autoencoder", "autoencoder_learning_rate",
                "autoencoder_grad_clip", "num_context",
                "imagination_horizon", "start_imagination_from_every",
                "return_lambda", "discount_factor", "critic_ema_decay")
        settings = {key: world[key] for key in keys if key in world}
        module = SOLDModule(
            autoencoder=autoencoder,
            dynamics_predictor=dynamics_factory,
            actor=head_factory(world["actor"]),
            critic=head_factory(world["critic"]),
            reward_predictor=head_factory(world["reward_predictor"]),
            env=env, **settings)
        return module.to(device)
