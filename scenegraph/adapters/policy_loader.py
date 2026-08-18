"""Load and run MS-HAB checkpoints, matching mshab/evaluate.py exactly.

Supports PPO and SAC (the two RL algos in the mshab checkpoint release). The
algorithm is auto-detected from the checkpoint's config.yml ``algo.name``.

Verified against installed mshab source:

PPO  (mshab.agents.ppo.Agent):
    ctor   : Agent(sample_obs, single_act_shape)
    action : agent.get_action(obs, deterministic=True)            -> tensor
    obs    : the whole wrapped obs dict (state + pixels)

SAC  (mshab.agents.sac.Agent):
    ctor   : Agent(model_pixel_obs_space, state_shape, act_shape, **algo_cfg)
             where model_pixel_obs_space reshapes each framestacked pixel space
             from (stack, C, H, W) -> (stack*C, H, W)   [evaluate.py]
    action : agent.actor(obs["pixels"], obs["state"],
                         compute_pi=False, compute_log_pi=False)[0]
    obs    : obs["pixels"] (dict of stacked depth) + obs["state"]

Both load weights from torch.load(path)["agent"].

The wrapped MS-HAB env (FetchDepthObservationWrapper + FrameStack) already
produces the obs both agents expect, including the nested ``pixels`` key SAC
reads -- so no extra obs restructuring is needed here.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np


class PolicyHandle:
    def __init__(self, fn: Callable[[Any], np.ndarray], kind: str):
        self._fn = fn
        self.kind = kind

    def act(self, obs: Any) -> np.ndarray:
        return self._fn(obs)


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def find_checkpoint(ckpt_dir: str) -> str:
    for name in ("policy.pt", "latest.pt", "best.pt", "ckpt.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    if os.path.isdir(ckpt_dir):
        for f in sorted(os.listdir(ckpt_dir)):
            if f.endswith(".pt"):
                return os.path.join(ckpt_dir, f)
    return ""


def detect_algo(ckpt_dir: str) -> Optional[str]:
    """Read algo.name from the checkpoint's config.yml ('ppo' | 'sac' | ...).

    Returns None and prints a clear warning if the directory is missing,
    config.yml is missing, the YAML is unreadable, or algo.name is absent.
    Silent None used to land users in the random-policy fallback without any
    hint that --ckpt-dir was wrong (e.g. a Windows path passed to Linux).
    """
    if not os.path.isdir(ckpt_dir):
        print(f"[policy] WARN: --ckpt-dir {ckpt_dir!r} is not a directory "
              f"(cwd={os.getcwd()!r}) -- will fall back to random actions")
        return None
    cfg_path = os.path.join(ckpt_dir, "config.yml")
    if not os.path.exists(cfg_path):
        print(f"[policy] WARN: no config.yml in {ckpt_dir!r} "
              f"-- will fall back to random actions")
        return None
    try:
        import yaml
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        print(f"[policy] WARN: failed to read {cfg_path}: {exc!r} "
              f"-- will fall back to random actions")
        return None
    algo = (raw or {}).get("algo", {}).get("name")
    if algo is None:
        print(f"[policy] WARN: {cfg_path} has no algo.name field "
              f"-- will fall back to random actions")
    return algo


def _algo_cfg(ckpt_dir: str):
    """Return the algo sub-config as an attribute-access object (OmegaConf)."""
    cfg_path = os.path.join(ckpt_dir, "config.yml")
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(cfg_path).algo
    except Exception:
        import yaml
        from types import SimpleNamespace
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        return SimpleNamespace(**raw["algo"])


# --------------------------------------------------------------------------- #
# Obs helpers
# --------------------------------------------------------------------------- #
def _to_tensor_obs(obs, device):
    import torch
    if isinstance(obs, dict):
        return {k: _to_tensor_obs(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device, dtype=torch.float)
    return torch.as_tensor(np.asarray(obs), device=device, dtype=torch.float)


# --------------------------------------------------------------------------- #
# Public entry: dispatch by algo
# --------------------------------------------------------------------------- #
def load_policy(ckpt_dir: str, venv, eval_obs, device: str = "cuda") -> PolicyHandle:
    algo = detect_algo(ckpt_dir)
    print(f"[policy] detected algo = {algo}")
    if algo == "ppo":
        return load_ppo_policy(ckpt_dir, venv, eval_obs, device)
    if algo == "sac":
        return load_sac_policy(ckpt_dir, venv, eval_obs, device)
    print(f"[policy] algo {algo!r} unsupported here -> random actions")
    return _random_policy(venv)


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
def load_ppo_policy(ckpt_dir, venv, eval_obs, device="cuda") -> PolicyHandle:
    ckpt = find_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else ckpt_dir
    if not ckpt or not os.path.exists(ckpt):
        print(f"[policy] no checkpoint under {ckpt_dir!r} -> random actions")
        return _random_policy(venv)
    try:
        import torch
        from mshab.agents.ppo import Agent as PPOAgent

        act_shape = venv.unwrapped.single_action_space.shape
        obs_t = _to_tensor_obs(eval_obs, device)
        policy = PPOAgent(obs_t, act_shape)
        policy.eval()
        state = torch.load(ckpt, map_location=device)
        policy.load_state_dict(state["agent"] if "agent" in state else state)
        policy.to(device)

        def _act(obs):
            o = _to_tensor_obs(obs, device)
            with torch.no_grad():
                a = policy.get_action(o, deterministic=True)
            return a.detach().cpu().numpy()

        print(f"[policy] loaded mshab PPO Agent from {ckpt}")
        return PolicyHandle(_act, kind="ppo")
    except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"[policy] PPO load failed ({exc}); random actions")
        return _random_policy(venv)


# --------------------------------------------------------------------------- #
# SAC  (mirrors evaluate.py sac branch)
# --------------------------------------------------------------------------- #
def load_sac_policy(ckpt_dir, venv, eval_obs, device="cuda") -> PolicyHandle:
    ckpt = find_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else ckpt_dir
    if not ckpt or not os.path.exists(ckpt):
        print(f"[policy] no checkpoint under {ckpt_dir!r} -> random actions")
        return _random_policy(venv)
    try:
        import torch
        from gymnasium import spaces
        from mshab.agents.sac import Agent as SACAgent

        algo_cfg = _algo_cfg(ckpt_dir)
        uenv = venv.unwrapped
        obs_space = uenv.single_observation_space
        act_space = uenv.single_action_space

        # Reshape framestacked pixel spaces (stack, C, H, W) -> (stack*C, H, W).
        pixels_obs_space = obs_space["pixels"]
        state_obs_space = obs_space["state"]
        model_pixel_obs_space = dict()
        for k, space in pixels_obs_space.items():
            shape, low, high, dtype = space.shape, space.low, space.high, space.dtype
            if len(shape) == 4:
                shape = (shape[0] * shape[1], shape[-2], shape[-1])
                low = low.reshape((-1, *low.shape[-2:]))
                high = high.reshape((-1, *high.shape[-2:]))
            model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
        model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)

        policy = SACAgent(
            model_pixel_obs_space,
            state_obs_space.shape,
            act_space.shape,
            actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
            critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
            critic_layer_norm=algo_cfg.critic_layer_norm,
            critic_dropout=algo_cfg.critic_dropout,
            encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
            encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
            cnn_features=list(algo_cfg.cnn_features),
            cnn_filters=list(algo_cfg.cnn_filters),
            cnn_strides=list(algo_cfg.cnn_strides),
            cnn_padding=algo_cfg.cnn_padding,
            log_std_min=algo_cfg.actor_log_std_min,
            log_std_max=algo_cfg.actor_log_std_max,
            device=device,
        )
        policy.eval()
        state = torch.load(ckpt, map_location=device)
        policy.load_state_dict(state["agent"] if "agent" in state else state)
        policy.to(device)

        def _act(obs):
            # Pass obs["pixels"] through unmodified -- matches evaluate.py:328-333
            # exactly. SharedCNN.forward handles the 5D (B,stack,C,H,W) ->
            # 4D (B,stack*C,H,W) collapse internally AND calls .contiguous(),
            # which is needed because FrameStack hands back a non-contiguous
            # stack view; flattening outside the model skips that guard and
            # later trips Flatten's .view() on cuDNN's channels_last output.
            o = _to_tensor_obs(obs, device)
            with torch.no_grad():
                a = policy.actor(
                    o["pixels"], o["state"],
                    compute_pi=False, compute_log_pi=False,
                )[0]
            return a.detach().cpu().numpy()

        print(f"[policy] loaded mshab SAC Agent from {ckpt}")
        return PolicyHandle(_act, kind="sac")
    except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"[policy] SAC load failed ({exc}); random actions")
        return _random_policy(venv)


def _random_policy(venv) -> PolicyHandle:
    space = venv.action_space

    def _act(obs):
        return space.sample()

    return PolicyHandle(_act, kind="random")
