"""TD-MPC2's own config, assembled without Hydra, plus the integration's.

``sim_vla/tdmpc2/config.yaml`` and ``common/__init__.py:MODEL_SIZE`` are the
real sources for every architecture number, and they are read here rather than
restated: a second copy of ``mlp_dim`` is a second thing to forget to change.
What ``common.parser.parse_cfg`` adds on top -- ``bin_size``, the multitask
flags, the task list -- is recomputed the same way, because ``parse_cfg``
itself cannot run outside a Hydra session (it calls
``hydra.utils.get_original_cwd``).

The result is a :class:`sim_vla.models.model_config.Node`: attribute access for
``cfg.latent_dim`` and the mapping protocol for ``cfg.obs_shape.keys()``,
assignable like the ``DictConfig`` the upstream code expects, so ``WorldModel``
and ``TDMPC2`` are constructed from exactly the object they were written for.

Two blocks are the integration's own and live under ``smolvla`` and ``stages``.
Nothing in them reaches the native code except through the hooks in
``policy.py``; with ``smolvla.enabled: false`` the agent that comes out of here
is the upstream one.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ...config import deep_merge
from ...models.model_config import Node
from ..vendor import TDMPC2 as VENDOR

INTEGRATION_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NATIVE_CONFIG = VENDOR.root / "config.yaml"

# What ``layers.conv`` accepts. Asserted upstream, so a recorded 112x112
# dataset has to be resampled and the size has to be one of these.
SUPPORTED_RENDER_SIZES = (64, 128)

# Settings the integration reports and a checkpoint is compared on. Everything
# here changes what the weights are, not merely how long they trained.
ARCHITECTURE_KEYS = (
    "obs", "include_state", "render_size", "num_cameras",
    "model_size", "num_enc_layers", "enc_dim", "num_channels", "mlp_dim",
    "latent_dim", "simnorm_dim", "num_q", "dropout", "num_bins", "vmin",
    "vmax", "task_dim", "action_dim", "rgb_state_enc_dim",
    "rgb_state_num_enc_layers", "rgb_state_latent_dim", "proprio_dim",
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def model_size_table() -> Dict[int, Dict[str, Any]]:
    """Upstream's own preset table, read from upstream."""
    with VENDOR.active():
        import common

        return dict(common.MODEL_SIZE)


def task_set() -> Dict[str, Any]:
    with VENDOR.active():
        import common

        return dict(common.TASK_SET)


def native_defaults() -> Dict[str, Any]:
    """``sim_vla/tdmpc2/config.yaml`` with the unusable placeholders dropped.

    ``???`` is Hydra's "must be supplied"; leaving the literal string in place
    would let ``cfg.model_size == '???'`` pass an ``is not None`` test and then
    fail three frames deeper inside the preset lookup.
    """
    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if v != "???"}
        return node

    return clean(_read_yaml(NATIVE_CONFIG))


def build_cfg(overrides: Optional[Mapping[str, Any]] = None, *,
              obs_shape: Optional[Mapping[str, Sequence[int]]] = None,
              action_dim: int = 0, episode_length: int = 0) -> Node:
    """A TD-MPC2 config object the upstream classes can be built from.

    ``obs_shape``, ``action_dim`` and ``episode_length`` are what
    ``envs/__init__.py:make_envs`` normally writes into the config after
    building the environment. Offline they come from the dataset instead, and
    the online entry point still lets the environment write them, so the two
    paths agree by construction rather than by a repeated constant.
    """
    cfg = deep_merge(native_defaults(), dict(overrides or {}))

    presets = model_size_table()
    size = cfg.get("model_size")
    if size is not None:
        size = int(size)
        if size not in presets:
            raise SystemExit(
                f"model_size={size} is not one of {sorted(presets)}; these are "
                "upstream's presets and picking a larger one only to approach "
                "a parameter budget is not a sizing decision.")
        cfg["model_size"] = size
        for key, value in presets[size].items():
            cfg[key] = value
    # Explicit overrides win over the preset they sit beside: the preset is a
    # starting point and the sizing decision is allowed to move one width.
    for key, value in (overrides or {}).items():
        if not isinstance(value, dict):
            cfg[key] = value

    tasks = task_set()
    cfg["multitask"] = cfg.get("env_id") in tasks
    if cfg["multitask"]:
        raise SystemExit(
            "the multi-task task sets are not part of this integration; "
            "SmolVLA is conditioned on one task instruction per run.")
    cfg["task_dim"] = 0
    cfg["tasks"] = [cfg.get("env_id")]
    cfg["bin_size"] = (cfg["vmax"] - cfg["vmin"]) / (cfg["num_bins"] - 1)

    render = int(cfg.get("render_size", 64))
    if str(cfg.get("obs")) == "rgb" and render not in SUPPORTED_RENDER_SIZES:
        raise SystemExit(
            f"render_size={render} but layers.conv asserts one of "
            f"{list(SUPPORTED_RENDER_SIZES)}. The demonstrations are resampled "
            "to this size, so pick one upstream supports rather than editing "
            "the encoder.")

    if obs_shape is not None:
        cfg["obs_shape"] = {str(k): tuple(int(x) for x in v)
                            for k, v in obs_shape.items()}
    if action_dim:
        cfg["action_dim"] = int(action_dim)
    if episode_length:
        cfg["episode_length"] = int(episode_length)
        cfg["seed_steps"] = max(1000, int(cfg["num_envs"]) * int(episode_length))

    # What ``parse_cfg`` writes into the ManiSkill blocks. Upstream's
    # ``make_envs`` reads them and writes the control mode and horizon back,
    # so they have to exist before an environment is built.
    for block in ("env_cfg", "eval_env_cfg"):
        node = dict(cfg.get(block) or {})
        node["env_id"] = cfg["env_id"]
        node["obs_mode"] = cfg["obs"]
        node["reward_mode"] = "normalized_dense"
        node["sim_backend"] = cfg["env_type"]
        node.setdefault("partial_reset", False)
        cfg[block] = node
    cfg["env_cfg"]["num_envs"] = int(cfg["num_envs"])
    cfg["eval_env_cfg"]["num_envs"] = int(cfg["num_eval_envs"])
    cfg["eval_env_cfg"]["num_eval_episodes"] = (
        int(cfg["eval_episodes_per_env"]) * int(cfg["num_eval_envs"]))

    # ``work_dir`` is a Path in the upstream config -- ``Logger`` does
    # ``cfg.work_dir / "models"`` -- and is used for logging only. Hydra
    # normally supplies it; here the caller does.
    cfg.setdefault("work_dir", Path.cwd())
    return wrap(cfg)


def wrap(cfg: Mapping[str, Any]):
    """OmegaConf when it is installed, and a plain attribute node otherwise.

    Upstream is written against Hydra's ``DictConfig`` and a couple of places
    depend on it -- ``Logger`` serialises the whole config through
    ``OmegaConf.to_container`` when wandb is on. So the real thing is used
    whenever it is available, with ``allow_objects`` set because ``work_dir``
    is a ``Path`` and ``obs_shape`` holds tuples.

    :class:`sim_vla.models.model_config.Node` is the fallback, for offline
    stages and for tests on a machine without omegaconf. It supports attribute
    access, item access, assignment and ``get`` -- everything upstream's model
    and agent touch.
    """
    try:
        from omegaconf import OmegaConf
    except Exception:                                      # noqa: BLE001
        return Node(dict(cfg))
    node = OmegaConf.create({})
    node._set_flag("allow_objects", True)
    OmegaConf.set_struct(node, False)
    for key, value in dict(cfg).items():
        node[key] = value
    return node


def architecture(cfg: Node) -> Dict[str, Any]:
    """The settings a checkpoint has to be compared on.

    ``true_latent_dim`` is deliberately absent: ``WorldModel.__init__`` writes
    it, so reading it here would report a value that only exists after the
    model was built. It is recovered from the parameter report instead.
    """
    out: Dict[str, Any] = {}
    for key in ARCHITECTURE_KEYS:
        value = cfg.get(key, None)
        if value is None:
            continue
        # OmegaConf hands back ListConfig rather than list, and a checkpoint
        # compared against a differently typed but equal value would report a
        # difference that is not one.
        if isinstance(value, (bool, int, float, str)):
            out[key] = value
        else:
            try:
                out[key] = [x for x in value]
            except TypeError:
                out[key] = str(value)
    return out


# --------------------------------------------------------------- integration
DEFAULT_SMOLVLA: Dict[str, Any] = {
    "enabled": True,
    "pretrained": "lerobot/smolvla_base",
    "revision": "",
    # 0 means "take the checkpoint's own value". These are not defaults to be
    # assumed: the loaded config decides, and the tests assert against it.
    "chunk_size": 0,
    "flow_steps": 0,
    "state_token_mode": "embedding",
    "adapter": {"hidden": 1024, "layers": 2},
    # Two systems, not three. See integrations/action_space.py.
    "action_normalization": "identity",
    # How many actions of a chunk are executed before the next chunk is
    # sampled. One, and separate from the chunk length, the planning horizon
    # and the imagination horizon.
    "execute": 1,
    # 0 keeps cfg.num_pi_trajs, which is TD-MPC2's own planning setting. A
    # smaller number here is a *cost* knob for the flow sampler and is
    # reported as a deviation from the native planner settings when set.
    "pi_trajs": 0,
    # Euler steps used for proposals inside planning. 0 keeps flow_steps.
    "proposal_flow_steps": 0,
    "instruction": "",
}

DEFAULT_STAGES: Dict[str, Any] = {
    "world_model": {"steps": 50_000, "log_every": 500},
    "imitation": {"steps": 20_000, "batch_size": 16, "lr": 1e-4,
                  "grad_clip": 1.0, "log_every": 100,
                  "sequence_length": 32, "burn_in": 0},
    "online": {"steps": 1_000_000},
}


def load(task: str = "pickcube",
         overrides: Optional[Mapping[str, Any]] = None,
         root: Path = INTEGRATION_CONFIGS) -> Dict[str, Any]:
    """The staged run's settings: task, world model, SmolVLA, stages."""
    from ...config import load_config as load_sim_vla

    base = _read_yaml(root / "tdmpc2.yaml")
    merged = deep_merge(base, dict(overrides or {}))
    # The task block -- env id, instruction, dataset path -- is the sim_vla
    # one, so all three backends point at the same demonstrations.
    sim_cfg = load_sim_vla(str(task), "dreamer")
    merged["task"] = deep_merge(dict(sim_cfg["task"]), dict(merged.get("task") or {}))
    merged["smolvla"] = deep_merge(DEFAULT_SMOLVLA, dict(merged.get("smolvla") or {}))
    merged["stages"] = deep_merge(DEFAULT_STAGES, dict(merged.get("stages") or {}))
    merged.setdefault("world_model", {})
    merged["world_model"] = dict(merged["world_model"])
    merged["world_model"].setdefault("env_id", merged["task"]["env_id"])
    if not merged["smolvla"].get("instruction"):
        merged["smolvla"]["instruction"] = str(
            merged["task"].get("instruction") or "")
    merged["backend"] = "tdmpc2"
    return merged
