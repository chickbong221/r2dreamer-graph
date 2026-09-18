"""What one demonstration episode contains, and what the arrays mean.

The dataset holds two array lengths and they are off by one. An episode with
``T`` executed actions carries ``T + 1`` observations::

    o_0, a_0, r_0, o_1, a_1, r_1, ... a_{T-1}, r_{T-1}, o_T

``r_t`` is the reward for executing ``a_t`` and arriving at ``o_{t+1}``. Which
length a field has is declared here rather than inferred at load: two arrays of
the same shape mean different things depending on whether they are indexed by
observation or by transition, and a loader that guesses wrong shifts every
reward by one step without failing.

Three things this module is deliberate about.

**Proprioception is an allowlist, not an exclusion list.** ManiSkill's ``extra``
dict is task-specific and grows: PickCube publishes ``is_grasped`` and
``goal_pos`` there, which the robot cannot observe. Naming what goes *in* means
a task that adds a field does not silently add it to the policy's input, which
is the failure mode a deny-list has. Everything not named here is still
recorded, under ``privileged``, where it is available for diagnostics and
cannot be mistaken for a model input.

**The Dreamer flags are not stored.** ``is_first`` / ``is_last`` /
``is_terminal`` are derived at load from ``terminated``, ``truncated`` and the
episode's ``end_reason``. Cutting a recording after success is a decision this
collector made; it is not a task termination, and writing one would teach a
world model that the task ends where the solver stopped being interesting.

**Metadata carries identities, not just shapes.** The graph vocabulary is
written token-by-token rather than as four sizes, because two runs can agree on
every array shape and still disagree on which integer means ``grasp``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Arrays indexed by observation, length T+1.
KIND_OBS = "obs"
# Arrays indexed by transition, length T.
KIND_STEP = "step"

# Why a recording stopped. Kept distinct from ``terminated`` on purpose: only
# the task decides that an episode ended, and none of these reasons are it.
END_SOLVER_FINISHED = "solver_finished"   # the scripted plan ran to its end
END_SUCCESS_CUT = "success_cut"           # trimmed after success settled
END_BUDGET = "budget_exceeded"            # rejected, never written
END_SOLVER_ERROR = "solver_error"         # the planner raised
END_INTERRUPTED = "interrupted"           # the collector was stopped

# What the robot can observe about itself. ``tcp_pose`` is forward kinematics
# from the joint angles, so it is proprioceptive in the sense that matters: a
# real arm can compute it without a camera or a simulator.
DEFAULT_PROPRIO: Tuple[Tuple[str, str], ...] = (
    ("agent", "qpos"),
    ("agent", "qvel"),
    ("extra", "tcp_pose"),
)

# Per task, where it differs. Empty means DEFAULT_PROPRIO applies; a task is
# listed here only when its own observation names something else, and the list
# is written into the dataset's metadata either way so a reader never has to
# come back to this file to find out what the columns were.
PROPRIO_BY_TASK: Dict[str, Tuple[Tuple[str, str], ...]] = {}


def proprio_fields(env_id: str) -> Tuple[Tuple[str, str], ...]:
    return PROPRIO_BY_TASK.get(str(env_id), DEFAULT_PROPRIO)


def unbatch(value, env_idx: int = 0) -> np.ndarray:
    """One env's row out of whatever ManiSkill returned.

    Everything comes back batched even at batch size one, and as a torch
    tensor even on the CPU backend. Indexed explicitly rather than squeezed: a
    squeeze also removes a genuine length-one feature axis, and silently
    changes the width of a column a loader is about to name.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr
    return arr[env_idx]


def flatten_proprio(
    obs: Mapping[str, Any],
    fields: Sequence[Tuple[str, str]],
    env_idx: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """The proprioception vector and one name per column.

    Concatenated in the order ``fields`` gives, which is the order the names
    record. A field the task does not publish is an error rather than a silent
    gap: a dataset missing a column it claims to have is worse than one that
    refused to be written.
    """
    parts: List[np.ndarray] = []
    names: List[str] = []
    for group, key in fields:
        section = obs.get(group)
        if section is None or key not in section:
            available = sorted(section or ()) if section is not None else []
            raise KeyError(
                f"proprio field {group}.{key} is not in this observation; "
                f"{group} has {available}. Name the task's own fields in "
                f"sim_vla.data.schema.PROPRIO_BY_TASK."
            )
        arr = np.asarray(unbatch(section[key], env_idx), dtype=np.float32).reshape(-1)
        parts.append(arr)
        names.extend(
            f"{group}.{key}" if arr.size == 1 else f"{group}.{key}[{i}]"
            for i in range(arr.size)
        )
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), names


def privileged_fields(
    obs: Mapping[str, Any], fields: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Everything in ``extra`` the proprio allowlist did not take.

    Recorded, and recorded separately. These are the fields that make an
    experiment privileged if they reach the policy, and keeping them in the
    file is what lets a diagnostic ask why an episode went wrong without
    re-running it.
    """
    taken = set(fields)
    return [("extra", key) for key in sorted(obs.get("extra") or ())
            if ("extra", key) not in taken]


def dir_digest(path: str | Path, pattern: str = "**/*") -> str:
    """Content hash of a config directory, or ``missing``.

    The whitelist and threshold files decide what the graph's tokens mean. Two
    datasets built a month apart can carry the same vocabulary sizes and the
    same relation names while one of them was mined against a different
    calibration, and this is the field that says so.
    """
    root = Path(path)
    if not root.exists():
        return "missing"
    digest = hashlib.sha1()
    if root.is_file():
        digest.update(root.read_bytes())
        return digest.hexdigest()
    for item in sorted(p for p in root.glob(pattern) if p.is_file()):
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def resolved_thresholds_path(path: str = "") -> str:
    """The thresholds file actually read, including the packaged default.

    An empty setting is not "no thresholds" -- it is
    ``scenegraph/configs/thresholds.yaml``, which carries the contact, grasp
    and support predicates every edge label depends on. Recording the literal
    string ``default`` for it would name the one file whose contents most need
    hashing and then not hash it.
    """
    if path:
        return str(path)
    # Located from this file rather than by importing the loader: the path is a
    # fixed fact about the repo layout, and making metadata depend on an import
    # means the one field that identifies the thresholds is the one that breaks
    # first when the module is stubbed or the package is reorganised.
    packaged = Path(__file__).resolve().parents[2] / "scenegraph/configs/thresholds.yaml"
    if packaged.is_file():
        return str(packaged)
    try:
        from scenegraph.configs import loader

        return str(Path(loader.__file__).with_name("thresholds.yaml"))
    except Exception:                                      # noqa: BLE001
        return str(packaged)


# Keys the graph builder writes back into the config once it has seen a scene.
# They are not settings: they are the union whitelist's contents, resolved
# against whatever the episode contained. ``whitelist_digest`` already
# identifies the asset they come from, so dropping them loses nothing and keeps
# the recorded configuration comparable between processes -- which matters
# because ``structural_surfaces`` is a Python set, and a set's iteration order
# depends on per-process string hash randomisation. Serialised, it comes out in
# a different order in every worker, and eight shards of one run then disagree
# about a configuration they share.
RUNTIME_CFG_KEYS: Tuple[str, ...] = (
    "structural_surfaces", "families", "site_declarations", "bin_edges",
    "site_specs",
)


def json_safe(value: Any) -> Any:
    """A config tree as JSON, keeping its shape and its order.

    The earlier version of this kept only scalars, which quietly dropped every
    nested block -- contact, grasp, support, selection -- so two datasets could
    agree on every recorded graph setting while disagreeing on the thresholds
    that decide what an edge label means.
    """
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()
                # Caches and handles the builder hangs off the config: runtime
                # state, not settings, and not serialisable either.
                if not str(k).startswith("_")}
    # Sorted, not stringified. An unordered container has to be given an order
    # here or it acquires a different one in every process.
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# Metadata that has to agree before two shards are one dataset. Vocabulary
# alone is not enough: two shards can map ``grasp`` to the same integer and
# still have been rendered at different resolutions, packed at different
# capacities, or recorded from different tasks.
MERGE_KEYS: Tuple[str, ...] = (
    "env_id", "graph", "camera_keys", "cameras", "image_size",
    "proprio_names", "proprio_fields", "privileged_fields",
    "controller", "reward_mode", "field_kinds", "budget",
)


def graph_config_snapshot(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The graph's settings, without the state the builder writes into them."""
    return {key: value for key, value in json_safe(cfg).items()
            if key not in RUNTIME_CFG_KEYS}


def sanitize_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """A metadata block with the builder's runtime state removed.

    Applied to what is *read back* as well as to what is written, so a dataset
    recorded before this distinction existed still compares correctly against
    one recorded after.
    """
    out = dict(meta)
    graph = out.get("graph")
    if isinstance(graph, Mapping) and isinstance(graph.get("config"), Mapping):
        out["graph"] = dict(graph) | {
            "config": graph_config_snapshot(graph["config"])}
    return out


def merge_conflicts(reference: Mapping[str, Any],
                    other: Mapping[str, Any]) -> List[str]:
    """Which of :data:`MERGE_KEYS` two metadata blocks disagree on."""
    left, right = sanitize_metadata(reference), sanitize_metadata(other)
    return [key for key in MERGE_KEYS if left.get(key) != right.get(key)]


def controller_metadata(env) -> Dict[str, Any]:
    """Action space, controller identity, and how the action is scaled.

    An action vector is meaningless without this: the same eight floats are
    absolute joint targets under one controller and deltas under another, and
    normalised to [-1, 1] or not depending on a flag that lives nowhere in the
    array.
    """
    base = getattr(env, "unwrapped", env)
    space = getattr(base, "single_action_space", None) or env.action_space
    out: Dict[str, Any] = {
        "control_mode": str(getattr(base, "control_mode", "")),
        "action_dim": int(np.prod(space.shape)),
        "action_low": np.asarray(space.low, dtype=float).reshape(-1).tolist(),
        "action_high": np.asarray(space.high, dtype=float).reshape(-1).tolist(),
        "control_freq": int(getattr(base, "control_freq", 0) or 0),
        "sim_freq": int(getattr(base, "sim_freq", 0) or 0),
        "action_repeat": 1,
    }
    # Best effort, and labelled as such: the controller config is a ManiSkill
    # dataclass whose fields move between versions, so a failure to read it
    # must not cost the whole dataset its metadata.
    try:
        from dataclasses import asdict, is_dataclass

        configs = base.agent.controller.configs
        out["controller_configs"] = {
            name: (asdict(cfg) if is_dataclass(cfg) else str(cfg))
            for name, cfg in configs.items()
        }
    except Exception as exc:                               # noqa: BLE001
        out["controller_configs"] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


def build_metadata(
    *,
    env,
    env_id: str,
    env_kwargs: Mapping[str, Any],
    reward_mode: str,
    reward_fallback: Sequence[str],
    cameras: Sequence[str],
    camera_keys: Mapping[str, str],
    image_size: Sequence[int],
    proprio_names: Sequence[str],
    proprio_spec: Sequence[Tuple[str, str]],
    privileged_spec: Sequence[Tuple[str, str]],
    graph_cfg: Mapping[str, Any],
    vocab,
    whitelist_dir: str,
    thresholds_path: str,
    temporal_k: int,
    n_max: int,
    e_max: int,
    n_cams: int,
    visibility_policy: str,
    use_target_flag: bool,
    object_object_spatial: bool,
    max_steps: int,
    pad: int,
    horizon: Optional[int],
    field_kinds: Mapping[str, str],
    config_source: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything a reader needs to interpret the arrays, in one dict."""
    base = getattr(env, "unwrapped", env)
    try:
        import mani_skill

        sim_version = str(mani_skill.__version__)
    except Exception:                                      # noqa: BLE001
        sim_version = "unknown"
    from graph_encoder_probe.dataset import git_revision

    return {
        "schema_version": 1,
        "env_id": str(env_id),
        "env_kwargs": dict(env_kwargs),
        "robot_uids": str(getattr(base, "robot_uids", "")),
        "reward_mode": str(reward_mode),
        "reward_mode_requested": list(reward_fallback or []),
        "cameras": [str(c) for c in cameras],
        "camera_keys": {str(k): str(v) for k, v in camera_keys.items()},
        "image_size": [int(v) for v in image_size],
        "proprio_names": list(proprio_names),
        "proprio_fields": [list(f) for f in proprio_spec],
        "privileged_fields": [list(f) for f in privileged_spec],
        "controller": controller_metadata(env),
        "graph": {
            # The whole resolved tree, nested blocks included: the contact,
            # grasp and support settings are what an edge label means. What the
            # builder later writes back into this dict from the scene is not a
            # setting and is left out -- see RUNTIME_CFG_KEYS.
            "config": graph_config_snapshot(graph_cfg),
            "temporal_k": int(temporal_k),
            "n_max": int(n_max),
            "e_max": int(e_max),
            "n_cams": int(n_cams),
            "visibility_policy": str(visibility_policy),
            "use_target_flag": bool(use_target_flag),
            "object_object_spatial": bool(object_object_spatial),
            "whitelist_dir": str(whitelist_dir),
            "whitelist_digest": dir_digest(whitelist_dir),
            "thresholds_path": resolved_thresholds_path(thresholds_path),
            "thresholds_digest": dir_digest(resolved_thresholds_path(thresholds_path)),
            "vocab_sizes": dict(vocab.sizes),
            # Token-by-token, not just sizes: two runs can agree on every array
            # shape and still disagree on which integer means which relation.
            "entity_tokens": dict(vocab.entity.token_to_id),
            "relation_tokens": dict(vocab.relation.token_to_id),
            "absolute_tokens": dict(vocab.absolute.token_to_id),
            "temporal_tokens": dict(vocab.temporal.token_to_id),
        },
        "budget": {
            "max_steps_to_success": int(max_steps),
            "pad_after_success": int(pad),
            "registered_horizon": None if horizon is None else int(horizon),
        },
        "field_kinds": dict(field_kinds),
        "notes": {
            "lengths": "fields of kind 'obs' have T+1 rows, 'step' have T",
            "flags": "is_first/is_last/is_terminal are derived at load from "
                     "terminated, truncated and end_reason; a success cut is "
                     "not a termination",
            "rerender": "env_states restore poses, so RGB can be re-rendered "
                        "at another resolution. Graph and reward cannot be "
                        "re-derived that way: contact impulses are a "
                        "simulator buffer that set_state_dict does not refresh",
            "graph_fields": "the nine graph_* arrays are an extension; "
                            "ManiSkill's trajectory tools do not preserve them",
        },
        "versions": {
            "mani_skill": sim_version,
            "repo_revision": git_revision(),
        },
        # Which settings came from the training config rather than a flag, and
        # which files they were read from. Empty means the config was not
        # consulted and every value below is a default or an explicit flag.
        "config_source": dict(config_source or {}),
    }
