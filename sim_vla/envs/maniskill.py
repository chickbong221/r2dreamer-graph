"""The online environment, built to match the demonstrations exactly.

Every setting that differs between collection and rollout is a difference the
policy has to absorb, so the ones that matter are read from the dataset's own
metadata rather than restated here: the robot, the controller, the camera set
and resolution, the proprioception field order and the control frequency. A
horizon of 150 and the recorded reward mode come from the config.

``ignore_terminations=True``, matching ``envs/maniskill.py`` and matching the
demonstration loader. Nothing terminates; episodes end at the horizon and the
value function bootstraps there.

The final observation is captured before the reset that follows it. A vector
env that resets on the same step it reports ``done`` hands back the *next*
episode's first observation in the slot where the last one belongs, and a world
model trained on that learns a transition between two unrelated scenes.

``num_envs > 1`` runs that many envs on ManiSkill's GPU backend, as the main
trainer does -- the CPU backend holds one env per process. They are stepped in
lockstep through :meth:`SimVlaEnv.reset_all` and :meth:`SimVlaEnv.step_all`,
one row per env. Nothing terminates and every episode is ``max_steps`` long, so
the envs start and finish together and there is no partial reset to get wrong.
The demonstrations stay as they were collected, on the CPU backend; the
physics differs in contact detail, not in the task, and Stage 2 fits the world
model to the online rollouts it trains on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..data.schema import flatten_proprio, unbatch


def _numpy_tree(value: Any) -> Any:
    """ManiSkill's nested observation, every tensor moved to host once.

    Per-env indexing of a CUDA tensor copies the *whole* batch to host for each
    env -- ``extract_camera_obs`` and ``unbatch`` both convert before they
    index -- so a 128-env frame would cross the bus 128 times per field.
    """
    if isinstance(value, Mapping):
        return {key: _numpy_tree(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _rows(value: Any, count: int, dtype) -> np.ndarray:
    """One entry per env from a batched step output.

    A scalar broadcasts -- ``info.get("success", False)`` on a task without a
    success flag -- and anything else must already have one entry per env.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.size == 1 and count > 1:
        return np.full(count, arr[0], dtype=dtype)
    if arr.size != count:
        raise ValueError(f"expected {count} entries, one per env, got "
                         f"{arr.size}")
    return arr


class SimVlaEnv:
    """ManiSkill env(s) producing the dataset's observation contract.

    One CPU env by default, which is what the demonstrations were collected
    in. ``num_envs > 1`` is that many envs on the GPU backend.
    """

    def __init__(self, metadata: Mapping[str, Any], *, graph_enabled: bool,
                 max_steps: int = 150, seed: int = 0,
                 record_graphs: bool = False, num_envs: int = 1,
                 reconfiguration_freq: Optional[int] = None):
        self.metadata = dict(metadata)
        self.graph_enabled = bool(graph_enabled)
        # Recording graphs is not learning from them. A baseline run may set
        # this for diagnostics; the objects stay out of the learning batch.
        self.record_graphs = bool(record_graphs) or self.graph_enabled
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError(f"num_envs={num_envs} must be at least 1")
        # ManiSkill's CPU backend holds a single env; more than one needs the
        # GPU backend, which is what the main trainer's env_num runs on.
        self.sim_backend = "cpu" if self.num_envs == 1 else "gpu"
        # None keeps ManiSkill's own default. For PegInsertionSide that is a
        # new peg and hole on every reset at num_envs=1 and a fixed set for
        # the whole run at num_envs>1, so the two are not the same task
        # distribution unless this is set.
        self.reconfiguration_freq = (None if reconfiguration_freq is None
                                     else int(reconfiguration_freq))
        self.env_id = str(metadata["env_id"])
        self.camera_keys = dict(metadata.get("camera_keys") or {})
        self.proprio_spec = [tuple(f) for f in metadata.get("proprio_fields") or []]
        self.proprio_names = list(metadata.get("proprio_names") or [])
        self.image_size = tuple(metadata.get("image_size") or (112, 112))
        controller = dict(metadata.get("controller") or {})
        self.control_mode = str(controller.get("control_mode") or "pd_joint_pos")
        self.action_dim = int(controller.get("action_dim") or 0)
        self._env = None
        # One graph source per env: temporal edges difference over each env's
        # own last K frames, so the envs cannot share a builder.
        self._graphs: Optional[List[Any]] = None
        self._vocab = None
        self._steps = 0

    # ------------------------------------------------------------------ build
    def build(self):
        import gymnasium as gym                            # noqa: F401
        import mani_skill.envs                             # noqa: F401

        from envs.maniskill import _make_with_supported_reward

        graph_meta = dict(self.metadata.get("graph") or {})
        kwargs: Dict[str, Any] = dict(
            id=self.env_id,
            obs_mode="rgb+segmentation" if self.record_graphs else "rgb",
            control_mode=self.control_mode,
            render_mode="rgb_array",
            sensor_configs=dict(width=int(self.image_size[1]),
                                height=int(self.image_size[0])),
            sim_backend=self.sim_backend,
            num_envs=self.num_envs,
            reward_mode=str(self.metadata.get("reward_mode") or "normalized_dense"),
            max_episode_steps=self.max_steps,
        )
        if self.reconfiguration_freq is not None:
            kwargs["reconfiguration_freq"] = self.reconfiguration_freq
        requested_reward = str(kwargs["reward_mode"])
        self._env = _make_with_supported_reward(kwargs, ["sparse"])
        # _make_with_supported_reward silently falls back when a task does not
        # offer the recorded reward mode. A different reward is a different
        # task: the demonstrations' returns, the reward head and the online
        # returns would all be on different scales while reporting one number.
        actual = str(getattr(self._env.unwrapped, "reward_mode",
                             requested_reward))
        if actual != requested_reward:
            self.close()
            raise SystemExit(
                f"{self.env_id} does not support the recorded reward mode "
                f"{requested_reward!r} and fell back to {actual!r}. The "
                "demonstrations were collected under the recorded mode, so "
                "training against this one compares two different tasks. "
                "Re-collect, or state the substitution deliberately.")
        if self.record_graphs:
            from scenegraph.adapters.graph_vocab import build_graph_vocab
            from scenegraph.figures.graph_source import FigureGraphSource

            self._graphs = [
                FigureGraphSource(
                    self._env, env_id=self.env_id,
                    thresholds_path=str(graph_meta.get("thresholds_path") or ""),
                    whitelist_dir=str(graph_meta.get("whitelist_dir") or ""),
                    use_target_flag=bool(
                        graph_meta.get("use_target_flag", False)),
                    object_object_spatial=bool(
                        graph_meta.get("object_object_spatial", True)),
                    visibility_policy=str(
                        graph_meta.get("visibility_policy") or "keep_tabletop"),
                    env_idx=index)
                for index in range(self.num_envs)]
            self._vocab = build_graph_vocab(self._graphs[0].whitelist_dir)
        try:
            self.validate()
        except BaseException:
            # A partially built env holds a renderer and a physics scene. Left
            # open by a failed construction it survives the exception and the
            # next build competes with it for the GPU.
            self.close()
            raise
        return self

    # -------------------------------------------------------------- validate
    def validate(self) -> Dict[str, Any]:
        """Check the live env against what the dataset recorded.

        The env is constructed from recorded metadata, but constructing from it
        is not the same as matching it: ManiSkill resolves a robot, a control
        frequency and a controller from the task id, and any of them can differ
        from the recording while every argument passed here was accepted. A
        mismatch changes what an action means, and nothing downstream would
        say so -- the run trains, the loss descends, and the policy commands a
        different robot than the demonstrations did.

        Checked for both arms. The graph capacities were already checked at
        dataset load; task identity was not checked anywhere.
        """
        if self._env is None:
            raise RuntimeError("validate() before build()")
        unwrapped = self._env.unwrapped
        controller = dict(self.metadata.get("controller") or {})
        problems: list[str] = []
        checked: list[str] = []

        def compare(name: str, recorded: Any, actual: Any) -> None:
            """Compare only what was recorded, and report what was compared.

            The list of checked keys is returned, because a validation that
            silently checked nothing reads exactly like one that passed. The
            first version of this looked for ``robot_uid`` and top-level
            frequencies; the collector writes ``robot_uids`` and puts the
            frequencies inside ``controller``, so every one of those compares
            saw ``None``, returned early, and reported ``validated=True`` for a
            deliberately mismatched robot.
            """
            if recorded in (None, "", 0, [], {}):
                return                                     # not recorded
            checked.append(name)
            if str(recorded) != str(actual):
                problems.append(f"{name}: recorded={recorded!r} env={actual!r}")

        compare("env_id", self.metadata.get("env_id"), self.env_id)
        compare("reward_mode", self.metadata.get("reward_mode"),
                getattr(unwrapped, "reward_mode", None))
        # sim_vla.data.schema.controller_metadata is what wrote these.
        compare("controller.control_mode", controller.get("control_mode"),
                getattr(unwrapped, "control_mode", None))
        compare("controller.control_freq", controller.get("control_freq"),
                getattr(unwrapped, "control_freq", None))
        compare("controller.sim_freq", controller.get("sim_freq"),
                getattr(unwrapped, "sim_freq", None))
        # environment_metadata writes robot_uids at the top level.
        compare("robot_uids", self.metadata.get("robot_uids"),
                getattr(unwrapped, "robot_uids", None))

        # The action width decides what a command even is.
        space = (getattr(unwrapped, "single_action_space", None)
                 or getattr(self._env, "action_space", None))
        shape = getattr(space, "shape", None)
        if controller.get("action_dim") and shape:
            checked.append("controller.action_dim")
            live = int(np.prod(np.asarray(shape)))
            if live != int(controller["action_dim"]):
                problems.append(
                    f"controller.action_dim: recorded={controller['action_dim']} "
                    f"env={live}")
        for edge in ("action_low", "action_high"):
            recorded = controller.get(edge)
            live = getattr(space, edge.split("_")[1], None)
            if recorded is None or live is None:
                continue
            recorded = np.asarray(recorded, dtype=float).reshape(-1)
            live = np.asarray(live, dtype=float).reshape(-1)
            if recorded.shape != live.shape:
                # The width mismatch above already says this; comparing the
                # values would just raise a broadcasting error on top of it.
                continue
            checked.append(f"controller.{edge}")
            if not np.allclose(recorded, live, atol=1e-6):
                problems.append(
                    f"controller.{edge}: the controller's bounds differ from "
                    "the recording, so the same float means a different "
                    "command")

        if self.graph_enabled:
            problems += self._validate_graph(checked)

        if problems:
            raise SystemExit(
                "the online environment does not match the recording:\n  "
                + "\n  ".join(problems)
                + "\nTraining against it would compare a policy to "
                  "demonstrations from a different task.")
        if not checked:
            raise SystemExit(
                "nothing in the recording could be validated against the live "
                f"environment: the metadata has {sorted(self.metadata)[:12]} "
                "and none of the keys this checks were present. A validation "
                "that compares nothing is not a validation.")
        return {"env_id": self.env_id, "control_mode": self.control_mode,
                "action_dim": self.action_dim, "checked": checked,
                "validated": True}

    def _validate_graph(self, checked: List[str]) -> List[str]:
        """Compare the graph vocabulary itself, not merely that files exist.

        A whitelist directory that exists is not the whitelist the recording
        used. What decides what a token *means* is the token-to-id mapping, so
        that is what is compared, and the recorded digests are compared against
        the live ones so a silently edited whitelist is caught too.
        """
        from ..data.schema import dir_digest, resolved_thresholds_path

        graph_meta = dict(self.metadata.get("graph") or {})
        problems: List[str] = []

        for key, path in (("whitelist_dir", self._graphs[0].whitelist_dir),
                          ("thresholds_path",
                           resolved_thresholds_path(
                               str(graph_meta.get("thresholds_path") or "")))):
            digest_key = key.replace("_dir", "").replace("_path", "") + "_digest"
            recorded = graph_meta.get(digest_key)
            if not recorded:
                continue
            checked.append(f"graph.{digest_key}")
            try:
                live = dir_digest(path)
            except Exception as exc:                       # noqa: BLE001
                problems.append(f"graph.{digest_key}: cannot read {path}: {exc}")
                continue
            if str(live) != str(recorded):
                problems.append(
                    f"graph.{digest_key}: recorded={recorded} live={live}; the "
                    f"contents of {path} are not what was recorded")

        if self._vocab is None:
            return problems
        for name, attribute in (("entity_tokens", "entity"),
                                ("relation_tokens", "relation"),
                                ("absolute_tokens", "absolute"),
                                ("temporal_tokens", "temporal")):
            recorded = graph_meta.get(name)
            table = getattr(self._vocab, attribute, None)
            live = getattr(table, "token_to_id", None)
            if not recorded or live is None:
                continue
            checked.append(f"graph.{name}")
            if {str(k): int(v) for k, v in dict(recorded).items()} != \
                    {str(k): int(v) for k, v in dict(live).items()}:
                problems.append(
                    f"graph.{name}: the token ids differ from the recording, "
                    "so the same integer in a packed graph means a different "
                    "thing")
        recorded_sizes = graph_meta.get("vocab_sizes")
        live_sizes = getattr(self._vocab, "sizes", None)
        if recorded_sizes and live_sizes is not None:
            checked.append("graph.vocab_sizes")
            if {str(k): int(v) for k, v in dict(recorded_sizes).items()} != \
                    {str(k): int(v) for k, v in dict(live_sizes).items()}:
                problems.append(
                    f"graph.vocab_sizes: recorded={dict(recorded_sizes)} "
                    f"live={dict(live_sizes)}")
        return problems

    # ------------------------------------------------------------ observation
    def _observation(self, raw: Mapping[str, Any],
                     env_idx: int = 0) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        sensors = raw.get("sensor_data") or {}
        for camera, key in self.camera_keys.items():
            out[key] = np.asarray(unbatch(sensors[camera]["rgb"], env_idx),
                                  dtype=np.uint8)
        vector, _ = flatten_proprio(raw, self.proprio_spec, env_idx)
        out["proprio"] = vector
        if self.graph_enabled:
            from scenegraph.adapters.graph_pack import pack_graph

            graph_meta = dict(self.metadata.get("graph") or {})
            packed = pack_graph(
                self._graphs[env_idx].step(raw), self._vocab,
                n_max=int(graph_meta.get("n_max", 8)),
                e_max=int(graph_meta.get("e_max", 168)),
                n_cams=int(graph_meta.get("n_cams", len(self.camera_keys))),
                use_target_flag=bool(graph_meta.get("use_target_flag", False)))
            out |= packed
        elif self.record_graphs:
            # Built for diagnostics and deliberately not returned: an object in
            # the observation dict is an object that reaches a batch.
            self._graphs[env_idx].step(raw)
        return out

    def _observations(self, raw: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """Every env's observation, stacked on a leading env axis.

        Built row by row through :meth:`_observation`, the same function the
        single env uses, so the two cannot disagree about the contract.
        """
        raw = _numpy_tree(raw)
        graphs = self._graphs is not None
        if graphs:
            from scenegraph.adapters.privileged_state import (
                begin_frame_cache, end_frame_cache)

            # One pose snapshot for all the envs' builders, as graph_obs takes
            # it for the main trainer, rather than one per builder.
            begin_frame_cache(getattr(self._env.unwrapped, "scene", None))
        try:
            rows = [self._observation(raw, index)
                    for index in range(self.num_envs)]
        finally:
            if graphs:
                end_frame_cache()
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}

    def _reset_graphs(self) -> None:
        """Per environment, and independent of the recurrent state reset.

        Temporal edges difference over the last K frames, and a builder
        carried across a reset describes the previous episode.

        After env.reset, never before -- which is what
        FigureGraphSource.on_reset documents. It sets merged-view aliasing on
        the *scene*, and a reset that reconfigures (PegInsertionSide rebuilds
        its hole every episode at num_envs=1) constructs a new one: called
        first, the flag lands on the scene about to be discarded, the new scene
        never receives it, and the segmentation ids the builder reads stop
        aliasing to the actors the whitelist names -- so a scheduled subject
        resolves to no node and goal_edges refuses the frame. It also drops
        caches keyed on the old actors, which is only correct once those
        actors are gone. sim_vla/data/collect.py wraps the env so the recording
        gets this order; the two have to agree or the graphs do not.
        """
        for source in self._graphs or ():
            source.on_reset()

    @property
    def live_reconfiguration_freq(self) -> Optional[int]:
        """What the built env actually uses, ManiSkill's default included."""
        live = getattr(getattr(self._env, "unwrapped", None),
                       "reconfiguration_freq", None)
        return self.reconfiguration_freq if live is None else int(live)

    def _single(self, method: str) -> None:
        if self.num_envs != 1:
            raise RuntimeError(
                f"{method}() drives one env and this one holds "
                f"{self.num_envs}; use {method}_all(), which takes and returns "
                "one row per env. Quietly driving env 0 alone would step the "
                "others with whatever action they last received.")

    # ----------------------------------------------------------------- driving
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        self._single("reset")
        if self._env is None:
            self.build()
        raw, _info = self._env.reset(seed=self.seed if seed is None else int(seed))
        self._reset_graphs()
        self._steps = 0
        obs = self._observation(raw)
        obs["is_first"] = np.array(True)
        return obs

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        self._single("step")
        raw, reward, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32))
        self._steps += 1
        obs = self._observation(raw)
        obs["is_first"] = np.array(False)
        done = bool(self._steps >= self.max_steps) or bool(
            np.asarray(unbatch(truncated)).reshape(-1)[0])
        return {
            "obs": obs,
            "reward": float(np.asarray(unbatch(reward)).reshape(-1)[0]),
            # ignore_terminations: the recorded flag is kept for diagnostics
            # and is not what ends an episode here.
            "terminated_recorded": bool(
                np.asarray(unbatch(terminated)).reshape(-1)[0]),
            "is_terminal": False,
            "is_last": done,
            "success": bool(np.asarray(
                unbatch(info.get("success", False))).reshape(-1)[0])
            if isinstance(info, dict) else False,
        }

    def reset_all(self, seeds: Optional[Sequence[int]] = None
                  ) -> Dict[str, np.ndarray]:
        """Reset every env; one seed per env, or ``self.seed`` spread by
        ManiSkill when none are given."""
        if self._env is None:
            self.build()
        if seeds is None:
            seed: Any = self.seed
        else:
            seed = [int(s) for s in seeds]
            if len(seed) != self.num_envs:
                raise ValueError(f"{len(seed)} seeds for {self.num_envs} envs")
        raw, _info = self._env.reset(seed=seed)
        self._reset_graphs()
        self._steps = 0
        obs = self._observations(raw)
        obs["is_first"] = np.ones(self.num_envs, dtype=bool)
        return obs

    def step_all(self, actions: np.ndarray) -> Dict[str, Any]:
        """Step every env with its own row of ``actions``.

        The same fields as :meth:`step`, each with one entry per env. The step
        count is shared -- the envs were reset together and nothing
        terminates -- so ``is_last`` is the horizon or ManiSkill's own
        truncation, per env.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(f"actions of shape {actions.shape} for "
                             f"{self.num_envs} envs; one row per env")
        raw, reward, terminated, truncated, info = self._env.step(actions)
        self._steps += 1
        count = self.num_envs
        obs = self._observations(raw)
        obs["is_first"] = np.zeros(count, dtype=bool)
        done = _rows(truncated, count, bool) | bool(self._steps >= self.max_steps)
        success = (_rows(info.get("success", False), count, bool)
                   if isinstance(info, dict) else np.zeros(count, dtype=bool))
        return {
            "obs": obs,
            "reward": _rows(reward, count, np.float32),
            # ignore_terminations, as in step().
            "terminated_recorded": _rows(terminated, count, bool),
            "is_terminal": np.zeros(count, dtype=bool),
            "is_last": done,
            "success": success,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

def load_metadata(dataset: str | Path) -> Dict[str, Any]:
    """The contract the online env has to match, from the dataset itself."""
    sidecar = Path(dataset).with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return dict(payload.get("metadata") or {})
