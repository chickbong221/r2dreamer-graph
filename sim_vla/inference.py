"""Step an SO-101 checkpoint (dreamer or graph_progress arm) one observation at a time.

    hf download TuanDepZai/r2dreamer-graph-so101-checkpoints \
        --include "stackcube_graph_progress_seed0/*" --local-dir checkpoint
    python -m sim_vla.inference --checkpoint checkpoint/stackcube_graph_progress_seed0 --steps 3

    from sim_vla.inference import SO101Policy, dummy_observation
    model = SO101Policy.load("checkpoint/stackcube_graph_progress_seed0")
    obs = dummy_observation(model)
    out = model(obs)        # out["action"]["single_arm"] (1, 5), out["action"]["gripper"] (1, 1)

A graph arm needs a scene graph for every observation. Unless obs["scene_graph"]
holds one, model(obs) saves both frames under the exchange folder and waits for
an LLM agent to write scene_graph.json beside them; instructions.md there says how.
"""

from __future__ import annotations

import argparse
import json
import numbers
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from real_robot.common import canonical_json, load_configs, read_json, write_json
from real_robot.graphs.pack import build_frame_graph
from real_robot.graphs.schema import GraphConfig
from real_robot.graphs.validate import fact_ids, gemini_box_to_normalized, normalize_entity
from real_robot.graphs.vocabulary import build_vocab, vocab_tables
from scenegraph.adapters.graph_pack import GRAPH_KEYS, pack_graph

from .data.convert_real import IMAGE_SIZE
from .data.normalization import Normalizer
from .models.action_space import ActionBounds, ActionCoordinates
from .models.model_config import REPO, load_model_config
from .models.world_model import build_world_model
from .runtime.checkpoint import CheckpointMeta, load
from .training.online import LatentPolicy
from .training.progress import HEAD_IDENTITY, build_progress, predict

JOINTS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
CAMERAS = ("top", "wrist")
LANGUAGE_KEY = "annotation.human.task_description"
TABLE = "table"
PRIMARY = ("grasp", "contain", "support", "contact")


def _single(value: Any, name: str, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(want not in (-1, got) for want, got in zip(shape, array.shape)):
        wanted = ", ".join("*" if want == -1 else str(want) for want in shape)
        raise ValueError(f"{name} must have shape ({wanted}), got {array.shape}")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be {np.dtype(dtype)}, got {array.dtype}")
    return array[0]


def _box(value: Any) -> Optional[Tuple[List[int], List[float]]]:
    """``(rounded [ymin, xmin, ymax, xmax], normalized [x0, x1, y0, y1])``, or None if malformed."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in value)):
        return None
    try:
        rounded = [int(round(float(v))) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    box = gemini_box_to_normalized(rounded)
    return (rounded, box) if box is not None and box[1] > box[0] and box[3] > box[2] else None


def training_labels(spec, vocab, folder: Path) -> np.ndarray:
    """The absolute label id of every fact on every training frame, ``(frames, facts)``."""
    table = np.full((len(vocab.entity), len(vocab.entity), len(vocab.relation)), -1, dtype=np.int64)
    for index, fact in enumerate(spec.facts):
        table[vocab.entity.encode(spec.entity(fact.src).key), vocab.entity.encode(spec.entity(fact.dst).key),
              vocab.relation.encode(fact.relation)] = index
    blocks = []
    for path in sorted(folder.glob("episode_*.npz")):
        with np.load(path) as data:
            ent = data["graph_node_ent"].astype(np.int64)
            src = np.take_along_axis(ent, data["graph_edge_src"].astype(np.int64), 1)
            dst = np.take_along_axis(ent, data["graph_edge_dst"].astype(np.int64), 1)
            rel, absolute = data["graph_edge_rel"].astype(np.int64), data["graph_edge_abs"]
        frames, edges = np.nonzero(rel)
        facts = table[src, dst, rel][frames, edges]
        if (facts < 0).any():
            raise SystemExit(f"{path} holds a fact that {spec.task} does not define")
        block = np.zeros((len(ent), len(spec.facts)), dtype=np.int64)
        block[frames, facts] = absolute[frames, edges]
        blocks.append(block)
    if not blocks:
        raise SystemExit(f"no training graphs in {folder}")
    return np.concatenate(blocks)


class SceneGraphChannel:
    """One scene graph per observation, written by an LLM agent and packed as the training graphs were."""

    def __init__(self, cfg: Mapping[str, Any], root: Path, *, poll: float = 0.5):
        configs = load_configs(["dataset", "graph", "labels"])
        graph_config = GraphConfig(configs["graph"])
        self.spec = graph_config.spec(str(cfg["task"]["env_id"]).split("/", 1)[1])
        self.vocab = build_vocab(graph_config)
        self.configs = configs
        folder = REPO / str(cfg["task"]["source"]["graphs"])
        manifest = read_json(folder / "manifest.json")
        if (canonical_json(manifest["graph"]) != canonical_json(graph_config.identity())
                or canonical_json(manifest["vocab"]) != canonical_json(vocab_tables(self.vocab))):
            raise SystemExit(f"real_robot/configs/graph.yaml no longer describes {folder}, "
                             "the graphs this checkpoint was trained on")
        model = cfg["model"]["graph"]
        packed = (self.spec.n_max, self.spec.e_max, len(self.spec.cameras))
        if packed != (int(model["n_max"]), int(model["e_max"]), int(model["n_cams"])) or self.spec.cameras != CAMERAS:
            raise SystemExit(f"the {self.spec.task} graphs are packed as {packed} over {self.spec.cameras}; "
                             f"the checkpoint expects {model} over {CAMERAS}")
        self.ids = fact_ids(self.spec)
        self.seen = training_labels(self.spec, self.vocab, folder)
        self.seen_ids = [set(np.unique(column).tolist()) for column in self.seen.T]
        self.label_name = {i: t for t, i in self.vocab.absolute.token_to_id.items()}
        self.root = Path(root)
        self.poll = float(poll)
        self.root.mkdir(parents=True, exist_ok=True)
        self.guide = self.root / "instructions.md"
        self.guide.write_text(self.instructions(str(cfg["task"]["instruction"])), encoding="utf-8")
        self.episode = -1
        self.reset()

    def reset(self) -> None:
        self.episode += 1
        self.step = 0
        self.history: List[List[str]] = []
        self.previous: Optional[Dict[str, Any]] = None

    # --------------------------------------------------------------- guidance
    def hint(self, index: int) -> str:
        """How the demonstrations' annotator labelled one fact."""
        column = self.seen[:, index]
        values, counts = np.unique(column, return_counts=True)
        if len(values) == 1:
            return f"always `{self.label_name[int(values[0])]}`"
        rank = [PRIMARY.index(f.relation) if f.relation in PRIMARY else len(PRIMARY) for f in self.spec.facts]
        for other in sorted(range(len(self.spec.facts)), key=lambda f: (rank[f], f)):
            if rank[other] >= min(rank[index], len(PRIMARY)):
                break
            key = self.seen[:, other]
            keys = np.unique(key)
            if len(keys) < 2 or len(np.unique(key * 64 + column)) != len(keys):
                continue
            rule = "; ".join(f"`{self.label_name[int(column[key == k][0])]}` when {self.ids[other]} is "
                             f"`{self.label_name[int(k)]}`" for k in keys)
            return f"{rule} ({self.ids[other]} is {self.spec.facts[other].label()})"
        order = np.argsort(-counts)
        return ", ".join(f"`{self.label_name[int(values[i])]}` {100 * counts[i] / len(column):.0f}%" for i in order)

    def instructions(self, task: str) -> str:
        from real_robot.preprocessing.annotate_episode import labels_text

        spec, labels = self.spec, self.configs["labels"]
        described = self.configs["dataset"]["source"]["camera_descriptions"]
        sections = labels_text(spec, labels).split("\n### ")
        definitions = "\n### ".join(part for part in sections if not part.startswith("Temporal"))
        lines = [
            f"# Scene graphs for `{spec.task}`, one per observation",
            "",
            f"Task: {task} One SO-101 arm with a two-jaw gripper, two synchronised cameras:",
            *(f"- `{camera}`: {described[camera]}" for camera in CAMERAS),
            "",
            "For every step the policy saves `episode_EEE/step_SSSSSS/top.png` and `wrist.png` and waits for "
            "`scene_graph.json` in the same folder. Start from `template.json` there (your previous answer, or "
            "blanks at the first step of an episode), change what the new frames show differently, and write "
            "the whole file at once. Label only what these two frames show. A rejected file is explained on "
            "the console and in `errors.txt`; fix it and save again.",
            "",
            "## Entities",
            *(f"- `{e.id}`: {e.description}. Reference point: {e.reference}." for e in spec.entities),
            "",
            "## active_target",
            f"One of {', '.join(f'`{t}`' for t in spec.targets)}: {spec.target_rule}",
            "",
            "## facts",
            "One label per fact from the labels listed for it. `Training` is how the annotator of the "
            "demonstrations labelled that fact; the policy has never seen other labels for it, so follow it. "
            f"Temporal labels are derived from your answers {spec.temporal_window} steps apart, as the "
            "annotator did; do not write them.",
            *(f"- `{fid}` {fact.label()}: {' | '.join(spec.legal_labels(fact.relation))}. "
              f"Training: {self.hint(index)}." for index, (fid, fact) in enumerate(zip(self.ids, spec.facts))),
            "",
            "## boxes",
            "Per camera and entity, `[ymin, xmin, ymax, xmax]` as integers 0-1000 relative to the image, tight "
            "around the visible part, or `null` when the entity is not visible in that camera. The table is "
            "not asked for: it is the whole image in both cameras, as in training.",
            "",
            "## Label definitions",
            "",
            definitions,
            "",
            "Approximate sizes: " + "; ".join(labels["reference_sizes"]) + ".",
            "",
            "## Format",
            "",
            "```json",
            json.dumps(self.blank(), indent=2),
            "```",
            "",
        ]
        return "\n".join(lines)

    def blank(self) -> Dict[str, Any]:
        constant = {}
        for index in range(len(self.spec.facts)):
            values = np.unique(self.seen[:, index])
            constant[index] = self.label_name[int(values[0])] if len(values) == 1 else None
        return self.answer(self.spec.targets[0] if len(self.spec.targets) == 1 else None,
                           [constant[i] for i in range(len(self.spec.facts))],
                           {camera: {e.id: None for e in self.spec.entities if e.id != TABLE}
                            for camera in CAMERAS})

    def answer(self, target, labels, boxes) -> Dict[str, Any]:
        return {"active_target": target,
                "facts": {f"{fid} {fact.label()}": label
                          for fid, fact, label in zip(self.ids, self.spec.facts, labels)},
                "boxes": boxes}

    # ----------------------------------------------------------------- answer
    def read(self, raw: Any) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
        """``(parsed, problems, warnings)``; parsed is None when there are problems."""
        if not isinstance(raw, Mapping):
            return None, ["the answer must be a JSON object"], []
        problems: List[str] = []
        warnings: List[str] = []
        target = normalize_entity(self.spec, raw.get("active_target"))
        if target not in self.spec.targets:
            problems.append(f"active_target must be one of {list(self.spec.targets)}, "
                            f"not {raw.get('active_target')!r}")
        given = raw.get("facts") if isinstance(raw.get("facts"), Mapping) else {}
        by_id = {str(key).split()[0]: value for key, value in given.items() if str(key).strip()}
        labels: List[str] = []
        for index, (fid, fact) in enumerate(zip(self.ids, self.spec.facts)):
            label = by_id.get(fid)
            legal = self.spec.legal_labels(fact.relation)
            if label not in legal:
                problems.append(f"{fid} {fact.label()}: {label!r} is not one of {' | '.join(legal)}")
            elif self.vocab.absolute.encode(label) not in self.seen_ids[index]:
                warnings.append(f"{fid} {fact.label()}={label!r}")
            labels.append(label)
        n_entities = len(self.spec.entities)
        boxes = np.zeros((n_entities, len(CAMERAS), 4), dtype=np.float32)
        visible = np.zeros((n_entities, len(CAMERAS)), dtype=bool)
        kept = {camera: {} for camera in CAMERAS}
        given_boxes = raw.get("boxes") if isinstance(raw.get("boxes"), Mapping) else {}
        for e, entity in enumerate(self.spec.entities):
            for c, camera in enumerate(CAMERAS):
                if entity.id == TABLE:
                    boxes[e, c], visible[e, c] = (0.0, 1.0, 0.0, 1.0), True
                    continue
                per_camera = given_boxes.get(camera)
                if not isinstance(per_camera, Mapping) or entity.id not in per_camera:
                    problems.append(f"boxes.{camera}.{entity.id} is missing; use null when it is not visible")
                    continue
                value = per_camera[entity.id]
                kept[camera][entity.id] = None
                if value is None:
                    continue
                box = _box(value)
                if box is None:
                    problems.append(f"boxes.{camera}.{entity.id}: {value!r} is not [ymin, xmin, ymax, xmax] "
                                    "in 0-1000 with ymin < ymax and xmin < xmax")
                    continue
                kept[camera][entity.id], boxes[e, c] = box
                visible[e, c] = True
        if problems:
            return None, problems, warnings
        return {"target": target, "labels": labels, "boxes": boxes, "visible": visible,
                "answer": self.answer(target, labels, kept)}, [], warnings

    def ask(self, frames: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        """Save the frames, then wait for a valid ``scene_graph.json`` written after them."""
        folder = self.root / f"episode_{self.episode:03d}" / f"step_{self.step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "scene_graph.json"
        # Anything already here answered other frames: a reused exchange folder or an interrupted step.
        for stale in (path, folder / "errors.txt"):
            stale.unlink(missing_ok=True)
        for camera in CAMERAS:
            cv2.imwrite(str(folder / f"{camera}.png"), cv2.cvtColor(frames[camera], cv2.COLOR_RGB2BGR))
        write_json(str(folder / "template.json"), self.previous or self.blank())
        print(f"[inference] step {self.step}: waiting for {path} "
              "(top.png, wrist.png and template.json are beside it)", flush=True)
        stamp = None
        while True:
            current = path.stat().st_mtime_ns if path.is_file() else None
            if current is None or current == stamp:
                time.sleep(self.poll)
                continue
            time.sleep(0.2)
            stamp = path.stat().st_mtime_ns
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                parsed, problems, warnings = None, [f"not readable as JSON: {exc}"], []
            else:
                parsed, problems, warnings = self.read(raw)
            if parsed is not None:
                (folder / "errors.txt").unlink(missing_ok=True)
                return raw
            (folder / "errors.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
            print(f"[inference] step {self.step}: {path.name} rejected; fix it and save again:\n  "
                  + "\n  ".join(problems), flush=True)

    def _warn(self, warnings: Sequence[str]) -> None:
        if warnings:
            print(f"[inference] step {self.step}: labels never seen for these facts in training: "
                  + ", ".join(warnings), flush=True)

    def pack(self, raw: Any) -> Dict[str, np.ndarray]:
        """The training arrays for one answer, with temporal labels from the answers K steps back."""
        from real_robot.preprocessing.annotate_stack_rgb import _temporal

        parsed, problems, warnings = self.read(raw)
        if parsed is None:
            raise ValueError("scene graph rejected:\n  " + "\n  ".join(problems))
        self._warn(warnings)
        labels, window = parsed["labels"], self.spec.temporal_window
        temporal = [_temporal([self.history[-window][i], label], 1, fact.relation)[1]
                    if fact.temporal and len(self.history) >= window else None
                    for i, (fact, label) in enumerate(zip(self.spec.facts, labels))]
        frame = SimpleNamespace(absolute=[[label] for label in labels], temporal=[[label] for label in temporal],
                                active_target=[parsed["target"]])
        graph = build_frame_graph(self.spec, frame, 0, parsed["boxes"], parsed["visible"])
        arrays = pack_graph(graph, self.vocab, n_max=self.spec.n_max, e_max=self.spec.e_max,
                            n_cams=len(self.spec.cameras), use_target_flag=True)
        self.history.append(labels)
        self.previous = parsed["answer"]
        self.step += 1
        return arrays


class _Policy(LatentPolicy):
    feature = None

    def _encode(self, obs):
        self.feature = super()._encode(obs)
        return self.feature


def observation_shapes(cfg: Mapping[str, Any], graph: bool) -> Dict[str, Tuple[int, ...]]:
    """What Stage 1A's probe batch showed the world model, in the same key order."""
    shapes = {f"image_{camera}": (*IMAGE_SIZE, 3) for camera in CAMERAS}
    if graph:
        n, e, c = (int(cfg["model"]["graph"][key]) for key in ("n_max", "e_max", "n_cams"))
        shapes |= {"graph_node_ent": (n,), "graph_node_bbox": (n, c, 4), "graph_node_centroid": (n, 3),
                   "graph_node_target": (n,)}
        shapes |= {key: (e,) for key in GRAPH_KEYS if key.startswith("graph_edge_")}
    shapes["proprio"] = (len(JOINTS),)
    return shapes


class SO101Policy:
    """``model(obs) -> out`` for one SO-101 run folder."""

    def __init__(self, world_model, actor, *, cfg: Mapping[str, Any], normalizer: Normalizer,
                 coords: ActionCoordinates, device: str = "cuda", progress_head=None,
                 channel: Optional[SceneGraphChannel] = None):
        self.cfg = cfg
        self.instruction = str(cfg["task"]["instruction"])
        self.normalizer = normalizer
        self.progress_head = progress_head
        self.channel = channel
        self.policy = _Policy(world_model, actor, device=device, normalizer=normalizer, coords=coords,
                              instruction=self.instruction,
                              execute=int((cfg.get("actor") or {}).get("execute") or 1))
        stats = normalizer.fields["proprio"]
        self.state_range = (np.asarray(stats.low, np.float32), np.asarray(stats.high, np.float32))
        self._told = set()

    @classmethod
    def load(cls, run: str | Path, *, device: str = "cuda", exchange: Optional[str | Path] = None,
             seed: int = 0) -> "SO101Policy":
        from .training.train_imitation import build_actor

        run = Path(run)
        missing = [name for name in ("normalization.json", "world_model.pt", "imitation.pt")
                   if not (run / name).is_file()]
        if missing:
            raise SystemExit(f"{run} has no {missing}; download the run folder from "
                             "TuanDepZai/r2dreamer-graph-so101-checkpoints")
        torch.manual_seed(int(seed))
        stored = json.loads((run / "world_model.json").read_text(encoding="utf-8"))
        cfg = dict(stored["config"])
        cfg["device"] = str(device)
        graph = bool(stored["graph_enabled"])
        model_cfg = load_model_config(
            cfg, model_yaml=REPO / "configs" / "model" / Path(cfg["runtime"]["model_config"]).name)
        normalizer = Normalizer.load(run / "normalization.json")
        normalizer.mode = str(cfg["data"].get("normalization") or "mean_std")
        actions = normalizer.fields["actions"]
        action_dim = len(actions.mean)
        coords = ActionCoordinates(
            normalizer, ActionBounds(low=np.asarray(actions.minimum, np.float32),
                                     high=np.asarray(actions.maximum, np.float32)),
            device=device, action_dim=action_dim)

        world_model = build_world_model(model_cfg, observation_shapes(cfg, graph), action_dim,
                                        graph_enabled=graph).to(device)
        head = build_progress(model_cfg, int(world_model.feature_dim), graph_enabled=graph,
                              progress_enabled=bool(cfg["model"]["progress"]["enabled"]))
        head = head.to(device) if head is not None else None
        meta = CheckpointMeta(graph_enabled=graph, stage="world_model", env_id=str(cfg["task"]["env_id"]),
                              feature_dim=int(world_model.feature_dim),
                              dataset_identity=dict(stored.get("dataset_identity") or {}),
                              normalization_identity=normalizer.descriptor(), config=cfg)
        load(run / "world_model.pt", meta, {"world_model": world_model, "progress": head},
             require_extra={"progress_head": HEAD_IDENTITY if head is not None else None})

        actor = build_actor(cfg, int(world_model.feature_dim), action_dim, device=device)
        load(run / "imitation.pt", replace(meta, stage="imitation",
                                           pretrained_revision=str(actor.loaded.revision)),
             {"adapter": actor.adapter, "actor": actor})
        for module in (world_model, actor, head):
            if module is not None:
                module.eval()

        channel = None
        if graph:
            root = Path(exchange) if exchange else (REPO / "logdir" / "sim_vla" / "real" / "inference"
                                                    / run.name / time.strftime("%Y%m%d_%H%M%S"))
            channel = SceneGraphChannel(cfg, root)
            print(f"[inference] {run.name} needs a scene graph per observation; an LLM agent reads "
                  f"{channel.guide} once, then writes scene_graph.json for every step", flush=True)
        return cls(world_model, actor, cfg=cfg, normalizer=normalizer, coords=coords, device=device,
                   progress_head=head, channel=channel)

    def reset(self) -> None:
        """Call before every episode: the recurrent state and the action queue start empty."""
        self.policy.reset()
        if self.channel is not None:
            self.channel.reset()

    def __call__(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        frames = {camera: _single(obs["video"][camera], f"video.{camera}", (1, -1, -1, 3), np.uint8)
                  for camera in CAMERAS}
        arm = _single(obs["state"]["single_arm"], "state.single_arm", (1, len(JOINTS) - 1))
        gripper = _single(obs["state"]["gripper"], "state.gripper", (1, 1))
        text = (obs.get("language") or {}).get(LANGUAGE_KEY)
        while isinstance(text, (list, tuple, np.ndarray)):
            text = text[0] if len(text) else None
        text = self.instruction if text is None else str(text)
        if text != self.instruction and text not in self._told:
            print(f"[inference] conditioning on {text!r}; this policy was trained only on "
                  f"{self.instruction!r}", flush=True)
            self._told.add(text)
        self.policy.instruction = text

        height, width = IMAGE_SIZE
        stored: Dict[str, np.ndarray] = {
            f"image_{camera}": cv2.resize(np.ascontiguousarray(frames[camera]), (width, height),
                                          interpolation=cv2.INTER_AREA)
            for camera in CAMERAS}
        stored["proprio"] = np.concatenate([arm, gripper]).astype(np.float32)
        out: Dict[str, Any] = {}
        if self.channel is not None:
            raw = obs.get("scene_graph")
            raw = self.channel.ask(frames) if raw is None else raw
            stored |= self.channel.pack(raw)
            out["scene_graph"] = raw
        action = np.asarray(self.policy(stored), dtype=np.float32)
        out["action"] = {"single_arm": action[None, :-1].copy(), "gripper": action[None, -1:].copy()}
        if self.progress_head is not None:
            with torch.no_grad():
                out["progress"] = float(predict(self.progress_head, self.policy.feature)[0])
        return out


def dummy_observation(model: SO101Policy, seed: int = 0) -> Dict[str, Any]:
    """One observation in the layout ``model(obs)`` takes, with the recorded dtypes and units."""
    rng = np.random.default_rng(seed)
    low, high = model.state_range
    state = rng.uniform(low, high).astype(np.float32)

    top_frame = rng.integers(0, 256, (1, 480, 640, 3), dtype=np.uint8)
    wrist_frame = rng.integers(0, 256, (1, 480, 640, 3), dtype=np.uint8)
    arm = state[None, :5]
    gripper = state[None, 5:]
    instruction = model.instruction

    return {
        "video": {
            "top": top_frame,        # (1, 480, 640, 3) uint8, RGB - observation.images.top, fixed camera above the table
            "wrist": wrist_frame,    # (1, 480, 640, 3) uint8, RGB - observation.images.wrist, camera on the gripper
        },
        "state": {
            "single_arm": arm,       # (1, 5) float32 - shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll .pos,
                                     #   the robot's own units as recorded (about -100..100), not radians
            "gripper": gripper,      # (1, 1) float32 - gripper.pos, 0..100
        },
        "language": {
            LANGUAGE_KEY: [[instruction]],   # the task text the checkpoint was trained with
        },
    }


def observation_from_robot(robot_obs: Mapping[str, Any], instruction: str) -> Dict[str, Any]:
    """``robot.get_observation()`` of a LeRobot SO-101 follower with cameras named top and wrist."""
    return {
        "video": {camera: np.asarray(robot_obs[camera], dtype=np.uint8)[None] for camera in CAMERAS},
        "state": {"single_arm": np.asarray([[robot_obs[joint] for joint in JOINTS[:-1]]], dtype=np.float32),
                  "gripper": np.asarray([[robot_obs[JOINTS[-1]]]], dtype=np.float32)},
        "language": {LANGUAGE_KEY: [[instruction]]},
    }


def action_to_robot(out: Mapping[str, Any]) -> Dict[str, float]:
    """``out["action"]`` as the dict ``robot.send_action`` takes."""
    values = np.concatenate([out["action"]["single_arm"][0], out["action"]["gripper"][0]])
    return {joint: float(value) for joint, value in zip(JOINTS, values)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Step an SO-101 checkpoint on dummy observations")
    parser.add_argument("--checkpoint", required=True,
                        help="a run folder holding normalization.json, world_model.pt and imitation.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--exchange", default=None,
                        help="where frames and scene graphs are exchanged; default "
                             "logdir/sim_vla/real/inference/<run>/<time>")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model = SO101Policy.load(args.checkpoint, device=args.device, exchange=args.exchange, seed=args.seed)
    for step in range(args.steps):
        obs = dummy_observation(model, seed=args.seed + step)
        out = model(obs)
        action = " ".join(f"{joint}={value:.2f}" for joint, value in action_to_robot(out).items())
        progress = "" if out.get("progress") is None else f" progress={out['progress']:.3f}"
        print(f"[inference] step {step}: {action}{progress}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
