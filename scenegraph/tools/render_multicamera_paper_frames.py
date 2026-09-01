"""Two-camera paper frames: the head view, the wrist view, and one graph.

``render_paper_frames`` writes the third-person human-render camera with a name
chip on every entity -- a picture of what the graph *means*. This writes
something different and deliberately plainer: the two raw sensor images a
wrist-camera robot actually returns, at figure resolution, beside the graph
built from the head one. Nothing is annotated, cropped or resized, because the
point of the figure is what the cameras saw.

Only the head camera feeds the graph. The wrist camera is a second picture of
the same instant, not a second source of nodes: it swings with the gripper and
sees the peg from a hand's width away, so admitting it would add and drop nodes
on arm motion alone and the diagram would stop describing the scene.

The exported diagram also drops the hole site, for the same reason the figure
exists at all: every other vertex is something the reader can find in the frame
beside it, and that one is a pose with nothing to look at. It is dropped from
what this tool writes and nowhere else -- the builder still emits it, and the
graph the model reads is unchanged.

    python -m scenegraph.tools.render_multicamera_paper_frames \
        --env-id PegInsertionSide-v1 --robot-uids panda_wristcam \
        --out data/paper_figures --episodes 1

``panda_wristcam`` is not decoration: plain ``panda`` renders no ``hand_camera``,
so there is no wrist view to export. Everything else about the rollout matches
the miner -- one CPU env, ``pd_joint_pos``, the task's own scripted solution --
because the graph in the figure has to be the graph the pipeline builds.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scenegraph.core.mask_extractor import extract_camera_obs
from scenegraph.core.schema import Graph
from scenegraph.core.sites import SITE_HOLE
from scenegraph.figures.graph_source import FigureGraphSource
from scenegraph.figures.multicamera_writer import (
    MulticameraEpisodeWriter, episode_path,
)
from scenegraph.figures.rollout import MotionPlanRunner
from scenegraph.viz.palette import ColorMap

# 500x500 prints at ~1.7in / 300dpi per panel, so the two views sit side by side
# in one column. Unlike the 128px mining sensors this is also the size the graph
# is extracted at, since here they are the same cameras.
PAPER_SENSOR_SIZE = (500, 500)
# Roles are aliases the caller sets, never inferred from the sensor list: which
# camera is "the wrist" is a fact about the robot, and a task that renamed its
# cameras should fail the preflight rather than have one guessed for it.
DEFAULT_HEAD_CAMERA = "base_camera"
DEFAULT_WRIST_CAMERA = "hand_camera"
DEFAULT_ROBOT_UIDS = "panda_wristcam"
# The scripted solutions read ``pose.sp``, which only exists unbatched, so the
# env is single-environment on CPU and there is exactly one row to unwrap.
ENV_IDX = 0
# Dropped from the figure by default. The hole site is a synthetic vertex --
# no pixels, no collision body, nothing in either camera to point at -- and it
# earns its place in the graph the model reads, where the insertion milestone
# is scored against it. In a printed diagram it is a vertex the reader cannot
# find in the picture beside it.
DEFAULT_HIDDEN_NODES = (SITE_HOLE,)


def make_multicamera_env(
    env_id: str,
    *,
    robot_uids: str = DEFAULT_ROBOT_UIDS,
    sensor_size: Sequence[int] = PAPER_SENSOR_SIZE,
    control_mode: str = "pd_joint_pos",
    obs_mode: str = "rgb+segmentation",
    sim_backend: str = "cpu",
):
    """A single-env ManiSkill task whose sensors are the figure's cameras.

    ``figures.render_camera.make_figure_env`` cannot serve here: it renders the
    figure from the human-render camera and leaves the sensors at mask
    resolution, and it has no way to ask for a robot with a wrist camera.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers the tasks

    kwargs: Dict[str, Any] = dict(
        id=env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        robot_uids=robot_uids,
        render_mode="rgb_array",
        sim_backend=sim_backend,
        sensor_configs=dict(
            width=int(sensor_size[1]), height=int(sensor_size[0])
        ),
    )
    try:
        return gym.make(**kwargs)
    except Exception as exc:                               # noqa: BLE001
        raise SystemExit(
            f"could not build {env_id} with robot_uids={robot_uids!r} "
            f"({type(exc).__name__}: {exc}). This exporter does not patch the "
            "task or the planner to make a robot fit; pick a robot the task "
            "supports, or report the incompatibility."
        ) from exc


def preflight(
    obs: dict,
    graphs: FigureGraphSource,
    *,
    head: str,
    wrist: str,
    graph_camera: str,
    sensor_size: Sequence[int],
) -> None:
    """Check the whole camera contract against the first observation.

    Every one of these is silent if it is left to fail later: a missing wrist
    camera reads as a KeyError forty steps into a rollout, a sensor built at the
    task's default size produces a figure whose panels are half the intended
    resolution, and a graph camera list that quietly picked up the wrist would
    produce a plausible-looking diagram of the wrong thing. All problems are
    collected so one run reports the whole mismatch.

    The remaining precondition -- that the task has a scripted solution -- is
    checked by ``MotionPlanRunner``'s constructor, which the caller builds
    before the reset that produced ``obs``.
    """
    sensor_data = obs.get("sensor_data") if isinstance(obs, dict) else None
    if not sensor_data:
        raise SystemExit(
            "preflight: the observation carries no sensor_data; the env has to "
            "be built with an obs_mode that includes rgb and segmentation"
        )

    height, width = (int(v) for v in sensor_size)
    problems: List[str] = []
    if head == wrist:
        problems.append(
            f"head and wrist are the same camera ({head!r}); the figure needs "
            "two views of the same instant"
        )
    for role, camera in (("head", head), ("wrist", wrist)):
        if camera not in sensor_data:
            problems.append(
                f"{role} camera {camera!r} is not rendered by this env; "
                f"have {sorted(sensor_data)}"
            )
            continue
        missing = [k for k in ("rgb", "segmentation") if k not in sensor_data[camera]]
        if missing:
            problems.append(
                f"{role} camera {camera!r} has no {' or '.join(missing)}"
            )
            continue
        rgb, seg, _depth = extract_camera_obs(obs, camera, ENV_IDX)
        if rgb.shape != (height, width, 3):
            problems.append(
                f"{role} camera {camera!r} renders {rgb.shape} rgb, expected "
                f"{(height, width, 3)}"
            )
        if seg.shape != (height, width):
            problems.append(
                f"{role} camera {camera!r} renders {seg.shape} segmentation, "
                f"expected {(height, width)}"
            )

    cameras = list(graphs.cameras)
    if cameras != [graph_camera]:
        problems.append(
            f"the graph would be built from {cameras}, expected exactly "
            f"[{graph_camera!r}]"
        )
    if graph_camera == wrist:
        problems.append(
            f"the graph camera is the wrist camera ({wrist!r}); nodes would "
            "appear and vanish with arm motion alone"
        )
    if graph_camera not in sensor_data:
        problems.append(
            f"graph camera {graph_camera!r} is not rendered by this env; "
            f"have {sorted(sensor_data)}"
        )

    if problems:
        raise SystemExit(
            "preflight failed:\n  - " + "\n  - ".join(problems)
        )


def without_nodes(graph: Graph, hidden: Sequence[str]) -> Graph:
    """A copy of ``graph`` without these vertices, or the graph itself.

    A copy, never an edit in place: the builder keeps its own graph and the next
    frame's temporal labels difference against it, so removing a vertex there
    would change the relations the *following* graph reports. Only what this
    exporter writes loses the vertex -- extraction, and everything downstream of
    it, sees the graph the builder built.

    Every edge touching a dropped vertex goes with it, and the three node counts
    in ``meta`` are recomputed, because they are derived: left alone they would
    describe a graph one vertex larger than the one in the file.
    """
    drop = {str(node_id) for node_id in hidden if node_id}
    present = drop & set(graph.node_ids())
    if not present:
        return graph
    nodes = [n for n in graph.nodes if n.node_id not in present]
    edges = [
        e for e in graph.edges
        if e.src not in present and e.dst not in present
    ]
    meta = dict(graph.meta)
    meta.update(
        # The same three sums ``GraphBuilder`` takes over its own node list.
        n_objects=sum(1 for n in nodes if n.node_type == "object"),
        n_visible=sum(1 for n in nodes if n.visible),
        n_in_frame=sum(1 for n in nodes if n.in_frame),
        # What is missing, recorded in the artifact itself: a reader comparing
        # this dump against a graph from the training path has to be able to
        # see that the difference was asked for.
        hidden_nodes=sorted(present),
    )
    return replace(graph, nodes=nodes, edges=edges, meta=meta)


class MulticameraSession:
    """Per-episode capture state, wired into the runner's two hooks.

    One colour map per episode so a node keeps its colour across the episode's
    diagrams, and one staging directory that is committed only if the scripted
    attempt succeeded.
    """

    def __init__(self, args, graphs: FigureGraphSource):
        self.args = args
        self.graphs = graphs
        self.root = Path(args.out)
        self.roles = {"head": args.head_camera, "wrist": args.wrist_camera}
        self.seed = int(args.seed)
        self.writer: Optional[MulticameraEpisodeWriter] = None
        self.colormap = ColorMap()

    @property
    def episode_name(self) -> str:
        return f"{self.args.env_id}_seed{self.seed:04d}"

    def prepare(self, seed: int) -> None:
        """Claim this seed's output name before the attempt runs.

        The refusal belongs here and not in ``on_reset``: the runner turns any
        exception raised inside a capture hook into a failed attempt, so a name
        collision would be reported as a planning failure and the loop would
        walk to the next seed and hit the same wall.
        """
        self.seed = int(seed)
        path = episode_path(self.root, self.episode_name)
        if path.exists() and not self.args.overwrite:
            raise SystemExit(
                f"{path} already exists; pass --overwrite to replace it, or "
                "write to a different --out"
            )

    # ------------------------------------------------------------- the hooks
    def on_reset(self, obs: dict) -> None:
        """A new episode: drop whatever the last one staged and start over.

        The reset observation is exported as index 0 -- the scene before the arm
        has touched it, and the graph's own frame 0.
        """
        self.close(commit=False)
        self.graphs.on_reset()
        self.colormap = ColorMap()
        self.writer = MulticameraEpisodeWriter(
            self.root,
            self.episode_name,
            frame_size=self.args.sensor_size,
            graph_image=not self.args.no_graph_image,
            graph_json=not self.args.no_graph_json,
            overwrite=self.args.overwrite,
        )
        self.writer.open()
        self._export(obs, self.graphs.step(obs))

    def on_step(self, obs: dict, _info: dict) -> None:
        """Every control step builds a graph; ``--stride`` decides what is kept.

        The graph is rebuilt on every step even when almost none are exported:
        temporal relation labels difference over the last K frames, so a builder
        fed one step in five would report changes spanning five times the
        horizon they were mined for.
        """
        graph = self.graphs.step(obs)
        if self.args.stride > 0 and graph.frame % self.args.stride == 0:
            self._export(obs, graph)

    # ------------------------------------------------------------ the export
    def _export(self, obs: dict, graph) -> None:
        """Both views of this observation, and the graph built from it.

        The frames are pulled from the same ``obs`` the graph was built from
        rather than re-rendered, so the three files cannot describe three
        slightly different instants.
        """
        if self.writer is None:
            return
        if self.args.max_frames and self.writer.count >= self.args.max_frames:
            return
        head, _seg, _depth = extract_camera_obs(obs, self.roles["head"], ENV_IDX)
        wrist, _wseg, _wdepth = extract_camera_obs(
            obs, self.roles["wrist"], ENV_IDX
        )
        # Filtered once, so the diagram and the JSON beside it hold the same
        # vertices; a caption written from one would otherwise describe a node
        # missing from the other.
        self.writer.write_step(
            step=graph.frame, head=head, wrist=wrist,
            graph=without_nodes(graph, self.args.hidden_nodes),
            colormap=self.colormap,
        )

    def close(self, *, commit: bool, attempt=None) -> Optional[Path]:
        """Publish the staged episode, or delete it. Returns the kept path."""
        writer, self.writer = self.writer, None
        if writer is None:
            return None
        if not commit:
            writer.discard()
            return None
        return writer.commit({
            "env_id": self.args.env_id,
            "robot_uids": self.args.robot_uids,
            "sensor_size": list(self.args.sensor_size),
            "frame_cameras": dict(self.roles),
            "graph_cameras": self.graphs.cameras,
            # Stated rather than implied: this exporter never runs the callout
            # pipeline, so a reader of the manifest knows the frames are raw
            # sensor pixels and not the labelled stills the other figure writes.
            "frame_annotations": False,
            "hidden_nodes": list(self.args.hidden_nodes),
            "attempt": None if attempt is None else attempt.to_dict(),
            "control_mode": self.args.control_mode,
            "stride": self.args.stride,
            "whitelist_dir": self.graphs.whitelist_dir,
        })


def run(args) -> int:
    env = make_multicamera_env(
        args.env_id,
        robot_uids=args.robot_uids,
        sensor_size=args.sensor_size,
        control_mode=args.control_mode,
    )
    graphs = FigureGraphSource(
        env,
        env_id=args.env_id,
        # The one camera the graph is allowed to see. Passed explicitly, so the
        # source never falls back to "every sensor the task renders" and picks
        # up the wrist.
        cameras=[args.graph_camera],
        thresholds_path=args.thresholds,
        whitelist_dir=args.whitelist_dir,
        # Normal ManiSkill: no MS-HAB target flag, and object-object spatial
        # edges are emitted. Same values ``configs/env/maniskill.yaml`` sets.
        use_target_flag=False,
        object_object_spatial=True,
    )
    session = MulticameraSession(args, graphs)
    # Built before the first reset: its constructor is what proves the task has
    # a scripted solution, and failing that is cheaper than reconfiguring a
    # scene first.
    runner = MotionPlanRunner(
        env, args.env_id, on_reset=session.on_reset, on_step=session.on_step
    )
    # Reset through the raw env rather than the runner's wrapper, so the
    # preflight observation reaches the checks without the capture hooks
    # writing an episode for it.
    obs, _info = env.reset(seed=args.seed)
    preflight(
        obs, graphs,
        head=args.head_camera, wrist=args.wrist_camera,
        graph_camera=args.graph_camera, sensor_size=args.sensor_size,
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    kept: List[Path] = []
    seed, attempts = int(args.seed), 0
    try:
        while len(kept) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            session.prepare(seed)
            attempt = runner.attempt(seed)
            seed += 1
            path = session.close(commit=attempt.success, attempt=attempt)
            status = "kept" if path else "dropped"
            note = f" ({attempt.error})" if attempt.error else ""
            print(
                f"[{attempts}] seed={attempt.seed} success={attempt.success} "
                f"steps={attempt.steps} -> {status}{note}",
                flush=True,
            )
            if path is not None:
                kept.append(path)
    except KeyboardInterrupt:
        session.close(commit=False)
        print("\ninterrupted; staged episode discarded", flush=True)
    finally:
        session.close(commit=False)
        env.close()

    print(f"\n{len(kept)}/{args.episodes} episodes written to {args.out}")
    for path in kept:
        print(f"  {path}")
    if not kept:
        print(
            "no episode succeeded; raise --max-attempts, or check that the "
            f"scripted solution for {args.env_id} runs on "
            f"robot_uids={args.robot_uids} on this install"
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render head + wrist camera frames and the head-camera "
                    "scene graph from a successful motion-planning episode",
    )
    p.add_argument("--env-id", default="PegInsertionSide-v1")
    p.add_argument("--robot-uids", default=DEFAULT_ROBOT_UIDS,
                   help="the wrist view exists only on a wrist-camera robot")
    p.add_argument("--out", default="data/paper_figures")
    p.add_argument("--episodes", type=int, default=1,
                   help="successful episodes to keep")
    p.add_argument("--seed", type=int, default=0, help="first seed to try")
    p.add_argument("--max-attempts", type=int, default=25)
    p.add_argument("--overwrite", action="store_true",
                   help="replace an episode directory that already exists; "
                        "off by default so a figure already in a paper cannot "
                        "be rewritten by accident")

    p.add_argument("--sensor-size", type=int, nargs=2,
                   default=list(PAPER_SENSOR_SIZE), metavar=("H", "W"),
                   help="resolution of both cameras; the frames are exported "
                        "at exactly this size, uncropped and unresized")
    p.add_argument("--control-mode", default="pd_joint_pos",
                   help="the scripted solutions are written for pd_joint_pos")

    p.add_argument("--head-camera", default=DEFAULT_HEAD_CAMERA,
                   help="sensor exported as the head view")
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA,
                   help="sensor exported as the wrist view")
    p.add_argument("--graph-camera", default=DEFAULT_HEAD_CAMERA,
                   help="the single sensor the graph is built from")

    p.add_argument("--stride", type=int, default=1,
                   help="export every Nth control step (frame 0 is always kept)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop exporting after this many frames; 0 is unlimited")
    p.add_argument("--no-graph-image", action="store_true",
                   help="skip the node-link PNG; the JSON is still written")
    p.add_argument("--no-graph-json", action="store_true")
    p.add_argument("--hide-node", action="append", metavar="NODE_ID",
                   help="drop this vertex and its edges from the exported "
                        f"graph (repeatable); defaults to {SITE_HOLE}, which "
                        "no camera can show. Pass an empty --hide-node '' to "
                        "export every vertex the builder produced")

    p.add_argument("--thresholds", default="",
                   help="thresholds.yaml override; empty uses the packaged one")
    p.add_argument("--whitelist-dir", default="",
                   help="mined whitelist dir override")
    args = p.parse_args(argv)
    # Resolved here rather than in ``main`` so every caller gets a namespace
    # that already says what the export hides. Replaces the default rather than
    # extending it: a run naming its own vertices is choosing the whole list,
    # and ``--hide-node ''`` is how that list is emptied.
    args.hidden_nodes = (
        list(DEFAULT_HIDDEN_NODES) if args.hide_node is None
        else [n.strip() for n in args.hide_node if n.strip()]
    )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stride < 1:
        raise SystemExit(
            f"--stride must be at least 1, got {args.stride}; to export only "
            "the reset frame use --max-frames 1"
        )
    if args.no_graph_image and args.no_graph_json:
        raise SystemExit(
            "--no-graph-image with --no-graph-json exports no scene graph at "
            "all; drop one of them"
        )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
