"""Paper figures for a ManiSkill task: labelled stills beside their graphs.

Drives the task's own scripted motion-planning solution, keeps the episodes that
succeeded, and writes -- per exported step -- the human-view camera frame with
one name chip per entity, the node-link diagram, and the graph as JSON. The
frame and the diagram are separate files: unlike the eval video, nothing here
composites them.

    python -m scenegraph.tools.render_paper_frames \
        --env-id PlaceSphere-v1 --out data/paper_figures --episodes 1

Cost lives almost entirely in the two renderers, so ``--stride`` is the knob
that matters: the diagram is a matplotlib figure at 1200px and the frame is a
1000px PNG, while the graph itself is rebuilt every step regardless -- temporal
relation labels difference over the last K frames and would be wrong otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scenegraph.figures.annotate import (
    DEFAULT_LABELS, build_callouts, draw_callouts, fixed_labels,
    label_every_node,
)
from scenegraph.figures.graph_source import FigureGraphSource
from scenegraph.figures.render_camera import (
    DEFAULT_RENDER_SIZE, DEFAULT_SENSOR_SIZE, FigureCamera, make_figure_env,
)
from scenegraph.figures.rollout import MotionPlanRunner
from scenegraph.figures.writer import EpisodeWriter
from scenegraph.viz.palette import ColorMap


class FigureSession:
    """Per-episode capture state, wired into the runner's two hooks.

    Everything here is scoped to one attempt: a fresh colour map so a node keeps
    one colour across the episode's frames and diagrams, and a fresh staging
    directory that is committed only if the attempt succeeds.
    """

    def __init__(self, args, camera: FigureCamera, graphs: FigureGraphSource):
        self.args = args
        self.camera = camera
        self.graphs = graphs
        self.root = Path(args.out)
        self.labels = (
            label_every_node if args.labels_all else fixed_labels(args.labels)
        )
        self.seed = int(args.seed)
        self.writer: Optional[EpisodeWriter] = None
        self.colormap = ColorMap()

    # ------------------------------------------------------------- the hooks
    def on_reset(self, obs: dict) -> None:
        """A new episode: drop whatever the last one staged and start over.

        The reset observation is exported as frame 0. It is the scene before the
        robot has touched it, which is the one still a figure almost always
        wants, and it is also the graph's own frame 0.
        """
        self.close(commit=False)
        self.graphs.on_reset()
        self.camera.invalidate()
        self.colormap = ColorMap()
        self.writer = EpisodeWriter(
            self.root,
            f"{self.args.env_id}_seed{self.seed:04d}",
            save_clean=self.args.save_clean,
            graph_image=not self.args.no_graph_image,
            graph_json=not self.args.no_graph_json,
        )
        self.writer.open()
        self._export(self.graphs.step(obs))

    def on_step(self, obs: dict, _info: dict) -> None:
        """Every control step builds a graph; ``--stride`` decides what is kept."""
        graph = self.graphs.step(obs)
        if self.args.stride > 0 and graph.frame % self.args.stride == 0:
            self._export(graph)

    # ------------------------------------------------------------- the export
    def _export(self, graph) -> None:
        if self.writer is None:
            return
        if self.args.max_frames and self.writer.count >= self.args.max_frames:
            return
        frame = self.camera.capture()
        callouts: List = []
        labelled = frame
        if not self.args.no_labels:
            callouts = build_callouts(
                graph, self.camera, self.graphs.entities,
                labels=self.labels, colormap=self.colormap,
                offsets=self.args.label_offsets,
            )
            labelled = draw_callouts(
                frame, callouts,
                font_size=self.args.font_size,
                lift=self.args.label_lift,
                draw_boxes=self.args.draw_boxes,
            )
        self.writer.write_step(
            step=graph.frame, frame=labelled, graph=graph, colormap=self.colormap,
            clean=frame if self.args.save_clean else None,
            callouts=callouts,
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
            "attempt": None if attempt is None else attempt.to_dict(),
            "render_size": list(self.args.render_size),
            "sensor_size": list(self.args.sensor_size),
            "camera": self.camera.name,
            "control_mode": self.args.control_mode,
            "shader": self.args.shader or "default",
            "stride": self.args.stride,
            "labels": (
                "every-node" if self.args.labels_all
                else (self.args.labels or DEFAULT_LABELS)
            ),
            "graph_cameras": self.graphs.cameras,
            "whitelist_dir": self.graphs.whitelist_dir,
        })


def run(args) -> int:
    env = make_figure_env(
        args.env_id,
        render_size=args.render_size,
        sensor_size=args.sensor_size,
        control_mode=args.control_mode,
        shader=args.shader,
    )
    camera = FigureCamera(env)
    graphs = FigureGraphSource(
        env,
        env_id=args.env_id,
        cameras=args.cameras or None,
        thresholds_path=args.thresholds,
        whitelist_dir=args.whitelist_dir,
        use_target_flag=args.use_target_flag,
        object_object_spatial=not args.no_object_object_spatial,
    )
    session = FigureSession(args, camera, graphs)
    runner = MotionPlanRunner(
        env, args.env_id, on_reset=session.on_reset, on_step=session.on_step
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    kept: List[Path] = []
    seed, attempts = int(args.seed), 0
    try:
        while len(kept) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            session.seed = seed
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
            f"scripted solution for {args.env_id} runs on this install"
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_map(values: Optional[List[str]], what: str) -> Dict[str, str]:
    """``node_id=value`` pairs. Split on the first ``=``: node ids carry a
    colon (``actor:sphere``) but never an equals sign."""
    out: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--{what} expects node_id=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_offset(node_id: str, value: str) -> List[float]:
    try:
        dx, dy = (float(v) for v in value.split(","))
    except ValueError:
        raise SystemExit(
            f"--label-offset {node_id} expects DX,DY in pixels, got {value!r}"
        ) from None
    return [dx, dy]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render labelled paper frames + scene graphs from a "
                    "successful motion-planning episode",
    )
    p.add_argument("--env-id", default="PlaceSphere-v1")
    p.add_argument("--out", default="data/paper_figures")
    p.add_argument("--episodes", type=int, default=1,
                   help="successful episodes to keep")
    p.add_argument("--seed", type=int, default=0, help="first seed to try")
    p.add_argument("--max-attempts", type=int, default=25)

    p.add_argument("--render-size", type=int, nargs=2,
                   default=list(DEFAULT_RENDER_SIZE), metavar=("H", "W"),
                   help="human-view camera resolution")
    p.add_argument("--sensor-size", type=int, nargs=2,
                   default=list(DEFAULT_SENSOR_SIZE), metavar=("H", "W"),
                   help="segmentation sensor resolution the graph is built from")
    p.add_argument("--shader", default="",
                   help="render-camera shader, e.g. rt-fast or rt; "
                        "empty keeps the task's default rasterizer")
    p.add_argument("--control-mode", default="pd_joint_pos",
                   help="the scripted solutions are written for pd_joint_pos")

    p.add_argument("--stride", type=int, default=1,
                   help="export every Nth control step (frame 0 is always kept)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop exporting after this many frames; 0 is unlimited")
    p.add_argument("--save-clean", action="store_true",
                   help="also write the unlabelled frame")
    p.add_argument("--no-graph-image", action="store_true",
                   help="skip the node-link PNG; the JSON is still written")
    p.add_argument("--no-graph-json", action="store_true")

    p.add_argument("--no-labels", action="store_true",
                   help="render the frame with no name chips at all")
    p.add_argument("--labels-all", action="store_true",
                   help="label every vertex, not just the figure's four")
    p.add_argument("--label", action="append", metavar="NODE_ID=TEXT",
                   help="add to or rename in the default label map, e.g. "
                        "--label actor:sphere=ball; an empty text drops that "
                        "node's chip (repeatable)")
    p.add_argument("--label-offset", action="append", metavar="NODE_ID=DX,DY",
                   help="nudge one chip in pixels, for figure tuning "
                        "(repeatable)")
    p.add_argument("--font-size", type=int, default=16,
                   help="camera-frame label text size in px; pass 0 to scale "
                        "with the frame")
    p.add_argument("--label-lift", type=float, default=0.0,
                   help="px from the object to its chip; 0 scales with frame")
    p.add_argument("--draw-boxes", action="store_true",
                   help="also outline each labelled object's projected AABB")

    p.add_argument("--cameras", nargs="*", default=None,
                   help="segmentation cameras for the graph; default is every "
                        "sensor the task renders")
    p.add_argument("--thresholds", default="",
                   help="thresholds.yaml override; empty uses the packaged one")
    p.add_argument("--whitelist-dir", default="",
                   help="mined whitelist dir override")
    p.add_argument("--use-target-flag", action="store_true",
                   help="MS-HAB style target flag; off for normal ManiSkill")
    p.add_argument("--no-object-object-spatial", action="store_true",
                   help="drop object-object spatial edges from the graph")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Merged onto the defaults rather than replacing them: the usual reason to
    # pass one is to rename a single chip, and replacing would silently drop
    # the other three. An empty text removes a node from the map.
    overrides = _parse_map(args.label, "label")
    args.labels = dict(DEFAULT_LABELS, **overrides) if overrides else None
    args.label_offsets = {
        key: _parse_offset(key, value)
        for key, value in _parse_map(args.label_offset, "label-offset").items()
    }
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
