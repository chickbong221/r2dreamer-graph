"""Output layout for a two-camera figure: head, wrist, and one graph per step.

:mod:`scenegraph.figures.writer` writes one camera per step, which is the right
shape for the single-view PlaceSphere figure and the wrong shape here: a
two-camera figure needs the head and the wrist view of the *same* instant, and a
reader has to be able to tell which is which from the path alone. So the frames
live under ``frames/head`` and ``frames/wrist`` and share an index with the one
graph, rather than a flat ``frames`` directory a second camera would have to be
interleaved into.

Only one graph is written per index on purpose. The wrist view is there to show
what the arm sees, not to contribute nodes -- the graph is built from the head
camera alone, so a second diagram would either duplicate this one or claim a
second graph exists.

Same staging discipline as the single-view writer, for the same reason: success
is a fact about the whole episode, the frames are far too large to hold until it
is known, and a half-written directory is indistinguishable from a kept one.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.schema import Graph
from ..viz.graph_draw import render_graph
from ..viz.palette import ColorMap
# The two file-writing conventions -- lossless uint8 PNG, and JSON that
# tolerates whatever numpy scalar a relation carried in from PhysX -- are
# already settled for figures. Re-deriving them here would let two figures from
# the same repo disagree about what a graph dump looks like.
from .writer import save_json, save_png

FRAME_DIR = "frames"
HEAD_DIR = "head"
WRIST_DIR = "wrist"
GRAPH_DIR = "graphs"
GRAPH_JSON_DIR = "graph_json"
MANIFEST = "episode.json"
# Distinct from the single-view writer's ``.staging_`` prefix so the two tools
# can never stage, or delete, each other's work.
STAGING_PREFIX = ".staging_multicam_"


def episode_path(root: Path | str, name: str) -> Path:
    """Where a committed episode lands.

    Public because the caller needs it before the episode runs: refusing a
    collision after a successful rollout has already been rendered wastes the
    rollout, and refusing it from inside a capture hook would be reported as a
    planning failure instead of a naming one.
    """
    return Path(root) / str(name)


class MulticameraEpisodeWriter:
    """Accumulates one episode's two-camera exports, then commits or drops them."""

    def __init__(
        self,
        root: Path | str,
        name: str,
        *,
        frame_size: Optional[Sequence[int]] = None,
        graph_image: bool = True,
        graph_json: bool = True,
        paper_graph_style: bool = True,
        overwrite: bool = False,
    ):
        # A name carrying a separator would point ``commit``'s replace step at
        # some directory other than a child of ``root``; rejected at
        # construction so the deletion below cannot be aimed by a caller.
        if not name or Path(name).name != name:
            raise ValueError(
                f"episode name must be a single directory name, got {name!r}"
            )
        self.root = Path(root)
        self.name = str(name)
        self.frame_size = None if frame_size is None else tuple(
            int(v) for v in frame_size
        )
        self.graph_image = bool(graph_image)
        self.graph_json = bool(graph_json)
        self.paper_graph_style = bool(paper_graph_style)
        self.overwrite = bool(overwrite)
        self.staging = self.root / f"{STAGING_PREFIX}{self.name}"
        self.records: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def count(self) -> int:
        return self._index

    @property
    def final(self) -> Path:
        return episode_path(self.root, self.name)

    def open(self) -> None:
        """Start (or restart) the staging directory for this episode."""
        self._refuse_existing()
        shutil.rmtree(self.staging, ignore_errors=True)
        subs = [f"{FRAME_DIR}/{HEAD_DIR}", f"{FRAME_DIR}/{WRIST_DIR}"]
        if self.graph_image:
            subs.append(GRAPH_DIR)
        if self.graph_json:
            subs.append(GRAPH_JSON_DIR)
        for sub in subs:
            (self.staging / sub).mkdir(parents=True, exist_ok=True)
        self.records = []
        self._index = 0

    def write_step(
        self,
        *,
        step: int,
        head: np.ndarray,
        wrist: np.ndarray,
        graph: Graph,
        colormap: Optional[ColorMap] = None,
    ) -> int:
        """Export both views of one instant and their graph.

        Returns the shared export index. The two frames and the graph are
        written under the same stem in the same call, so a figure can never pair
        a head frame with the wrist frame of a different step.
        """
        self._check_frame("head", head)
        self._check_frame("wrist", wrist)
        stem = f"{self._index:04d}"
        record: Dict[str, Any] = {
            "index": self._index,
            "step": int(step),
            "head": f"{FRAME_DIR}/{HEAD_DIR}/frame_{stem}.png",
            "wrist": f"{FRAME_DIR}/{WRIST_DIR}/frame_{stem}.png",
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
        }
        save_png(self.staging / FRAME_DIR / HEAD_DIR / f"frame_{stem}.png", head)
        save_png(self.staging / FRAME_DIR / WRIST_DIR / f"frame_{stem}.png", wrist)
        if self.graph_image:
            render_graph(
                graph,
                str(self.staging / GRAPH_DIR / f"graph_{stem}.png"),
                colormap=colormap,
                paper_style=self.paper_graph_style,
            )
            record["graph"] = f"{GRAPH_DIR}/graph_{stem}.png"
        if self.graph_json:
            save_json(
                self.staging / GRAPH_JSON_DIR / f"graph_{stem}.json",
                graph.to_dict(),
            )
            record["graph_json"] = f"{GRAPH_JSON_DIR}/graph_{stem}.json"
        self.records.append(record)
        self._index += 1
        return self._index - 1

    def commit(self, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Publish the staged episode under its real name and return the path."""
        self._refuse_existing()
        payload = dict(metadata or {})
        payload["name"] = self.name
        payload["exported_frames"] = len(self.records)
        payload["steps"] = self.records
        save_json(self.staging / MANIFEST, payload)
        final = self.final
        if final.exists():
            self._replace(final)
        self.staging.rename(final)
        return final

    def discard(self) -> None:
        """Delete everything this episode staged. Safe to call twice."""
        shutil.rmtree(self.staging, ignore_errors=True)
        self.records = []
        self._index = 0

    # ------------------------------------------------------------- internals
    def _refuse_existing(self) -> None:
        """Stop before an episode nobody asked to replace is replaced.

        ``data/paper_figures`` holds figures that have already been placed in a
        paper; overwriting one silently would change a published picture and
        leave no trace of what it used to be.
        """
        if self.overwrite:
            return
        final = self.final
        if final.exists():
            raise FileExistsError(
                f"{final} already exists; pass --overwrite to replace it, or "
                "write to a different --out"
            )

    def _replace(self, final: Path) -> None:
        """Remove the episode being overwritten, and nothing else.

        The two facts that make this ``rmtree`` safe -- it is a directory, and
        it is the direct child of ``root`` this writer is named for -- are
        checked rather than assumed. Without them a name assembled elsewhere
        could aim the deletion at ``data/paper_figures`` itself.
        """
        if final.parent != self.root or final.name != self.name:
            raise RuntimeError(
                f"refusing to remove {final}: not the episode directory "
                f"{self.name!r} under {self.root}"
            )
        if not final.is_dir():
            raise RuntimeError(f"refusing to remove {final}: not a directory")
        shutil.rmtree(final)

    def _check_frame(self, role: str, image: np.ndarray) -> None:
        """Reject a frame that is not the exact picture the figure was sized for.

        The whole point of this exporter is two raw sensor frames at one fixed
        resolution: nothing crops, resizes or annotates them. A frame arriving
        at another size means the env was built with different sensor configs,
        and the failure should name that rather than produce a figure whose two
        panels do not line up.
        """
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{role} frame must be [H, W, 3] RGB, got shape {arr.shape}"
            )
        if self.frame_size is not None and arr.shape[:2] != self.frame_size:
            raise ValueError(
                f"{role} frame is {arr.shape[:2]}, expected {self.frame_size}"
            )
