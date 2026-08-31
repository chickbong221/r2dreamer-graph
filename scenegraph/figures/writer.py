"""Output layout: one directory per kept episode, frames and graphs apart.

The eval video composites the node-link diagram onto the camera strip, which is
right for a video and wrong for a paper: a figure wants to place the two at
different sizes, crop one, or use either alone. So a step writes three files --
the labelled frame, the graph diagram, the graph as data -- that share only an
index, and never one file carrying both pictures.

An episode is written into a staging directory and renamed on success. Success
is a fact about the whole episode and the frames are far too large to hold in
memory until it is known, so the alternative to staging is a half-written
episode directory that looks exactly like a good one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.schema import Graph
from ..viz.graph_draw import render_graph
from ..viz.palette import ColorMap

FRAME_DIR = "frames"
CLEAN_DIR = "frames_clean"
GRAPH_DIR = "graphs"
GRAPH_JSON_DIR = "graph_json"
MANIFEST = "episode.json"


def _json_default(value: Any) -> Any:
    """Coerce whatever a fact carried in from the simulator.

    Relation values, node boxes and graph metadata come from numpy and PhysX,
    and a single ``float32`` raw value in one edge would otherwise abort the
    whole export. Anything still unknown is stringified rather than raised: a
    figure's graph dump losing fidelity on one exotic attribute is a far better
    outcome than losing the frame it belongs to.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    return str(value)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )


def save_png(path: Path, image: np.ndarray) -> None:
    """Write an ``[H, W, 3]`` uint8 array losslessly."""
    from PIL import Image

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(arr[..., :3])).save(str(path))


class EpisodeWriter:
    """Accumulates one episode's exports, then commits or drops them whole."""

    def __init__(
        self,
        root: Path | str,
        name: str,
        *,
        save_clean: bool = False,
        graph_image: bool = True,
        graph_json: bool = True,
        paper_graph_style: bool = False,
    ):
        self.root = Path(root)
        self.name = str(name)
        self.save_clean = bool(save_clean)
        self.graph_image = bool(graph_image)
        self.graph_json = bool(graph_json)
        self.paper_graph_style = bool(paper_graph_style)
        self.staging = self.root / f".staging_{self.name}"
        self.records: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def count(self) -> int:
        return self._index

    def _subdirs(self) -> List[str]:
        out = [FRAME_DIR]
        if self.save_clean:
            out.append(CLEAN_DIR)
        if self.graph_image:
            out.append(GRAPH_DIR)
        if self.graph_json:
            out.append(GRAPH_JSON_DIR)
        return out

    def open(self) -> None:
        """Start (or restart) the staging directory for this episode."""
        shutil.rmtree(self.staging, ignore_errors=True)
        for sub in self._subdirs():
            (self.staging / sub).mkdir(parents=True, exist_ok=True)
        self.records = []
        self._index = 0

    def write_step(
        self,
        *,
        step: int,
        frame: np.ndarray,
        graph: Graph,
        colormap: Optional[ColorMap] = None,
        clean: Optional[np.ndarray] = None,
        callouts: Optional[Sequence[Any]] = None,
    ) -> int:
        """Export one frame and its graph. Returns the shared export index."""
        stem = f"{self._index:04d}"
        record: Dict[str, Any] = {
            "index": self._index,
            "step": int(step),
            "frame": f"{FRAME_DIR}/frame_{stem}.png",
            "nodes": graph.node_ids(),
            "n_edges": len(graph.edges),
        }
        save_png(self.staging / FRAME_DIR / f"frame_{stem}.png", frame)
        if self.save_clean and clean is not None:
            save_png(self.staging / CLEAN_DIR / f"frame_{stem}.png", clean)
            record["frame_clean"] = f"{CLEAN_DIR}/frame_{stem}.png"
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
        if callouts:
            record["callouts"] = [c.to_dict() for c in callouts]
        self.records.append(record)
        self._index += 1
        return self._index - 1

    def commit(self, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Publish the staged episode under its real name and return the path."""
        payload = dict(metadata or {})
        payload["name"] = self.name
        payload["exported_frames"] = len(self.records)
        payload["steps"] = self.records
        save_json(self.staging / MANIFEST, payload)
        final = self.root / self.name
        shutil.rmtree(final, ignore_errors=True)
        self.staging.rename(final)
        return final

    def discard(self) -> None:
        """Delete everything this episode staged. Safe to call twice."""
        shutil.rmtree(self.staging, ignore_errors=True)
        self.records = []
        self._index = 0
