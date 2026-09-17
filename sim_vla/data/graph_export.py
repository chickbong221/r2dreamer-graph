"""Readable graphs for a sample of episodes, one JSONL file per episode.

The packed arrays are what training reads; nothing in them says which node was
the peg. This writes the ``Graph`` objects those arrays were packed from, so a
frame can be inspected without a simulator -- and one file per episode rather
than one per frame, because a 500-episode task at 130 frames each is 65,000
files that no filesystem enjoys and no reader wants to open.

A sample by default, not everything: the JSONL is several times the size of the
packed arrays it describes, and its purpose is to check a handful of episodes
rather than to be read in bulk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

# One definition of "how a simulator value becomes JSON" for the repo: the
# figure exporter already had to answer this for the same Graph objects.
from scenegraph.figures.writer import _json_default


class GraphJsonlExporter:
    """Writes one ``<dir>/ep_<id>.jsonl`` per sampled episode."""

    def __init__(self, out_dir: str | Path, sample: int = 3):
        self.out_dir = Path(out_dir)
        self.sample = int(sample)
        self.written: list[Path] = []

    def wants(self, episode_index: int) -> bool:
        """Whether this episode's graphs are worth keeping readable.

        Indexed by accepted episode, so the sample is the first few episodes
        written rather than the first few attempted -- a rejected attempt
        should not use up a slot in a sample meant for inspection.
        """
        return self.sample > 0 and int(episode_index) < self.sample

    def write(self, episode_id: int, graphs: Sequence[Any]) -> Path | None:
        if not graphs:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"ep_{int(episode_id):05d}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for frame, graph in enumerate(graphs):
                payload = graph.to_dict()
                # The row index is what ties a line back to row ``frame`` of
                # the packed arrays. ``graph.frame`` is the builder's own
                # counter and the two are not required to agree.
                handle.write(json.dumps(
                    {"row": frame} | payload, default=_json_default) + "\n")
        self.written.append(path)
        return path
