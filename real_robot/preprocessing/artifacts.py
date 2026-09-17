"""Input identities of preprocessing artifacts.

An artifact is reused only when everything it was made from is unchanged.
Each stage records, beside its output, the digests of the files it read and the
settings it used; before reusing an output it compares that record with the
current inputs, and a mismatch means the output is made again. Consumers check
the same records, so a corrected annotation invalidates its tracks, its
geometry, the reward scales fitted on it and the packed episode built from it,
and nothing downstream can silently keep the old version.

    annotation  <- graph, frozen bins, prompts, Gemini settings, prepared videos, validation rules
    tracks      <- annotation file, tracking settings, source videos
    depth       <- source video, Depth Pro weights, fixed focal length, cache settings
    alignment   <- the episodes' tracks and depth, alignment settings, camera check
    geometry    <- annotation file, tracks, depth, alignment, geometry settings, camera check
    scales      <- every training episode's annotation and geometry
    episode     <- annotation, tracks, geometry, scales, action specification, selection
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..common import canonical_json, file_sha256, identity_mismatches, read_json

_DIGESTS: Dict[Tuple[str, int, int], str] = {}


def file_digest(path: str) -> Optional[str]:
    """SHA-256 of a file, cached for the process by path, size and modification time; None if absent."""
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    key = (os.path.abspath(path), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _DIGESTS:
        _DIGESTS[key] = file_sha256(path)
    return _DIGESTS[key]


def source_video_digest(source, episode: int, camera: str) -> Optional[str]:
    """The pinned snapshot's recorded checksum when download recorded one, else the file's digest."""
    path = source.video_path(episode, camera)
    try:
        record = source.source_record()
        rel = os.path.relpath(path, source.source_root).replace(os.sep, "/")
        entry = record.get("files", {}).get(rel)
        if entry and entry.get("sha256"):
            return str(entry["sha256"])
    except (FileNotFoundError, ValueError):
        pass
    return file_digest(path)


def sidecar(path: str) -> str:
    """The JSON record written beside an ``.npz`` artifact."""
    return path[:-4] + ".json" if path.endswith(".npz") else path + ".json"


def stored_inputs(path: str) -> Optional[Dict[str, Any]]:
    record = sidecar(path)
    if not os.path.isfile(record):
        return None
    return read_json(record).get("inputs")


def stale_reason(expected: Mapping[str, Any], stored: Optional[Mapping[str, Any]]) -> Optional[str]:
    """None when ``stored`` matches ``expected``; otherwise which inputs changed."""
    if stored is None:
        return "no record of its inputs"
    problems = identity_mismatches(expected, stored)
    if not problems:
        return None
    names = [p.split(":", 1)[0] for p in problems]
    return "inputs changed: " + ", ".join(names)


def reusable(path: str, expected: Mapping[str, Any]) -> Tuple[bool, str]:
    """Whether the artifact at ``path`` exists and was made from ``expected``."""
    if not os.path.isfile(path):
        return False, "missing"
    reason = stale_reason(expected, stored_inputs(path))
    return reason is None, reason or "current"


def same(a: Any, b: Any) -> bool:
    return canonical_json(a) == canonical_json(b)


def require_current(what: str, expected: Mapping[str, Any], stored: Optional[Mapping[str, Any]]) -> None:
    reason = stale_reason(expected, stored)
    if reason is not None:
        raise SystemExit(f"{what} is out of date ({reason}); run its stage again before building on it")


def mismatched_fields(expected: Mapping[str, Any], stored: Mapping[str, Any]) -> List[str]:
    return [p.split(":", 1)[0] for p in identity_mismatches(expected, stored)]
