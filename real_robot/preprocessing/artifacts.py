"""Digests of the files an artifact was made from, for deciding whether it is current."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from ..common import file_sha256

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
    """Hash the actual source file, so local edits or missing files invalidate prepared copies."""
    path = source.video_path(episode, camera)
    return file_digest(path)
