"""Configuration, paths, identity and small I/O helpers shared by every stage.

Nothing here knows about the task, the model or the simulator, and nothing
imports torch: preprocessing runs in its own environment and has to be able to
read the same configs and write the same identity records as training.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
CONFIG_DIR = os.path.join(PACKAGE_DIR, "configs")
PROMPT_DIR = os.path.join(PACKAGE_DIR, "prompts")

if REPO_ROOT not in sys.path:
    # Entry points run as ``python -m real_robot...`` from the repository root,
    # which already puts it on the path. Importing the package from anywhere
    # else still has to find ``graph``, ``rssm`` and ``scenegraph``.
    sys.path.insert(0, REPO_ROOT)

# Named episode selections. Every episode trains; ``diagnostic`` is a fixed
# subset of those same episodes and ``pilot`` the first milestone's three.
SELECTIONS = ("training", "diagnostic", "pilot")
# Names from the earlier held-out design, refused with an explanation.
REMOVED_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def repo_path(path: str) -> str:
    """Absolute path for a repository-relative one; absolute paths pass."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def episode_name(index: int) -> str:
    return f"episode_{int(index):06d}"


def makedirs(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def config_path(name_or_path: str) -> str:
    """``reward`` -> ``real_robot/configs/reward.yaml``; paths pass through."""
    if name_or_path.endswith((".yaml", ".yml")) or os.sep in name_or_path or "/" in name_or_path:
        return repo_path(name_or_path)
    return os.path.join(CONFIG_DIR, f"{name_or_path}.yaml")


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set ``a.b.c`` in place. Unknown keys are refused, not created.

    A typo in an override that silently created a new key would run the
    experiment with the default the person meant to change.
    """
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"override {dotted!r}: no section {part!r}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"override {dotted!r}: no such setting")
    node[parts[-1]] = value


def get_by_path(cfg: Mapping, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def load_config(name_or_path: str, overrides: Sequence[str] = ()) -> dict:
    """One config with ``key.path=value`` overrides applied (YAML-typed)."""
    cfg = load_yaml(config_path(name_or_path))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_configs(names: Iterable[str], overrides: Sequence[str] = ()) -> Dict[str, dict]:
    """Several named configs; each override names its config first.

    ``--set reward.settle.frames=10`` changes ``reward.yaml``. An override
    whose first segment names no loaded config is an error.
    """
    names = list(names)
    per_config: Dict[str, List[str]] = {name: [] for name in names}
    for item in overrides:
        head = item.split("=", 1)[0].split(".", 1)[0].strip()
        if head not in per_config:
            raise KeyError(
                f"override {item!r} names config {head!r}; loaded configs are {names}"
            )
        per_config[head].append(item.split(".", 1)[1])
    return {name: load_config(name, per_config[name]) for name in names}


def add_config_arguments(parser) -> None:
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="CONFIG.KEY=VALUE",
        help="override one setting, e.g. --set reward.settle.frames=10",
    )


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def _json_default(value: Any):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data: Any, indent: Optional[int] = 2) -> str:
    """Write through a temporary file in the same directory, then rename.

    An interrupted write must not leave a truncated file where a complete one
    was: every later stage reads these as the source of truth.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=indent, default=_json_default, ensure_ascii=False)
            out.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_jsonl(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default,
                      ensure_ascii=False)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Order-independent digest of a JSON-compatible value."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class IdentityError(RuntimeError):
    """Two artifacts that were built under different contracts."""


def identity_mismatches(expected: Mapping[str, Any], stored: Mapping[str, Any],
                        fields: Optional[Iterable[str]] = None) -> List[str]:
    keys = list(fields) if fields is not None else sorted(set(expected) | set(stored))
    problems = []
    for key in keys:
        want, got = expected.get(key), stored.get(key)
        if canonical_json(want) != canonical_json(got):
            problems.append(f"{key}: stored {got!r} vs expected {want!r}")
    return problems


def require_identity(expected: Mapping[str, Any], stored: Mapping[str, Any],
                     what: str, fields: Optional[Iterable[str]] = None) -> None:
    """Refuse, naming every field that disagrees. Never a warning."""
    problems = identity_mismatches(expected, stored, fields)
    if problems:
        raise IdentityError(
            f"{what} was built under a different contract:\n  " + "\n  ".join(problems)
        )


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Episode selection
# --------------------------------------------------------------------------- #
def parse_episodes(text: str, selections: Mapping[str, Sequence[int]],
                   available: Sequence[int]) -> List[int]:
    """Resolve an episode selection.

    ``all`` | ``training`` | ``diagnostic`` | ``pilot`` | ``diagnostic:3``
    (the first three, in selection order) | ``3,17,42`` | ``0-9``. Explicit
    indices must exist. There are no train/val/test splits: every episode
    trains, and those names are refused rather than guessed at.
    """
    text = (text or "all").strip()
    available_set = set(int(i) for i in available)
    if text == "all":
        return sorted(available_set)
    head, _, count = text.partition(":")
    if head in REMOVED_SPLITS:
        raise KeyError(
            f"{head!r}: there are no train/val/test splits -- every episode trains. "
            "Use all, training, diagnostic, pilot or explicit episode ids."
        )
    if head in SELECTIONS:
        if head not in selections:
            raise KeyError(f"no {head!r} selection exists yet; run `python -m real_robot.data.selection create`")
        chosen = [int(i) for i in selections[head]]
        if count:
            chosen = chosen[: int(count)]
        return chosen
    chosen = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            chosen.extend(range(int(lo), int(hi) + 1))
        else:
            chosen.append(int(part))
    missing = [i for i in chosen if i not in available_set]
    if missing:
        raise KeyError(f"episodes {missing} are not in the dataset")
    return chosen


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class RunLogger:
    """JSONL always; TensorBoard when it is installed; a short console line.

    The repository's ``tools.Logger`` filters its console output down to the
    online run's keys, which would hide every offline metric, so this keeps
    its own list of what to print.
    """

    def __init__(self, logdir: str, console_keys: Sequence[str] = (), tensorboard: bool = True):
        self.logdir = makedirs(logdir)
        self.console_keys = tuple(console_keys)
        self._path = os.path.join(self.logdir, "metrics.jsonl")
        self._writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=self.logdir, max_queue=1000)
            except Exception:
                self._writer = None

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        values = {key: float(value) for key, value in scalars.items()}
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), **values}) + "\n")
        if self._writer is not None:
            for key, value in values.items():
                self._writer.add_scalar(key, value, int(step))
            self._writer.flush()
        shown = [(k, values[k]) for k in self.console_keys if k in values]
        if shown:
            print(f"[{step}] " + " / ".join(f"{k} {v:.4g}" for k, v in shown), flush=True)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
