"""Importing two vendored trees that both claim generic top-level names.

``sim_vla/tdmpc2`` and ``sim_vla/sold/sold`` are upstream checkouts, not
packages. Neither has an ``__init__.py``; both are meant to be run with their
own directory as the working directory, and both import their own modules by
bare top-level name::

    tdmpc2   from common import math          common, envs, trainer, tdmpc2
    sold     from modeling.sold import ...    modeling, datasets, utils, envs

Three collisions follow, and the first two are silent:

* ``envs`` is claimed by *both* backends.
* ``envs``, ``trainer``, ``train``, ``tools``, ``buffer`` and ``tests`` are also
  top-level modules of this repository, which is on ``sys.path`` whenever
  anything is run as ``python -m sim_vla...``. ``sim_vla.models.world_model``
  imports the repository's ``networks`` and ``rssm``; if a backend's ``envs``
  were left installed under the bare name, the next repository import would get
  the wrong one.
* Once a name is in ``sys.modules`` it stays. Prepending a directory to
  ``sys.path`` and importing is a one-way door: the *second* backend imported
  in a process would silently reuse the first one's ``envs``.

So this module does not "add a path". It owns the bare names for the duration
of a ``with`` block and hands them back afterwards:

    activate    stash whatever currently occupies this backend's top-level
                names, install the modules this backend loaded last time, and
                prepend its root to ``sys.path``
    deactivate  take the backend's modules back out of ``sys.modules`` into a
                private cache, and restore what was stashed

Modules are *cached*, not re-imported: a class created inside the block keeps
working outside it (its ``__module__`` is only a string), and re-entering the
block reinstalls exactly the module objects its classes were defined in, so
``isinstance`` keeps holding. Re-importing instead would produce a second copy
of every class and break that.

Nothing outside the block is disturbed: a name this backend does not own is
never touched, and ``sys.path`` is restored to the list it was.
"""

from __future__ import annotations

import importlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

SIM_VLA = Path(__file__).resolve().parents[1]
REPO = SIM_VLA.parent

TDMPC2_ROOT = SIM_VLA / "tdmpc2"
SOLD_ROOT = SIM_VLA / "sold" / "sold"
SOLD_CONFIGS = SIM_VLA / "sold" / "configs"

# One lock for every backend. The vendored trees are activated by mutating
# process-global ``sys.modules``; two threads doing that at once would each see
# the other's names.
_LOCK = threading.RLock()


def _root_of(dotted: str) -> str:
    return dotted.split(".", 1)[0]


class Vendored:
    """One vendored source tree, importable without leaking its names."""

    def __init__(self, name: str, root: Path):
        self.name = str(name)
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.name} was not checked out at {self.root}")
        self.owned: Set[str] = self._top_level_names()
        self._cache: Dict[str, ModuleType] = {}
        self._stashed: Dict[str, ModuleType] = {}
        self._depth = 0
        self._path_entry = str(self.root)

    # ------------------------------------------------------------------ names
    def _top_level_names(self) -> Set[str]:
        """Every bare name an import inside this tree could resolve to.

        Read from the directory rather than hard-coded: a name that appears
        upstream later would otherwise be left installed after the block, which
        is exactly the failure this class exists to prevent.
        """
        names: Set[str] = set()
        for entry in self.root.iterdir():
            if entry.name.startswith((".", "__")):
                continue
            if entry.is_dir() and any(entry.glob("*.py")):
                names.add(entry.name)
            elif entry.suffix == ".py":
                names.add(entry.stem)
        return names

    def owns(self, dotted: str) -> bool:
        return _root_of(dotted) in self.owned

    # --------------------------------------------------------------- lifetime
    def _install(self) -> None:
        self._stashed = {name: module for name, module in sys.modules.items()
                         if self.owns(name)}
        for name in self._stashed:
            del sys.modules[name]
        sys.modules.update(self._cache)
        sys.path.insert(0, self._path_entry)

    def _uninstall(self) -> None:
        # Remove exactly the entry that was inserted, by identity of position:
        # ``list.remove`` would take an equal string put there by someone else.
        for index in range(len(sys.path)):
            if sys.path[index] is self._path_entry:
                del sys.path[index]
                break
        else:                                              # pragma: no cover
            if self._path_entry in sys.path:
                sys.path.remove(self._path_entry)
        self._cache = {name: module for name, module in sys.modules.items()
                       if self.owns(name)}
        for name in list(self._cache):
            del sys.modules[name]
        sys.modules.update(self._stashed)
        self._stashed = {}

    @contextmanager
    def active(self) -> Iterator["Vendored"]:
        """Own this backend's top-level names for the duration of the block."""
        with _LOCK:
            if self._depth == 0:
                self._install()
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._uninstall()

    # ---------------------------------------------------------------- imports
    def module(self, dotted: str) -> ModuleType:
        """Import one module from this tree, leaving no bare name behind."""
        if not self.owns(dotted):
            raise ValueError(
                f"{dotted!r} is not part of {self.name}; it owns "
                f"{sorted(self.owned)}")
        with self.active():
            return importlib.import_module(dotted)

    def load(self, *dotted: str) -> List[ModuleType]:
        with self.active():
            return [importlib.import_module(name) for name in dotted]

    def get(self, dotted: str, attribute: str) -> Any:
        """One attribute of one module. The usual way to reach a class."""
        return getattr(self.module(dotted), attribute)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "root": str(self.root),
                "top_level": sorted(self.owned),
                "loaded": sorted(self._cache)}


TDMPC2 = Vendored("tdmpc2", TDMPC2_ROOT)
SOLD = Vendored("sold", SOLD_ROOT)

BACKENDS = {"tdmpc2": TDMPC2, "sold": SOLD}


def backend(name: str) -> Vendored:
    try:
        return BACKENDS[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; known: {sorted(BACKENDS)}") from None
