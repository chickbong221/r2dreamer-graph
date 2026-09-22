"""Per-phase wall time and CUDA peak memory, for the parts that actually cost.

An actor update is not one thing. Under ``flow_reinforce`` it is collection
(no-grad rollout), target computation, a critic step, thousands of small
scored-transition backwards, and optionally an anchor -- and which of those
dominates is not guessable from the code. A single "update took 3.1s" number
cannot tell you whether raising ``actor_transition_microbatch`` would help.

Two things this is careful about.

**Timing GPU work needs a synchronize.** CUDA kernels are queued
asynchronously, so a ``perf_counter`` around a launch measures the launch, not
the work. Every phase boundary synchronizes when profiling is on -- which is
itself a cost, and the reason this is off by default rather than always on.

**Peak memory is per phase, not cumulative.** ``reset_peak_memory_stats`` runs
at each phase start, so the number reported for "scoring" is scoring's own
high-water mark rather than the run's. The allocator does not return freed
blocks to the driver, so *reserved* is reported beside *allocated*: a phase
that allocates little inside a large reservation is not cheap to run next to
something else.
"""

from __future__ import annotations

import contextlib
import time
from typing import Dict, List, Optional

import torch


class Phases:
    """Named phases of one update, measured only when asked.

    Disabled, every method is a no-op and the context manager costs an
    attribute lookup: this sits in the training loop, so it must not be
    something to remember to remove.
    """

    def __init__(self, device=None, enabled: bool = False):
        self.enabled = bool(enabled)
        self.device = torch.device(device) if device is not None else None
        self.cuda = bool(self.enabled and self.device is not None
                         and self.device.type == "cuda")
        self.seconds: Dict[str, float] = {}
        self.allocated: Dict[str, float] = {}
        self.reserved: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def _sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - started
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            if self.cuda:
                mega = 1024.0 ** 2
                self.allocated[name] = max(
                    self.allocated.get(name, 0.0),
                    torch.cuda.max_memory_allocated(self.device) / mega)
                self.reserved[name] = max(
                    self.reserved.get(name, 0.0),
                    torch.cuda.max_memory_reserved(self.device) / mega)

    def merge(self, other: "Phases") -> None:
        for name, value in other.seconds.items():
            self.seconds[name] = self.seconds.get(name, 0.0) + value
            self.counts[name] = self.counts.get(name, 0) + other.counts[name]
        for name, value in other.allocated.items():
            self.allocated[name] = max(self.allocated.get(name, 0.0), value)
        for name, value in other.reserved.items():
            self.reserved[name] = max(self.reserved.get(name, 0.0), value)

    def metrics(self, prefix: str = "profile") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, value in self.seconds.items():
            out[f"{prefix}_{name}_s"] = float(value)
        for name, value in self.allocated.items():
            out[f"{prefix}_{name}_alloc_mb"] = float(value)
        for name, value in self.reserved.items():
            out[f"{prefix}_{name}_reserved_mb"] = float(value)
        return out

    def table(self, title: str = "") -> str:
        """A human-readable summary, with each phase's share of the total."""
        if not self.seconds:
            return f"[profile] {title}: nothing measured"
        total = sum(self.seconds.values())
        lines: List[str] = [
            f"[profile] {title}" if title else "[profile]",
            f"[profile] {'phase':<22}{'calls':>7}{'seconds':>11}"
            f"{'share':>8}{'alloc MB':>11}{'reserved MB':>13}",
        ]
        for name in sorted(self.seconds, key=self.seconds.get, reverse=True):
            share = 100.0 * self.seconds[name] / total if total else 0.0
            lines.append(
                f"[profile] {name:<22}{self.counts.get(name, 0):>7}"
                f"{self.seconds[name]:>11.3f}{share:>7.1f}%"
                f"{self.allocated.get(name, float('nan')):>11.1f}"
                f"{self.reserved.get(name, float('nan')):>13.1f}")
        lines.append(f"[profile] {'total':<22}{'':>7}{total:>11.3f}")
        if not self.cuda:
            lines.append("[profile] CUDA not in use: memory columns are NaN "
                         "and timings do not reflect GPU behaviour")
        return "\n".join(lines)
