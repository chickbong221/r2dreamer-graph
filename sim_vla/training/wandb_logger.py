"""One Weights & Biases run per pipeline invocation, shared by all three stages.

The three stages are one experiment, not three: Stage 1B trains against the very
world model Stage 1A produced, and Stage 2 continues with both. So they log into
a single run rather than three, and the stage is a key prefix rather than a
separate run name.

**Each stage keeps its own x-axis.** Stage 1A counts gradient steps, Stage 1B
counts its own gradient steps, and Stage 2 counts environment steps -- three
different units. Forcing them onto wandb's implicit global step would draw a
world-model loss against a number that means gradient steps for the first third
of the chart and environment steps for the last. ``define_metric`` binds each
prefix to its own step instead, which is what makes the curves readable.

**Logging never decides whether training runs.** A missing wandb package, a
failed login or an unreachable host is reported and then ignored: a 48-hour
training job should not die because a metrics sink is down. What is *not*
silent is the arm and the task, which go into the run's config and tags, since
a run that cannot say which arm it was is not worth comparing.

For a compute node without outbound network, set ``wandb.mode: offline`` and
run ``wandb sync`` on the run directory afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

# Every metric a stage logs is prefixed with the stage name, and each prefix
# plots against the step metric named here. "online" uses environment steps
# because that is the budget the arms are compared at -- updates are a
# consequence of updates_per_collect, not of how much experience was gathered.
STEP_METRICS = {
    "world": "step",
    "imitation": "step",
    "online": "env_steps",
}


@dataclass
class WandbSettings:
    """The ``wandb`` block of the merged config, with the defaults resolved."""

    enabled: bool = False
    project: str = "sim_vla"
    entity: str = ""
    group: str = ""
    name: str = ""
    mode: str = "online"
    tags: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "WandbSettings":
        """Read the block, filling group/name/tags from the arm being run.

        The defaults are chosen for the comparison this repository exists to
        make: the group is the task, so a task's arms land side by side, and
        the arm is in both the name and the tags so a chart legend says which
        is which without opening the config.
        """
        block = dict((cfg.get("wandb") or {}))
        experiment = dict((cfg.get("experiment") or {}))
        task = str(experiment.get("task") or "")
        arm = str(experiment.get("arm") or "")

        tags = list(block.get("tags") or [])
        for value in (task, arm):
            if value and value not in tags:
                tags.append(value)

        return cls(
            enabled=bool(block.get("enabled", False)),
            project=str(block.get("project") or "sim_vla"),
            entity=str(block.get("entity") or ""),
            group=str(block.get("group") or task),
            name=str(block.get("name") or (f"{task}-{arm}" if task and arm
                                           else "")),
            mode=str(block.get("mode") or "online"),
            tags=tuple(tags),
        )


class RunLogger:
    """A wandb run, or an object with the same surface that does nothing.

    Callers never branch on whether logging is on; :meth:`log` is safe to call
    either way.
    """

    def __init__(self, run: Any = None,
                 settings: Optional[WandbSettings] = None):
        self._run = run
        self.settings = settings or WandbSettings()

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: Dict[str, float], *, stage: str) -> None:
        """Log one stage's metrics, prefixed and plotted against its own step.

        ``metrics`` carries the stage's own step under the name
        :data:`STEP_METRICS` gives for it -- ``step`` for the two offline
        stages, ``env_steps`` for the online one -- and that key is what the
        prefix is bound to.
        """
        if self._run is None:
            return
        payload = {f"{stage}/{key}": float(value)
                   for key, value in metrics.items()
                   if isinstance(value, (int, float, bool))}
        if not payload:
            return
        try:
            self._run.log(payload)
        except Exception as exc:                           # noqa: BLE001
            # One failed log must not take the training loop with it, and must
            # not print once per step either.
            self._warn_once(f"wandb.log failed: {type(exc).__name__}: {exc}")

    def summary(self, values: Dict[str, Any]) -> None:
        """Values that describe the run as a whole rather than a step."""
        if self._run is None:
            return
        try:
            for key, value in values.items():
                self._run.summary[key] = value
        except Exception as exc:                           # noqa: BLE001
            self._warn_once(f"wandb summary failed: {type(exc).__name__}: {exc}")

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=int(exit_code))
        except Exception as exc:                           # noqa: BLE001
            print(f"[wandb] finish failed: {type(exc).__name__}: {exc}",
                  flush=True)
        self._run = None

    def _warn_once(self, message: str) -> None:
        if not getattr(self, "_warned", False):
            print(f"[wandb] {message} (further logging errors are silent)",
                  flush=True)
            self._warned = True


def start_run(cfg: Dict[str, Any], *,
              extra_config: Optional[Dict[str, Any]] = None) -> RunLogger:
    """Open the run for this pipeline invocation, or return an inert logger.

    Returns a :class:`RunLogger` in every case, including when wandb is off,
    not installed, or unreachable, so no caller has to branch.
    """
    settings = WandbSettings.from_config(cfg)
    if not settings.enabled:
        print("[wandb] disabled (set wandb.enabled: true to log)", flush=True)
        return RunLogger(None, settings)

    try:
        import wandb
    except ImportError:
        print("[wandb] enabled in the config but the package is not "
              "installed; training continues unlogged. `pip install wandb`.",
              flush=True)
        return RunLogger(None, settings)

    init_kwargs: Dict[str, Any] = {
        "project": settings.project,
        "group": settings.group or None,
        "name": settings.name or None,
        "mode": settings.mode,
        "tags": list(settings.tags),
        "config": dict(cfg) | dict(extra_config or {}),
    }
    if settings.entity:
        init_kwargs["entity"] = settings.entity

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:                               # noqa: BLE001
        print(f"[wandb] init failed ({type(exc).__name__}: {exc}); training "
              "continues unlogged. For a node without outbound network set "
              "wandb.mode: offline and `wandb sync` the run directory later.",
              flush=True)
        return RunLogger(None, settings)

    # Bind each stage's prefix to its own step metric before anything is
    # logged: a metric that has already been logged against the implicit step
    # cannot be rebound afterwards.
    try:
        for stage, step_key in STEP_METRICS.items():
            run.define_metric(f"{stage}/{step_key}")
            run.define_metric(f"{stage}/*", step_metric=f"{stage}/{step_key}")
    except Exception as exc:                               # noqa: BLE001
        print(f"[wandb] define_metric failed ({type(exc).__name__}: {exc}); "
              "charts will use the global step instead", flush=True)

    where = getattr(run, "url", "") or f"mode={settings.mode}"
    print(f"[wandb] {settings.project} / {settings.name or run.id} -> {where}",
          flush=True)
    return RunLogger(run, settings)
