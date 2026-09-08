"""One command: collect, build the probe set, train, report.

    python -m graph_encoder_probe.run --config graph_encoder_probe/config.yaml

Collection is the expensive stage and runs once. A cache that matches the
configured collection is reused; a probe set built from that exact cache is
reused too, because the pairs have to be identical across every training run
that is going to be compared. Each run gets its own directory and its own
resolved configuration.

``--smoke`` is the pre-flight from the plan's last step: a short collection, a
handful of updates, and the six plumbing checks. Run it before spending the full
collection budget.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

from . import EDIT_GROUPS
from .dataset import (
    FIELD_DTYPES,
    GraphDataset,
    GraphFrames,
    git_revision,
    load_packed_sample,
    write_frames,
)
from .evaluate import PROBE_ROWS, final_table, group_table, probe
from .model import build_model, load_checkpoint
from .pairs import CONTROL_GROUP, PairSet, build_pairs
from .train import FINAL_CHECKPOINT, TrainResult, build_pool, resolve_device, train

from scenegraph.adapters.graph_pack import GRAPH_KEYS

REPORT_NAME = "report.md"
RESOLVED_CONFIG = "resolved_config.yaml"

# Volume only. Widths, seeds and the loss stay at their configured values, so a
# smoke run exercises the same code the real one does.
SMOKE_OVERRIDES: dict[str, Any] = {
    "collect.episodes": 2,
    "collect.max_frames": 400,
    "collect.shard_size": 128,
    "collect.out_dir": "graph_encoder_probe/outputs/dataset_smoke",
    "pairs.per_group": 3,
    "pairs.controls": 2,
    "pairs.out_dir": "graph_encoder_probe/outputs/pairs_smoke",
    "train.batch_size": 16,
    "train.max_updates": 12,
    "train.probe_every": 4,
    "train.min_updates": 0,
    "train.monitor_frames": 64,
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: str, overrides: Sequence[str] = ()) -> dict:
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node: Any = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if part not in node:
                raise SystemExit(f"--set {key}: no section {part!r} in the config")
            node = node[part]
        if parts[-1] not in node:
            raise SystemExit(f"--set {key}: no such setting")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def apply_overrides(cfg: dict, overrides: Mapping[str, Any]) -> dict:
    for key, value in overrides.items():
        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def dataset_dir(cfg: Mapping) -> str:
    return os.path.join(str(cfg["collect"]["out_dir"]), str(cfg["collect"]["env_id"]))


def pairs_dir(cfg: Mapping) -> str:
    return os.path.join(str(cfg["pairs"]["out_dir"]), str(cfg["collect"]["env_id"]))


def run_dir(cfg: Mapping) -> str:
    name = str(cfg["output"].get("run_name") or "")
    if not name:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(str(cfg["output"]["runs_dir"]), name)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_collect(cfg: Mapping, *, force: bool = False) -> GraphDataset:
    """Reuse a compatible cache, otherwise start the simulator."""
    path = dataset_dir(cfg)
    if not force and os.path.isfile(os.path.join(path, "meta.json")):
        dataset = GraphDataset.load(path)
        ok, why = dataset.compatible_with(cfg["collect"])
        if ok:
            print(f"[collect] reusing {len(dataset)} cached graphs at {path}")
            return dataset
        print(f"[collect] cache at {path} does not match this config ({why}); recollecting")

    from .collect import collect  # imported here: it needs the simulator, the rest does not

    collect(cfg["collect"], path)
    return GraphDataset.load(path)


def stage_pairs(cfg: Mapping, dataset: GraphDataset, *, force: bool = False) -> PairSet:
    """Build the probe set once; every later run measures the same pairs."""
    path = pairs_dir(cfg)
    if not force and os.path.isfile(os.path.join(path, "pairs.json")):
        pairset = PairSet.load(path)
        want = dataset.frames.fingerprint()
        got = str(pairset.meta.get("dataset_fingerprint") or "")
        if got == want:
            pairset.verify_against(dataset.frames)
            print(f"[pairs] reusing {len(pairset)} pairs at {path}")
            return pairset
        print(f"[pairs] the set at {path} was built from a different cache; rebuilding")

    pair_cfg = cfg["pairs"]
    pairset = build_pairs(
        dataset.frames,
        per_group=int(pair_cfg["per_group"]),
        controls=int(pair_cfg["controls"]),
        seed=int(pair_cfg.get("seed", cfg.get("seed", 0))),
        geometry_delta_m=tuple(pair_cfg["geometry_delta_m"]),
        groups=EDIT_GROUPS,
    )
    pairset.save(path)
    counts = {name: len(rows) for name, rows in pairset.groups().items()}
    print(f"[pairs] built {len(pairset)} pairs at {path}: {counts}")
    return pairset


# --------------------------------------------------------------------------- #
# Plumbing checks
# --------------------------------------------------------------------------- #
def check_cache_roundtrip(dataset: GraphDataset, source_dir: Optional[str] = None) -> str:
    """Written and re-read graphs are the same packed tensors, bit for bit.

    Two halves. If the collection kept a sample of frames exactly as the packer
    emitted them, the loaded cache is compared against *those* -- otherwise the
    check could only ever compare the cache with another copy of itself. Then
    the cache is re-serialised and reloaded, comparing both the numpy tables and
    the tensors the encoder is handed, so a dtype that survived the arrays but
    not the batch conversion still fails here.
    """
    against_source = "no packer sample was kept"
    sample = load_packed_sample(source_dir) if source_dir else None
    if sample is not None:
        rows, fields = sample
        for key in GRAPH_KEYS:
            stored = dataset.frames.fields[key][rows]
            if not np.array_equal(stored, fields[key]):
                return f"FAIL: {key} in the cache differs from what the packer emitted"
        against_source = f"{rows.size} frames match the packer's own output"

    tmp = tempfile.mkdtemp(prefix="probe_roundtrip_")
    try:
        count = min(len(dataset), 64)
        index = [
            {key: dataset.index[key][i] for key in dataset.index} for i in range(count)
        ]
        write_frames(
            tmp,
            (dataset.frames.frame(i) for i in range(count)),
            index,
            {"roundtrip": True, "vocab_sizes": dataset.meta.get("vocab_sizes", {})},
        )
        again = GraphDataset.load(tmp)
        for key in GRAPH_KEYS:
            before = dataset.frames.fields[key][:count]
            after = again.frames.fields[key]
            if after.dtype != FIELD_DTYPES[key]:
                return f"FAIL: {key} reloaded as {after.dtype}, expected {FIELD_DTYPES[key]}"
            if not np.array_equal(before, after):
                return f"FAIL: {key} differs after a save/load round trip"
        left = dataset.frames.torch_batch(np.arange(count))
        right = again.frames.torch_batch(np.arange(count))
        for key in GRAPH_KEYS:
            if left[key].dtype != right[key].dtype or not bool((left[key] == right[key]).all()):
                return f"FAIL: {key} tensor differs after a round trip"
        return f"ok ({against_source}; {count} frames round-tripped, arrays and tensors identical)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_pairs(dataset: GraphDataset, pairset: PairSet) -> str:
    """Every edit touches exactly the cells its spec names, and nothing else."""
    fingerprint = str(pairset.meta.get("dataset_fingerprint") or "")
    if fingerprint and fingerprint != dataset.frames.fingerprint():
        return "FAIL: the probe set was built from a different cache"
    pairset.verify_against(dataset.frames)
    edited = sum(1 for spec in pairset.specs if spec.edited_field is not None)
    return f"ok ({edited} edits and {len(pairset) - edited} controls verified against the packed tensors)"


def check_gradients(model, pool: GraphFrames, device, *, batch: int = 8) -> str:
    """Reconstruction reaches both modules.

    Not every parameter must move -- an embedding row for a label this batch
    never used correctly gets a zero gradient -- so the check is that each
    module receives gradients at all and that some of them are non-zero.
    """
    import torch

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(pool.torch_batch(np.arange(min(batch, len(pool))), device))
    out.total.backward()
    report = []
    for name, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        params = [p for p in module.parameters() if p.requires_grad]
        missing = [n for n, p in module.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            return f"FAIL: {name} parameters received no gradient: {missing[:4]}"
        moved = sum(1 for p in params if bool(torch.any(p.grad != 0)))
        if moved == 0:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            return f"FAIL: every {name} gradient was exactly zero"
        report.append(f"{name} {moved}/{len(params)} tensors non-zero")
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return "ok (" + ", ".join(report) + ")"


def check_controls(result) -> str:
    """Unchanged graphs stay inside the tolerance they define."""
    controls = [m for m in result.measurements if m.group == CONTROL_GROUP]
    if not controls:
        return "FAIL: the probe set has no controls"
    worst = max(m.rms for m in controls)
    flagged = [m.name for m in controls if m.detected]
    if flagged:
        return f"FAIL: controls reported as different: {flagged} (worst {worst:.3e})"
    return f"ok (worst control {worst:.3e} against tolerance {result.tolerance.value:.3e})"


def check_probe_is_read_only(model, pool, pairset, index_a, index_b, device, probe_kwargs) -> str:
    """Probing does not update weights."""
    import torch

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    probe(model, pool, pairset, index_a, index_b, update=-1, device=device, **probe_kwargs)
    for name, p in model.named_parameters():
        if not torch.equal(before[name], p.detach()):
            return f"FAIL: probing changed {name}"
    return f"ok ({len(before)} parameter tensors unchanged by a probe)"


def check_reload(run_directory, pool, pairset, index_a, index_b, device, probe_kwargs, reference) -> str:
    """A saved model reproduces the distances it reported."""
    path = os.path.join(run_directory, FINAL_CHECKPOINT)
    if not os.path.isfile(path):
        return f"FAIL: no checkpoint at {path}"
    reloaded, _extra = load_checkpoint(path, device=device)
    result = probe(reloaded, pool, pairset, index_a, index_b, update=-1, device=device, **probe_kwargs)
    got = result.lookup()
    worst, worst_name = 0.0, ""
    for item in reference.measurements:
        other = got.get(item.name)
        if other is None:
            return f"FAIL: the reloaded model produced no measurement for {item.name}"
        delta = abs(other.rms - item.rms)
        if delta > worst:
            worst, worst_name = delta, item.name
    # The floor is the run's own measured non-determinism, not zero: the same
    # weights on the same GPU already disagree with themselves by that much. The
    # relative term covers float32 accumulation; a checkpoint that actually
    # reloaded wrong would be out by a fraction of the token, not a millionth.
    limit = max(reference.tolerance.repeat_max, 1e-6 * reference.token_scale, 1e-9)
    if worst > limit:
        return f"FAIL: {worst_name} moved by {worst:.3e} after reloading (limit {limit:.3e})"
    return f"ok (largest change after reload {worst:.3e}, limit {limit:.3e})"


def plumbing_checks(cfg, dataset, pairset, result: TrainResult) -> tuple[bool, list[str]]:
    """The six pre-flight checks from the plan's final step."""
    device = resolve_device(cfg["train"].get("device", "auto"))
    pool, index_a, index_b = build_pool(dataset, pairset)
    probe_cfg = dict(cfg["probe"])
    probe_kwargs = dict(
        batch_size=int(probe_cfg.get("batch_size", 64)),
        repeats=int(probe_cfg.get("repeats", 3)),
        control_factor=float(probe_cfg.get("tolerance_control_factor", 5.0)),
        rel_floor=float(probe_cfg.get("tolerance_rel_floor", 1e-3)),
        zero_token_eps=float(probe_cfg.get("zero_token_eps", 1e-6)),
    )
    model = build_model(
        cfg["model"], dataset.meta, loss_scales=cfg["train"].get("loss_scales"), device=device
    )
    # A dataset that was not produced by the collect stage -- the tests' -- has
    # no directory to read the packer sample from, and check 1 says so.
    source_dir = dataset_dir(cfg) if cfg.get("collect") else None

    lines = [
        f"1. cache round trip        {check_cache_roundtrip(dataset, source_dir)}",
        f"2. pair edits              {check_pairs(dataset, pairset)}",
        f"3. gradients reach both    {check_gradients(model, pool, device)}",
        f"4. unchanged controls      {check_controls(result.last)}",
        f"5. probing is read-only    "
        f"{check_probe_is_read_only(model, pool, pairset, index_a, index_b, device, probe_kwargs)}",
        f"6. reload reproduces       "
        f"{check_reload(result.run_dir, pool, pairset, index_a, index_b, device, probe_kwargs, result.last)}",
    ]
    passed = not any("FAIL" in line for line in lines)
    print("\n[checks] " + "\n[checks] ".join(lines))
    print(f"[checks] {'all six passed' if passed else 'FAILURES above'}")
    return passed, lines


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(
    path: str,
    cfg: Mapping,
    dataset: GraphDataset,
    pairset: PairSet,
    result: TrainResult,
    checks: Optional[Sequence[str]] = None,
) -> str:
    first, last = result.first, result.last
    summary = dataset.meta.get("summary") or {}
    lines = [
        "# Graph encoder sensitivity probe",
        "",
        "Does a small edit to a packed graph produce a measurably different "
        "encoder token, after reconstruction training on a pool that contains "
        "both members of every pair?",
        "",
        "## Run",
        "",
        f"- environment: `{dataset.meta.get('env_id')}` "
        f"({summary.get('attempts', '?')} attempts, {summary.get('successes', '?')} succeeded)",
        f"- dataset: {len(dataset)} graphs, {summary.get('unique_frames', '?')} of them distinct, "
        f"cameras {dataset.meta.get('cameras')}",
        f"- training pool: {result.pool_size} frames (collected + edited copies), "
        f"no held-out split by design",
        f"- probe set: {len(pairset)} pairs, fixed before the first update",
        f"- updates: {result.updates} ({result.stop_reason}); "
        f"{'plateau' if result.converged else 'the budget was reached, which is not convergence'}",
        f"- device {result.device}, float32, monitor subset {result.monitor_size} frames",
        f"- collection revision `{dataset.meta.get('revision')}`, run revision `{git_revision()}`",
        f"- {last.tolerance.describe()}",
        "",
        "## By edit group",
        "",
        group_table(first, last),
        "",
        "## Every pair",
        "",
        f"Distances are RMS over the {int(cfg['model']['simple_units'])}-dimensional pooled token. "
        f"Full per-probe history is in `{PROBE_ROWS}`.",
        "",
        final_table(first, last, pairset),
        "",
    ]
    if result.decoder is not None:
        from .decoder_eval import report_table

        note = (
            "These frames were **held out** of training."
            if result.eval_split == "holdout"
            else "These are **training** frames -- the experiment defines no held-out "
            "split, so this measures what the decoder recovered from graphs it was "
            "fitted on, not generalisation. `train.holdout_frames` carves a real one."
        )
        lines += [
            "## What the decoder recovered",
            "",
            "Each head's argmax against the packed truth, under the decoder's own "
            "masks: the target among admissible rows, the absolute label among "
            "those its relation may legally take, the temporal label among the "
            "non-padding classes.",
            "",
            note,
            "",
            report_table(result.decoder),
            "`decoder_confusion.png` shows which label is mistaken for which, "
            "`decoder_accuracy.png` how agreement moved during training, and "
            "`decoder_examples.png` a few frames item by item. "
            "`decoder_predictions.csv` is the same comparison, one row per node "
            "and per fact.",
            "",
        ]
    if result.warnings:
        lines += ["## Flags", "", *[f"- {item}" for item in result.warnings], ""]
    if checks:
        lines += ["## Plumbing checks", "", "```", *checks, "```", ""]
    lines += [
        "## What this does and does not show",
        "",
        "- Measured: the encoder's pooled token, before any RSSM processing, on "
        "graphs the model was trained on.",
        "- Not measured: unseen-graph generalisation, and any behaviour of the "
        "full Dreamer stack.",
        "- Some edited graphs are physically inconsistent on purpose -- one field "
        "moves while the fields that would co-vary with it are pinned.",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="graph_encoder_probe/config.yaml")
    parser.add_argument(
        "--stage", default="all", choices=["all", "collect", "pairs", "train"],
        help="stop after this stage; 'train' includes the report",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="short collection, a few updates, and the six plumbing checks")
    parser.add_argument("--checks", action="store_true",
                        help="run the plumbing checks after training (implied by --smoke)")
    parser.add_argument("--recollect", action="store_true", help="ignore an existing cache")
    parser.add_argument("--rebuild-pairs", action="store_true", help="redraw the probe set")
    parser.add_argument("--run-name", default="", help="output directory name under runs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one config leaf, e.g. --set train.max_updates=500")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.smoke:
        apply_overrides(cfg, SMOKE_OVERRIDES)
        if not cfg["output"].get("run_name"):
            cfg["output"]["run_name"] = "smoke"
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name

    dataset = stage_collect(cfg, force=args.recollect)
    if args.stage == "collect":
        return 0

    pairset = stage_pairs(cfg, dataset, force=args.rebuild_pairs)
    if args.stage == "pairs":
        return 0

    directory = run_dir(cfg)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, RESOLVED_CONFIG), "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    result = train(cfg, dataset, pairset, directory)

    checks: Optional[list[str]] = None
    passed = True
    if args.smoke or args.checks:
        passed, checks = plumbing_checks(cfg, dataset, pairset, result)

    report = write_report(
        os.path.join(directory, REPORT_NAME), cfg, dataset, pairset, result, checks
    )
    with open(os.path.join(directory, "result.json"), "w") as handle:
        json.dump(
            {
                "updates": result.updates,
                "stop_reason": result.stop_reason,
                "converged": result.converged,
                "tolerance": result.last.tolerance.__dict__,
                "by_group_before": result.first.by_group,
                "by_group_after": result.last.by_group,
                "warnings": result.warnings,
                "checks_passed": passed,
                "decoder_eval": (
                    None if result.decoder is None
                    else {
                        "split": result.eval_split,
                        "frames": result.eval_frames,
                        **{f"acc/{k}": v.accuracy for k, v in result.decoder.heads.items()},
                        "bbox_mae": result.decoder.bbox_mae,
                    }
                ),
            },
            handle,
            indent=2,
        )
    print(f"\n[run] report written to {report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
