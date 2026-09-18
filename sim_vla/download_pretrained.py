"""Resolve, optionally snapshot, and report the pretrained SmolVLA checkpoint.

Two things this is careful to say accurately.

**Where the weights go.** By default they go to the shared Hugging Face cache
(``HF_HOME``/``HF_HUB_CACHE``), and ``--out`` receives only the report. Pass
``--snapshot`` to download the files themselves into ``--out`` and load from
there, for a machine that should not depend on that cache.

**What was checked.** ``from_pretrained`` raises on a genuine key mismatch, so
a clean load is itself the check -- but reading ``policy._missing_keys`` and
finding an empty list proves nothing, because the attribute does not exist and
``getattr`` invents the empty list. The report therefore states whether the
attributes were present at all, and says "not reported by this version" when
they were not, rather than printing zeros that look like a verified result.

    python -m sim_vla.download_pretrained --out data/pretrained
    python -m sim_vla.download_pretrained --out data/pretrained --snapshot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .models.pretrained import (DEFAULT_REPO, SUPPORTED_LEROBOT,
                                PretrainedError, load_policy, model_facts)

_SENTINEL = object()


def key_report(policy) -> Dict[str, Any]:
    """Whether this version reports load keys, and what it reported."""
    out: Dict[str, Any] = {}
    for name in ("_missing_keys", "_unexpected_keys"):
        value = getattr(policy, name, _SENTINEL)
        if value is _SENTINEL:
            out[name] = "not reported by this version"
        else:
            out[name] = list(value or [])
    out["note"] = ("from_pretrained raises on a genuine mismatch, so a clean "
                   "load is the check; an empty list here is only meaningful "
                   "if the attribute was actually present")
    return out


def build_report(repo_id: str, revision: str, snapshot_dir) -> Dict[str, Any]:
    loaded = load_policy(repo_id, revision, local_dir=snapshot_dir)
    facts = model_facts(loaded)
    policy = loaded.policy
    return {
        "repo_id": loaded.repo_id,
        "requested_revision": loaded.requested,
        "resolved_revision": loaded.revision,
        "weights_location": loaded.local_dir or "huggingface cache (default)",
        "lerobot_version": loaded.lerobot_version,
        "lerobot_supported": SUPPORTED_LEROBOT,
        "lerobot_matches_supported": loaded.lerobot_version == SUPPORTED_LEROBOT,
        "parameters": int(sum(p.numel() for p in policy.parameters())),
        "model_facts": facts,
        "load_keys": key_report(policy),
        "has_language_tokenizer": getattr(policy, "language_tokenizer", None)
                                  is not None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve and report the pretrained SmolVLA checkpoint")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default="data/pretrained",
                        help="where the report goes; also the snapshot "
                             "directory when --snapshot is passed")
    parser.add_argument("--snapshot", action="store_true",
                        help="download the weights into --out instead of "
                             "leaving them in the Hugging Face cache")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        report = build_report(args.repo_id, args.revision,
                              out / "snapshot" if args.snapshot else None)
    except PretrainedError as exc:
        print(f"[pretrained] FAILED: {exc}")
        return 1

    target = out / "smolvla_loading_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[pretrained] {report['repo_id']}")
    print(f"[pretrained] requested {report['requested_revision']} -> "
          f"resolved {report['resolved_revision']}")
    print(f"[pretrained] weights: {report['weights_location']}")
    print(f"[pretrained] lerobot {report['lerobot_version']} "
          f"(written against {report['lerobot_supported']})")
    if not report["lerobot_matches_supported"]:
        print("[pretrained] WARNING version differs from the one this "
              "integration was written against; run stage 4 before training")
    print(f"[pretrained] {report['parameters']:,} parameters")
    for key, value in report["model_facts"].items():
        print(f"[pretrained]   {key}: {value}")
    print(f"[pretrained] load keys: {report['load_keys']['_missing_keys']} / "
          f"{report['load_keys']['_unexpected_keys']}")
    print(f"[pretrained] wrote {target}")
    print()
    print("Put this in sim_vla/configs/base.yaml:")
    print(f"  actor.revision: {report['resolved_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
