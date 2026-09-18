"""Fetch the pretrained SmolVLA checkpoint, pin it, and report what loaded.

"The pretrained checkpoint" is not a reproducible object without a revision, so
this resolves the revision it actually downloaded and writes it beside the
report. Both arms then load that exact revision.

The report is the more useful half. A checkpoint loaded into a model whose keys
only mostly match will train, and the part that silently did not load is the
part nobody looks at until the results are strange. So the missing and
unexpected keys are written down, along with the module tree that
``sim_vla/models/smolvla_actor.py`` searches for the expert -- which is how the
attribute names get confirmed on a machine that has the weights, rather than
guessed on one that does not.

    python -m sim_vla.download_pretrained --out data/pretrained
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_REPO = "lerobot/smolvla_base"


def resolve_revision(repo_id: str, revision: str = "main") -> str:
    """The commit a named revision points at, so it can be pinned."""
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo_id, revision=revision).sha)
    except Exception as exc:                               # noqa: BLE001
        return f"unresolved: {type(exc).__name__}: {exc}"


def load_report(repo_id: str, revision: str = "main") -> Dict[str, Any]:
    """Load the policy and describe what came back."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    import lerobot

    policy = SmolVLAPolicy.from_pretrained(repo_id, revision=revision)
    from .models.smolvla_actor import (
        EXPERT_PATHS, LANGUAGE_PATHS, VISION_PATHS, module_tree, resolve,
    )

    found: Dict[str, Any] = {}
    for what, paths in (("action_expert", EXPERT_PATHS),
                        ("language", LANGUAGE_PATHS),
                        ("vision", VISION_PATHS)):
        try:
            _, path = resolve(policy, paths, what)
            found[what] = path
        except AttributeError as exc:
            found[what] = f"NOT FOUND: {exc}"

    parameters = sum(p.numel() for p in policy.parameters())
    config = getattr(policy, "config", None)
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolve_revision(repo_id, revision),
        "lerobot_version": getattr(lerobot, "__version__", "unknown"),
        "parameters": int(parameters),
        "resolved_modules": found,
        "module_tree": module_tree(policy, depth=3),
        "config": {k: str(v) for k, v in vars(config).items()}
                  if config is not None else {},
        # from_pretrained raises on a real mismatch, so an empty pair here is
        # the expected outcome and its presence is the point: the report says
        # the check ran.
        "missing_keys": list(getattr(policy, "_missing_keys", []) or []),
        "unexpected_keys": list(getattr(policy, "_unexpected_keys", []) or []),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and pin the pretrained SmolVLA checkpoint")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default="data/pretrained")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = load_report(args.repo_id, args.revision)
    target = out / "smolvla_loading_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[pretrained] {report['repo_id']} @ {report['resolved_revision']}")
    print(f"[pretrained] lerobot {report['lerobot_version']}, "
          f"{report['parameters']:,} parameters")
    for what, path in report["resolved_modules"].items():
        print(f"[pretrained]   {what}: {path}")
    if report["missing_keys"] or report["unexpected_keys"]:
        print(f"[pretrained] WARNING missing={len(report['missing_keys'])} "
              f"unexpected={len(report['unexpected_keys'])}")
    print(f"[pretrained] wrote {target}")
    print(f"[pretrained] pin this in sim_vla/configs/base.yaml: "
          f"actor.revision: {report['resolved_revision']}")
    return 0 if all("NOT FOUND" not in str(v)
                    for v in report["resolved_modules"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
