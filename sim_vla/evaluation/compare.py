"""Put two arms' results side by side, and say what was held equal.

A comparison is only a comparison if the settings that were supposed to match
did. This reads both runs' checkpoint metadata and reports any difference
beyond the graph flag itself, because a graph arm that also had a different
demonstration set or a different pretrained revision is not evidence about
graphs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Everything that must be identical between arms. The graph flag is excluded:
# it is the one thing that is supposed to differ.
MUST_MATCH = ("env_id", "pretrained_revision", "dataset_identity",
              "normalization_identity")


def load_run(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))


def compare(baseline: str | Path, graph: str | Path,
            results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    left, right = load_run(baseline), load_run(graph)
    differences: List[str] = [
        key for key in MUST_MATCH if left.get(key) != right.get(key)]
    out: Dict[str, Any] = {
        "baseline": str(baseline),
        "graph": str(graph),
        "graph_flags": [left.get("graph_enabled"), right.get("graph_enabled")],
        "uncontrolled_differences": differences,
        "feature_dims": [left.get("feature_dim"), right.get("feature_dim")],
        "results": results or {},
    }
    if left.get("graph_enabled") == right.get("graph_enabled"):
        out["warning"] = ("both runs have the same graph flag; this is not a "
                          "graph comparison")
    elif differences:
        out["warning"] = (f"these differ besides the graph flag: {differences}. "
                          "The comparison is confounded.")
    return out


def render(report: Dict[str, Any]) -> str:
    lines = [f"[compare] baseline {report['baseline']}",
             f"[compare] graph    {report['graph']}",
             f"[compare] feature dims {report['feature_dims']}"]
    for arm, values in (report.get("results") or {}).items():
        lines.append(f"[compare] {arm}: "
                     + ", ".join(f"{k}={v}" for k, v in values.items()))
    if "warning" in report:
        lines.append(f"[compare] WARNING {report['warning']}")
    return "\n".join(lines)
