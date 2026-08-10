"""Load thresholds.yaml and flatten the chosen scale profile into ``cfg``.

The runtime relevance gate is the per-subtask whitelist directory
(``whitelists.dir``). The affordance asset is still
required; the whitelist directory is required ONLY when the probe runs the
selector (it's resolved lazily at episode reset). Pass ``require_assets=False``
to skip the affordance check -- useful for unit tests that wire their own
minimal config.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from ..core.affordance import load_affordance_set


_MISSING_AFFORDANCE_MSG = (
    "Affordance asset missing or empty at {path!r}.\n"
    "Mine it before running the probe:\n"
    "  python -m scenegraph.tools.prepare_assets \\\n"
    "      --mshab-task tidy_house prepare_groceries set_table --subtask pick"
)


def _abs_asset_path(cfg_dir: str, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(cfg_dir, rel))


def load_config(
    profile: str,
    path: Optional[str] = None,
    *,
    require_assets: bool = True,
) -> dict:
    """profile in {"tabletop", "room_scale"}.

    A falsy ``path`` means "use the packaged thresholds": callers reading from
    elements.Config cannot express None and pass "" instead.
    """
    if not path:
        path = os.path.join(os.path.dirname(__file__), "thresholds.yaml")
    cfg_dir = os.path.dirname(os.path.abspath(path))

    with open(path) as f:
        raw = yaml.safe_load(f)

    if profile not in raw["profiles"]:
        raise ValueError(
            f"unknown profile {profile!r}; have {list(raw['profiles'])}"
        )

    affordances_cfg = dict(raw.get("affordances", {"asset_path": "affordances.json"}))
    affordances_cfg["asset_path_abs"] = _abs_asset_path(
        cfg_dir, affordances_cfg.get("asset_path")
    )

    whitelists_cfg = dict(raw.get("whitelists", {"dir": "subtask_whitelists"}))
    whitelists_cfg["dir_abs"] = _abs_asset_path(cfg_dir, whitelists_cfg.get("dir"))

    selection_cfg = dict(raw.get("selection") or {})
    selection_cfg.setdefault("n_max", 11)
    selection_cfg.setdefault("k_persist", -1)

    aff_set = load_affordance_set(affordances_cfg["asset_path_abs"])

    if require_assets:
        if aff_set.is_empty():
            raise FileNotFoundError(
                _MISSING_AFFORDANCE_MSG.format(path=affordances_cfg["asset_path_abs"])
            )

    cfg = {
        "temporal": raw["temporal"],
        "contact": raw["contact"],
        "grasp": raw["grasp"],
        "support": raw["support"],
        "affordances": affordances_cfg,
        "affordance_set": aff_set,
        "whitelists": whitelists_cfg,
        "whitelist_dir": whitelists_cfg["dir_abs"],
        "selection": selection_cfg,
        "profile": raw["profiles"][profile],
        "profile_name": profile,
    }
    if "compat_norm" in raw and isinstance(raw["compat_norm"], dict):
        cfg["compat_norm"] = dict(raw["compat_norm"])
    return cfg
