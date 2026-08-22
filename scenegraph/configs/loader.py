"""Load thresholds.yaml into the runtime ``cfg``.

The runtime relevance gate is the per-subtask whitelist directory
(``whitelists.dir``). The affordance asset is still
required; the whitelist directory is required ONLY when the probe runs the
selector (it's resolved lazily at episode reset). Pass ``require_assets=False``
to skip the affordance check -- useful for unit tests that wire their own
minimal config.

Relation bin edges are NOT read from here. They come from the mined whitelist
union asset, which the graph builder binds per subtask; this file carries only
the settings a demonstration cannot mine (contact/grasp/support predicates,
compatibility normalizers, selection capacity).
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from ..core.affordance import load_affordance_set
from ..core.whitelist import whitelist_group_dir


_MISSING_AFFORDANCE_MSG = (
    "Affordance asset missing or empty at {path!r}.\n"
    "It is mined per MS-HAB task; mine this run's group first:\n"
    "  python -m scenegraph.tools.prepare_assets \\\n"
    "      --mshab-task {group} --subtask pick"
)


def _abs_asset_path(cfg_dir: str, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(cfg_dir, rel))


def load_config(
    path: Optional[str] = None,
    *,
    task_group: Optional[str] = None,
    require_assets: bool = True,
) -> dict:
    """Read ``thresholds.yaml`` and resolve its per-task-group asset paths.

    A falsy ``path`` means "use the packaged thresholds": callers reading from
    elements.Config cannot express None and pass "" instead.

    ``task_group`` is the MS-HAB task being run (``set_table``, ``tidy_house``,
    ...). Both mined assets are namespaced by it -- ``affordances/<group>.json``
    and ``subtask_whitelists/<group>/`` -- because the same object is mined
    against a different scene in each task, and a file from the wrong group
    loads and validates perfectly while describing furniture the run will never
    see. It is required whenever assets are.
    """
    if not path:
        path = os.path.join(os.path.dirname(__file__), "thresholds.yaml")
    cfg_dir = os.path.dirname(os.path.abspath(path))

    with open(path) as f:
        raw = yaml.safe_load(f)

    group = str(task_group or "")
    if require_assets and not group:
        raise ValueError(
            "load_config needs task_group when require_assets is set: the "
            "affordance and whitelist assets are mined per MS-HAB task and "
            "there is no task-independent default to fall back to"
        )

    affordances_cfg = dict(raw.get("affordances", {"asset_dir": "affordances"}))
    affordances_dir = _abs_asset_path(cfg_dir, affordances_cfg.get("asset_dir"))
    affordances_cfg["asset_dir_abs"] = affordances_dir
    affordances_cfg["asset_path_abs"] = (
        os.path.join(affordances_dir, f"{group}.json")
        if affordances_dir and group else None
    )

    whitelists_cfg = dict(raw.get("whitelists", {"dir": "subtask_whitelists"}))
    whitelists_root = _abs_asset_path(cfg_dir, whitelists_cfg.get("dir"))
    whitelists_cfg["root_abs"] = whitelists_root
    whitelists_cfg["dir_abs"] = whitelist_group_dir(whitelists_root, group)

    selection_cfg = dict(raw.get("selection") or {})
    selection_cfg.setdefault("n_max", 11)

    aff_set = load_affordance_set(affordances_cfg["asset_path_abs"])

    if require_assets:
        if aff_set.is_empty():
            raise FileNotFoundError(
                _MISSING_AFFORDANCE_MSG.format(
                    path=affordances_cfg["asset_path_abs"], group=group)
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
        "task_group": group,
    }
    if "compat_norm" in raw and isinstance(raw["compat_norm"], dict):
        cfg["compat_norm"] = dict(raw["compat_norm"])
    return cfg
