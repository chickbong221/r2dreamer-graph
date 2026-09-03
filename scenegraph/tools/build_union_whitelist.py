"""Merge per-target whitelists into one ``<subtask>_all.json``.

The merged file is not a runtime membership whitelist -- an episode binds the
per-target file for its own target. What the runtime reads from it is
``bin_edges``, the one relation-bin set the whole run shares. Each per-target
file calibrates its bins against the scenes that target appeared in, so binding
those per episode would leave the same relation token meaning a different
metric distance from one episode to the next.

Bin edges are re-derived rather than copied: this takes the elementwise maximum
of the observed statistics and runs the same ``derive_bin_edges`` the miner
uses, so the merged bins are never narrower than any target's and nothing
clips. Members and roles are unioned for inspection only.

The merge is per task group and never crosses one. ``--whitelist-dir`` points
at a single group's directory, and sources disagreeing on ``task_group`` abort
the merge: bins widened by another task's scene would stretch this task's
relation tokens to distances its own demonstrations never produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from scenegraph.core.whitelist import (
    WHITELIST_SCHEMA_VERSION,
    derive_bin_edges,
)

UNION_TARGET = 'all'


def _sources(whitelist_dir: Path, subtask: str) -> List[Path]:
    """Per-target files for one subtask, excluding a previous union."""
    return sorted(
        p for p in whitelist_dir.glob(f'{subtask}_*.json')
        if p.name != f'{subtask}_{UNION_TARGET}.json'
    )


def merge(whitelist_dir: Path, subtask: str) -> Dict:
    """Union of every per-target whitelist for ``subtask`` in one task group."""
    members: Dict[str, Dict] = {}
    robust: Dict[str, float] = {}
    # Site declarations are reviewed task semantics, identical in every
    # per-target file of one group, so the union carries them through rather
    # than deriving anything.
    sites: Dict[str, Dict] = {}
    rollouts = 0
    groups: Dict[str, str] = {}
    policies = set()
    sources = _sources(whitelist_dir, subtask)
    if not sources:
        raise FileNotFoundError(
            f'no {subtask}_*.json under {whitelist_dir}; mine the per-target '
            'whitelists first')

    for path in sources:
        raw = json.loads(path.read_text())
        groups[path.name] = str(raw.get('task_group') or '')
        policies.add(str(raw.get('membership_policy') or ''))
        rollouts += int(raw.get('_n_successful_rollouts') or 0)
        for key, entry in (raw.get('members') or {}).items():
            merged = members.setdefault(
                key, {'roles': set(), 'interaction_types': set(),
                      'supports': set(), 'kind': entry.get('kind')})
            merged['roles'] |= set(entry.get('roles') or ())
            merged['interaction_types'] |= set(entry.get('interaction_types') or ())
            merged['supports'] |= set(entry.get('supports') or ())
            # Classification travels with the member. Dropping it here is
            # what leaves the runtime demanding a shared height scale the
            # per-target files deliberately no longer carry.
            for field in ('family', 'structural_surface',
                          'structural_surface_reason', 'name'):
                if entry.get(field) is not None:
                    previous = merged.get(field)
                    if previous is not None and previous != entry[field]:
                        raise ValueError(
                            f'{key!r} is {field}={previous!r} in one target '
                            f'whitelist and {entry[field]!r} in another. One '
                            'member cannot carry two classifications; re-mine '
                            'the group.')
                    merged[field] = entry[field]
        for key, entry in (raw.get('sites') or {}).items():
            previous = sites.get(key)
            if previous is not None and previous != entry:
                raise ValueError(
                    f'site {key!r} is declared differently in two target '
                    f'whitelists of this group. Declarations are reviewed '
                    'task semantics and must be identical.')
            sites[key] = entry
        for stat, value in (raw.get('bin_stats_robust') or {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            robust[stat] = max(robust.get(stat, value), value)

    distinct = sorted(set(groups.values()))
    if len(distinct) != 1 or not distinct[0]:
        raise ValueError(
            f'{whitelist_dir} mixes task groups {distinct}: ' +
            ', '.join(f'{name}={group or "<none>"}'
                      for name, group in sorted(groups.items())) +
            '. One directory holds one task group; relation bins are not '
            'merged across groups.')
    task_group = distinct[0]

    out_members = {}
    for key, entry in sorted(members.items()):
        out = {
            'roles': sorted(entry['roles']),
            'interaction_types': sorted(entry['interaction_types']),
            'kind': entry['kind'],
        }
        if entry['supports']:
            out['supports'] = sorted(entry['supports'])
        for field in ('family', 'structural_surface',
                      'structural_surface_reason', 'name'):
            if entry.get(field) is not None:
                out[field] = entry[field]
        out_members[key] = out

    return {
        '_schema_version': WHITELIST_SCHEMA_VERSION,
        'subtask': subtask,
        'task_group': task_group,
        'membership_policy': sorted(policies)[0] if len(policies) == 1 else 'mixed',
        'target': UNION_TARGET,
        'members': out_members,
        'sites': sites,
        'bin_edges': derive_bin_edges(robust),
        'bin_stats_robust': robust,
        '_n_successful_rollouts': rollouts,
        '_merged_from': [p.name for p in sources],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--whitelist-dir', required=True)
    parser.add_argument('--subtask', nargs='+', default=['pick'])
    args = parser.parse_args(argv)

    whitelist_dir = Path(args.whitelist_dir)
    for subtask in args.subtask:
        try:
            data = merge(whitelist_dir, subtask)
        except FileNotFoundError as exc:
            print(f'[union] skip {subtask}: {exc}', file=sys.stderr)
            continue
        except ValueError as exc:
            print(f'[union] FAILED {subtask}: {exc}', file=sys.stderr)
            return 1
        out = whitelist_dir / f'{subtask}_{UNION_TARGET}.json'
        out.write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f'[union] wrote {out.name}: {len(data["members"])} members from '
              f'{len(data["_merged_from"])} targets '
              f'(task group {data["task_group"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
