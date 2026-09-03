"""Mine every offline asset one MS-HAB task group needs, in one command.

Five stages, each a tool that still runs standalone: collect successful
rollouts, mine affordances, mine raw whitelists, prune them into the runtime
assets, embed instructions.

**One task group per invocation.** The same object is a different mining
problem in each task -- set_table's bowl rests in a counter drawer,
prepare_groceries' on the counter -- so every path this writes is namespaced by
group and nothing it touches belongs to another group::

    $MS_ASSET_DIR/data/robot_success_states/fetch/<group>/<subtask>/*.pkl
    scenegraph/configs/affordances/<group>.json
    scenegraph/configs/subtask_whitelists_raw/<group>/
    scenegraph/configs/subtask_whitelists/<group>/

Run it once per group::

    python -m scenegraph.tools.prepare_assets --mshab-task set_table \\
        --subtask pick --membership-policy full-evidence

Start with ``--dry-run``: it prints the coverage report and every subcommand
without running any of them, and collection is measured in sim-hours.

``--clean`` deletes the previous artifacts **of the selected group only**. The
miners only write the keys they mined, so without it a whitelist for an object
no longer in the task plans survives the rebuild and is still loadable.

The instruction table is the one asset that is not per group: its keys are
``<subtask>/actor:<object>`` and "pick up the bowl" is the same instruction
wherever it appears, so one table serves every run. ``--clean`` therefore never
deletes it, and it is rebuilt over ``--instruction-task`` (every MS-HAB task by
default) so mining one group cannot shrink it out from under another.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from scenegraph.tools import collect_robot_success_states

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / 'scenegraph' / 'configs'

# The MS-HAB long-horizon tasks. Used only as the default instruction coverage;
# collect_pairs skips a task whose plans are not installed.
MSHAB_TASKS = ('prepare_groceries', 'set_table', 'tidy_house')


def _plan_pairs(mshab_task, subtasks, splits, obj):
    """``{split: {(subtask, canonical)}}`` the task plans can produce."""
    from scenegraph.tools.build_instruction_embeddings import collect_pairs
    return collect_pairs([mshab_task], subtasks, splits, obj)


def _checkpointed_objects(ckpt_root: Path, mshab_task: str,
                          subtasks: List[str], algo: str = '') -> Set[str]:
    """Objects the collector can drive under THIS task group.

    Scoped to the group on purpose: a bowl policy released under
    prepare_groceries cannot roll out set_table's drawer scene, so counting it
    here would report coverage this group does not have.

    ``algo`` is the training-algorithm tree above the task level, and has to
    match what the collector will use -- a coverage report read off one
    algorithm while the collection runs another describes a different set of
    policies.
    """
    prefix = f'{algo}/' if algo else ''
    found = set()
    for subtask in subtasks:
        for pt in ckpt_root.glob(f'{prefix}{mshab_task}/{subtask}/*/policy.pt'):
            found.add(pt.parent.name)
    return found


def _unreadable_supporters(runtime_dir: Path) -> List[str]:
    """``<file>: <key>`` for every admitted supporter with no extent.

    A supporter is admitted on interaction evidence alone, but whether it is
    an extended surface is decided by its collision extent. A member the
    classification could not reach carries no family, and the runtime then
    measures it from its actor origin -- ~0.9m below a counter's own top,
    which is the error the classification exists to remove. A warning is fine
    while iterating; a finished asset is not.
    """
    import json as _json

    out: List[str] = []
    for path in sorted(Path(runtime_dir).glob('*.json')):
        try:
            members = _json.loads(path.read_text()).get('members') or {}
        except (OSError, ValueError):
            continue
        for key, entry in sorted(members.items()):
            if not isinstance(entry, dict):
                continue
            if 'support' not in (entry.get('interaction_types') or ()):
                continue
            if entry.get('family'):
                continue
            out.append(f'{path.name}: {key}')
    return out


def _report(needed: Dict[str, Set[Tuple[str, str]]], ckpt_objects: Set[str],
            whitelist_dir: Path, table: Path) -> List[Tuple[str, str]]:
    """Print what the plans want against what exists. Returns uncollectable."""
    from scenegraph.core.whitelist import resolve_whitelist_path

    pairs = sorted(set().union(*needed.values())) if needed else []
    print(f'\n[prep] task plans name {len(pairs)} (subtask, object) pairs')

    have_table = set()
    if table.is_file():
        import numpy as np
        have_table = {str(k) for k in np.load(table, allow_pickle=False)['keys']}

    uncollectable = []
    for kind, canonical in pairs:
        wl = resolve_whitelist_path(str(whitelist_dir), kind, f'actor:{canonical}')
        marks = [
            'ckpt' if canonical in ckpt_objects else 'NO-CKPT',
            'wl' if wl else 'no-wl',
            'instr' if f'{kind}/actor:{canonical}' in have_table else 'no-instr',
        ]
        print(f'  {kind:6s} {canonical:28s} {" ".join(marks)}')
        if canonical not in ckpt_objects:
            uncollectable.append((kind, canonical))

    if uncollectable:
        print(f'\n[prep] {len(uncollectable)} object(s) have no per-object '
              'policy under --ckpt-root for this task group, so no rollouts '
              'can be collected for them and they will have no whitelist:')
        for kind, canonical in uncollectable:
            print(f'  {kind}/{canonical}')
    return uncollectable


def _clean(paths: List[Path], dry_run: bool) -> None:
    for path in paths:
        if not path.exists():
            continue
        print(f'  rm -r {path}')
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _run(cmd: List[str], dry_run: bool) -> None:
    print('  $ ' + ' '.join(str(c) for c in cmd))
    if dry_run:
        return
    # -u because collection runs for hours and its output is usually piped to
    # a log: Python block-buffers stdout at 8KB when it is not a terminal, so
    # progress would arrive in bursts and anything still buffered when the
    # process is killed would never be written at all.
    code = subprocess.call([sys.executable, '-u', '-m', *cmd], cwd=REPO)
    if code != 0:
        raise SystemExit(f'[prep] stage failed with exit code {code}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--mshab-task', required=True,
        help='The ONE task group to prepare. Run the command again per group; '
             'passing several would make each shared object mine in whichever '
             'task sorted first.',
    )
    parser.add_argument('--subtask', nargs='+', default=['pick'])
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--obj', default='all')
    parser.add_argument(
        '--ckpt-root',
        default=str(collect_robot_success_states.DEFAULT_CKPT_ROOT))
    parser.add_argument(
        '--algo', default=collect_robot_success_states.DEFAULT_CKPT_ALGO,
        help='Training-algorithm tree under --ckpt-root (bc, dp, rl). '
             'Forwarded to the collector so the coverage report and the '
             'collection read the same policies.')
    parser.add_argument('--asset-dir', default=None,
                        help='data root holding robot_success_states/; '
                             'defaults to $MS_ASSET_DIR/data then ~/.maniskill/data')
    parser.add_argument('--robot', default='fetch')
    parser.add_argument('--n-success', type=int, default=30)
    parser.add_argument('--num-envs', type=int, default=35)
    parser.add_argument('--model', default='t5-base')
    parser.add_argument(
        '--membership-policy', default='full-evidence',
        choices=['full-evidence', 'target-supporters'],
        help='What the RAW whitelists keep. full-evidence (default) preserves '
             'every interacted entity so a different runtime rule costs a '
             're-prune instead of another collection run.',
    )
    parser.add_argument(
        '--prune-policy', default='target-supporters',
        choices=['full-evidence', 'target-supporters'],
        help='What the RUNTIME whitelists keep, pruned from the raw ones.',
    )
    parser.add_argument(
        '--instruction-task', nargs='+', default=list(MSHAB_TASKS),
        help='Task groups the instruction table must cover. Defaults to every '
             'MS-HAB task so preparing one group never shrinks the shared '
             'table out from under another.',
    )
    parser.add_argument(
        '--allow-missing-checkpoints', action='store_true',
        help='Prepare the covered targets even though some target the task '
             'plans name has no per-object policy. Off by default: the run '
             'would otherwise spend sim-hours and still end with gaps.',
    )
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-collect', action='store_true')
    parser.add_argument('--skip-affordances', action='store_true')
    parser.add_argument('--skip-whitelists', action='store_true')
    parser.add_argument('--skip-prune', action='store_true')
    parser.add_argument('--skip-instructions', action='store_true')
    args = parser.parse_args(argv)

    import os
    group = args.mshab_task
    asset_dir = Path(args.asset_dir or (
        Path(os.environ.get('MS_ASSET_DIR', os.path.expanduser('~/.maniskill')))
        / 'data')).resolve()
    success_root = asset_dir / 'robot_success_states'
    group_success = success_root / args.robot / group
    raw_dir = CONFIGS / 'subtask_whitelists_raw' / group
    runtime_dir = CONFIGS / 'subtask_whitelists' / group
    affordances = CONFIGS / 'affordances' / f'{group}.json'
    table = CONFIGS / 'instructions.npz'
    ckpt_root = (REPO / args.ckpt_root).resolve()

    print(f'[prep] task group {group!r}, subtasks {args.subtask}')
    needed = _plan_pairs(group, args.subtask, args.splits, args.obj)
    ckpt_objects = _checkpointed_objects(
        ckpt_root, group, args.subtask, args.algo)
    uncollectable = _report(needed, ckpt_objects, runtime_dir, table)

    # Stop here, before --clean deletes anything and before hours of GPU
    # collection, rather than at the verify step at the end. A target with no
    # per-object policy under this group cannot produce rollouts, so the run is
    # already known to end in missing whitelists.
    if uncollectable and not args.allow_missing_checkpoints:
        print(
            f'[prep] ABORT: {len(uncollectable)} target(s) named by '
            f'{group} task plans have no checkpoint under {ckpt_root}. '
            'Collection would run for hours and still leave them without '
            'a whitelist. Pass --allow-missing-checkpoints to prepare the '
            'covered targets anyway.',
            file=sys.stderr,
        )
        return 1

    if args.clean:
        # Group-scoped, every entry. Nothing here can name another group's
        # rollouts, affordances or whitelists, and the shared instruction
        # table is deliberately absent.
        print(f'\n[prep] clean (task group {group!r} only)')
        targets = [raw_dir, runtime_dir, affordances]
        if args.skip_collect:
            # --clean would otherwise delete the rollouts that --skip-collect
            # exists to reuse, and the run would then mine nothing from an
            # empty tree. Re-mining hours-old evidence is the main reason to
            # combine the two flags, so keep the pickles.
            print('  keeping rollouts under '
                  f'{group_success} (--skip-collect)')
        else:
            targets += [group_success / s for s in args.subtask]
        _clean(targets, args.dry_run)

    if not args.skip_collect:
        print('\n[prep] collect')
        for subtask in args.subtask:
            _run([
                'scenegraph.tools.collect_robot_success_states',
                '--ckpt-root', str(ckpt_root),
                # Forwarded so the collection reads the policies the coverage
                # report above counted.
                '--algo', str(args.algo),
                '--subtask', subtask,
                '--task', group,
                '--n-success', str(args.n_success),
                '--num-envs', str(args.num_envs),
                '--asset-dir', str(asset_dir),
                '--no-skip-done',
            ], args.dry_run)

    if not args.skip_affordances:
        print('\n[prep] affordances')
        # build_affordances creates its own output parent; doing it here too
        # would make --dry-run write to disk.
        for index, subtask in enumerate(args.subtask):
            cmd = [
                'scenegraph.tools.build_affordances',
                '--success-states-dir', str(success_root),
                '--robot', args.robot,
                '--task-group', group,
                '--subtask', subtask,
                '--out', str(affordances),
            ]
            # The first subtask writes the file; the rest add to it.
            if index:
                cmd.append('--merge-existing')
            _run(cmd, args.dry_run)

    if not args.skip_whitelists:
        print(f'\n[prep] raw whitelists ({args.membership_policy})')
        _run([
            'scenegraph.tools.build_subtask_whitelists',
            '--success-states-dir', str(success_root),
            '--robot', args.robot,
            '--task-group', group,
            '--membership-policy', args.membership_policy,
            '--out-dir', str(raw_dir),
            '--affordance-json', str(affordances),
        ], args.dry_run)

        # One merged file per subtask for runs whose target changes each
        # episode. Written from the per-target files, so it follows them.
        _run([
            'scenegraph.tools.build_union_whitelist',
            '--whitelist-dir', str(raw_dir),
            '--subtask', *args.subtask,
        ], args.dry_run)

    if not args.skip_prune:
        print(f'\n[prep] prune -> runtime ({args.prune_policy})')
        _run([
            'scenegraph.tools.prune_whitelists',
            '--raw-dir', str(raw_dir),
            '--out-dir', str(runtime_dir),
            '--task-group', group,
            '--subtask', *args.subtask,
            '--policy', args.prune_policy,
        ], args.dry_run)

    if not args.skip_instructions:
        print('\n[prep] instructions (shared across task groups)')
        _run([
            'scenegraph.tools.build_instruction_embeddings',
            '--mshab-task', *args.instruction_task,
            '--subtask', *args.subtask,
            '--splits', *args.splits,
            '--obj', args.obj,
            '--out', str(table),
            '--model', args.model,
        ], args.dry_run)

    if args.dry_run:
        print('\n[prep] dry run: nothing was written')
        return 0

    print('\n[prep] verify')
    _report(needed, ckpt_objects, runtime_dir, table)
    unreadable = _unreadable_supporters(runtime_dir)
    if unreadable:
        print(f'[prep] FAILED: {len(unreadable)} admitted supporter(s) '
              'carry no height family, so their collision extent was '
              'unreadable and they cannot be tested for being an '
              'extended surface:', file=sys.stderr)
        for line in unreadable[:20]:
            print(f'          {line}', file=sys.stderr)
        if len(unreadable) > 20:
            print(f'          ... and {len(unreadable) - 20} more',
                  file=sys.stderr)
        print('       Extents are read from the simulator at collection '
              'time and cannot be mined later; re-collect these targets.',
              file=sys.stderr)
        return 1
    from scenegraph.core.whitelist import resolve_whitelist_path
    pairs = sorted(set().union(*needed.values())) if needed else []
    gaps = [
        f'{k}/{c}' for k, c in pairs
        if resolve_whitelist_path(str(runtime_dir), k, f'actor:{c}') is None
    ]
    if gaps:
        print(f'[prep] FAILED: {len(gaps)} whitelist(s) still missing for '
              f'{group}: {", ".join(gaps)}', file=sys.stderr)
        return 1
    print(f'[prep] every (subtask, object) pair {group} names is covered')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
