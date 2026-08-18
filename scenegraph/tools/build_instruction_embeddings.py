"""Embed one instruction per ``(subtask, object)`` the task plans can produce.

Reads the MS-HAB task-plan JSON directly rather than building a simulator, so
this runs on a machine with no GPU. Every split named on the command line is
walked and their union is embedded: a category held out of training still needs
a row, because the point of a frozen language encoder is that its vector for an
unseen object is already in a sensible place.

The key format matches ``embodied/envs/instruction.py``: ``<subtask>/actor:
<canonical>``, where the canonical half is the same key the mined whitelists
are named after, so a coverage gap in one shows up as a coverage gap in both.

``--random`` writes the control table: the same keys with random orthonormal
rows instead of language features. A fixed random code followed by the
encoder's first linear layer spans the same functions as a learned embedding
table, so swapping the file is the ablation -- no model change.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List, Set, Tuple

import numpy as np

# "pick up" reads as an instruction; "pick" alone reads as a label.
_VERBS = {
    'pick': 'pick up',
    'place': 'place',
    'open': 'open',
    'close': 'close',
}
_ASSET_PREFIX = re.compile(r'^\d+_')


def _phrase(canonical: str) -> str:
    """``024_bowl`` -> ``bowl``: the part a language model has seen before."""
    name = _ASSET_PREFIX.sub('', canonical).replace('_', ' ').strip()
    return name or canonical


def _instruction(subtask: str, canonical: str) -> str:
    return f'{_VERBS.get(subtask, subtask)} the {_phrase(canonical)}'


def collect_pairs(mshab_tasks: List[str], subtask_dirs: List[str],
                  splits: List[str], obj: str) -> Dict[str, Set[Tuple[str, str]]]:
    """``{split: {(subtask type, canonical key)}}`` straight from the plans.

    Every task and subtask asked for is unioned into one table. Keys carry no
    trace of which long-horizon task produced them, because "pick up the bowl"
    is the same instruction wherever it appears, so one table serves every run.
    """
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    from scenegraph.core.affordance import canonical_affordance_key

    rearrange = ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange'
    found: Dict[str, Set[Tuple[str, str]]] = {s: set() for s in splits}
    for mshab_task in mshab_tasks:
        for subtask_dir in subtask_dirs:
            for split in splits:
                path = (rearrange / 'task_plans' / mshab_task / subtask_dir
                        / split / f'{obj}.json')
                if not path.is_file():
                    # Not every task defines every subtask.
                    print(f'[instr] skip {mshab_task}/{subtask_dir}/{split}: '
                          f'no {path.name}')
                    continue
                pairs: Set[Tuple[str, str]] = set()
                for plan in plan_data_from_file(path).plans:
                    for entry in getattr(plan, 'subtasks', []) or []:
                        obj_id = getattr(entry, 'obj_id', None)
                        kind = getattr(entry, 'type', None)
                        if not obj_id or not kind:
                            continue
                        canonical = canonical_affordance_key(str(obj_id))
                        if canonical:
                            pairs.add((str(kind), str(canonical)))
                found[split] |= pairs
                print(f'[instr] {mshab_task}/{subtask_dir}/{split}: '
                      f'{len(pairs)} (subtask, object) pairs')
    return found


def _embed(texts: List[str], model_name: str) -> np.ndarray:
    """Mean-pooled T5 encoder states over the non-pad tokens of each string."""
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).eval()
    batch = tokenizer(texts, padding=True, return_tensors='pt')
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state
    mask = batch['attention_mask'][..., None].to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return pooled.numpy().astype(np.float32)


def _random(count: int, dim: int, seed: int) -> np.ndarray:
    """Orthonormal rows: distinct, equidistant, carrying no semantics."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((max(count, dim), dim)))[0]
    return np.ascontiguousarray(basis[:count], dtype=np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mshab-task', nargs='+', required=True,
                        help='e.g. prepare_groceries tidy_house set_table')
    parser.add_argument('--subtask', nargs='+', default=['pick'],
                        help='task_plans subdirectories: pick place open close')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--obj', default='all',
                        help='task plan filename stem, matching mshab_obj')
    parser.add_argument('--out', required=True, help='destination .npz')
    parser.add_argument('--model', default='t5-base')
    parser.add_argument('--random', action='store_true',
                        help='write the control table instead of T5 features')
    parser.add_argument('--random-dim', type=int, default=768)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    found = collect_pairs(args.mshab_task, args.subtask, args.splits, args.obj)
    pairs = sorted(set().union(*found.values())) if found else []
    if not pairs:
        print('[instr] no (subtask, object) pairs found; check --mshab-task, '
              '--subtask and --obj against the task_plans tree', file=sys.stderr)
        return 2

    keys = [f'{kind}/actor:{canonical}' for kind, canonical in pairs]
    texts = [_instruction(kind, canonical) for kind, canonical in pairs]

    if args.random:
        vectors = _random(len(keys), args.random_dim, args.seed)
        model = f'random-orthonormal-{args.random_dim}-seed{args.seed}'
    else:
        vectors = _embed(texts, args.model)
        model = args.model
    print(f'[instr] {len(keys)} entries, dim {vectors.shape[1]}, model {model}')

    # Which split each key came from, so the holdout is auditable from the
    # table rather than remembered.
    membership = np.array(
        [','.join(s for s in args.splits if pair in found.get(s, ()))
         for pair in pairs])
    np.savez(
        args.out,
        keys=np.array(keys),
        texts=np.array(texts),
        vectors=vectors,
        model=np.array(model),
        splits=membership,
    )
    print(f'[instr] wrote {args.out}')

    if len(args.splits) > 1:
        first, *rest = args.splits
        for other in rest:
            only_other = sorted(found[other] - found[first])
            only_first = sorted(found[first] - found[other])
            print(f'[instr] in {other} but not {first}: '
                  f'{[c for _, c in only_other] or "none"}')
            print(f'[instr] in {first} but not {other}: '
                  f'{[c for _, c in only_first] or "none"}')
    for key, text in zip(keys, texts):
        print(f'  {key:44s} {text}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
