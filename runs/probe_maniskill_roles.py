"""Which whitelist members the runtime actually admits, and whether the
compiled schedule's roles resolve against them.

Nodes are seeded from segmentation. An actor that renders no pixels in any
camera -- a goal marker ManiSkill hides before sensor capture, for instance --
is mined into the whitelist from poses but never becomes a vertex, and a
schedule role bound to it can never resolve. That failure is silent: the
progress target simply reads invalid for every frame.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from hydra import compose, initialize_config_dir

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(
            config_name="configs",
            overrides=[
                "env=maniskill",
                "model=size50M_graph_simple",
                f"env.task=maniskill_{args.task}",
                f"env.env_num={args.num_envs}",
                "env.eval_episode_num=0",
                f"device={args.device}",
                "wandb.enabled=false",
            ],
        )

    import torch

    from envs import make_envs
    from envs.maniskill import task_schedule_source
    from scenegraph.adapters.graph_vocab import build_entity_vocab
    from scenegraph.core.schedule import compile_from_files

    print(f"resolved progress.mode = {config.model.progress.mode!r}", flush=True)

    envs, _, _, act_space = make_envs(config.env)
    env_id, whitelist_dir = task_schedule_source(envs)
    vocab = build_entity_vocab(whitelist_dir)
    name_of = {i: k for k, i in vocab.token_to_id.items()}
    schedule = compile_from_files(
        env_id, config.model.progress.schedule_dir,
        str(pathlib.Path(whitelist_dir).parent.parent), vocab,
    )

    n = int(envs.env_num)
    reset = torch.ones(n, dtype=torch.bool, device=args.device)
    seen = set()
    for step in range(args.steps):
        action = torch.as_tensor(
            act_space.sample(), device=args.device, dtype=torch.float32
        )
        if action.ndim == 1:
            action = action.unsqueeze(0).repeat(n, 1)
        trans, _ = envs.step(action, reset)
        reset = torch.zeros(n, dtype=torch.bool, device=args.device)
        seen.update(int(v) for v in trans["graph_node_ent"].reshape(-1).tolist())
    envs.close()

    seen.discard(vocab.pad_id)
    print(f"\nadmitted over {args.steps} steps:")
    for i in sorted(seen):
        print(f"  {i:>3}  {name_of.get(i, '<unknown>')}")
    never = sorted(set(vocab.token_to_id.values()) - seen - {vocab.pad_id})
    if never:
        print("declared in the whitelist but never a vertex:")
        for i in never:
            print(f"  {i:>3}  {name_of.get(i, '<unknown>')}")

    print(f"\nschedule {env_id}: {len(schedule.phases)} phases, "
          f"{sum(len(p.clauses) for p in schedule.phases)} clauses, "
          f"{len(schedule.slots)} distinct facts")
    bad = []
    for role, ent in sorted(schedule.role_entity_ids.items()):
        ok = ent in seen
        print(f"  role {role:<14} -> {schedule.roles.get(role, name_of.get(ent))}"
              f"  (id {ent}) {'ok' if ok else 'NEVER ADMITTED'}")
        if not ok:
            bad.append(role)
    if bad:
        print(f"\nUNRESOLVABLE roles: {', '.join(bad)}. Every phase naming one "
              "reads invalid, so progress/valid_fraction collapses to zero and "
              "the head trains on nothing.")
    else:
        print("\nall schedule roles resolve")


if __name__ == "__main__":
    main()
