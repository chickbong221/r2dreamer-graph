# sim_vla

A world model, a pretrained SmolVLA action expert, and one switch that decides
whether the scene graph is in the pipeline at all.

The existing simulator code is unchanged. Everything here imports from it —
`networks`, `rssm`, `graph`, `envs.maniskill`, `scenegraph` — and adds its own
training loops beside them.

## The three arms

| Experiment | `model.graph.enabled` | `model.progress.enabled` | actor/critic state |
|---|---|---|---|
| Dreamer + SmolVLA | false | false | `(h, z)` |
| + graph | true | false | `(h, z, g)` |
| + graph + progress | true | true | `(h, z, g)` |

Arms 1 and 2 differ in one switch, which is what isolates the graph's
contribution. Arm 3 is compared against arm 2, not arm 1: it measures the
progress method, not the graph.

`progress.enabled` without `graph.enabled` is refused at config load. Its
targets come from the graph schedule, and there is no schedule without a graph.

**Graph disabled means disabled everywhere.** No graph encoder, no `g`, no
graph decoder, no graph losses, no graph-derived progress, and no graph arrays
in the batch — `sim_vla/data/dataset.py` does not open them, so a baseline
batch has no graph key to leak through. The acceptance test corrupts and then
deletes the stored graphs and asserts a baseline's batches are byte-identical.

An arm is chosen **before** its world model is trained. A graph-trained world
model with `g` hidden from the actor is not a baseline: the graph has already
reached `h` and `z`. `sim_vla/runtime/checkpoint.py` refuses to load one as the
other rather than letting it happen quietly.

## Terminations

The online env is built with `ignore_terminations=True`, matching
`envs/maniskill.py`. Nothing terminates: episodes end at the horizon and the
value function bootstraps there.

The demonstrations were recorded without that wrapper and do carry terminal
flags — ManiSkill sets `terminated` from the task's success. Those flags are
kept in the dataset as diagnostics and are **not** what the loader reports:
`is_terminal` is false everywhere, so the continuation head is not trained on a
signal the rollout never produces. `ignore_terminations=False` restores the
recording's own semantics, for a trainer configured the same way.

First success and settled success are tracked separately. The flag flickers —
PickCube's success wants a static robot as well as a placed cube — so the step
an episode *reaches* success and the step it *keeps* success are different
numbers, and neither is inferred from the other.

## Stages

```
demonstrations ──► 1A world-model pretraining (per arm, separate checkpoints)
                   └─► 1B adapter + SmolVLA action expert, frozen world model
                       └─► 2 online model-based RL
```

The adapter is a single state token: world-model features → LayerNorm → MLP →
one conditioning token, beside the fixed task instruction. No context-token
block. Only the input width differs between arms, and
`LatentAdapter.parameter_report()` states the resulting capacity difference.

The actor is the real pretrained SmolVLA expert, trained with flow matching.
The online actor update differentiates the imagined return with respect to the
sampled action — it is **not** `log pi(a) * advantage`, because a flow policy
has no tractable log-probability and the flow-matching loss is a regression,
not a log-probability. Both arms use the same objective.

## Running it

Tests, staged, stopping at the first failure:

```bash
bash sim_vla/run_tests.sh
```

A stage that skipped everything reports **INCOMPLETE** and exits 2. A missing
dependency never reads as a pass.

Pin the pretrained checkpoint first; it writes the revision to put in
`configs/base.yaml` and prints the module tree the actor resolves against:

```bash
python -m sim_vla.download_pretrained --out data/pretrained
```

Training is separate from the test suite and is never invoked by it:

```bash
python -m sim_vla.training.pretrain_world_model --task pickcube --experiment graph --steps 50000
```

## Layout

```
configs/      base + task + experiment yaml; the switch lives here
data/         collection, audit, loader, sequences, normalization, replay
models/       world_model, latent_adapter, smolvla_actor, flow_sampler, critics
envs/         the online env, matched to the dataset's recorded contract
training/     pretrain, imitation, imagination, actor_critic, progress, online
evaluation/   world-model diagnostics, policy rollouts, arm comparison
runtime/      checkpoints that carry the decisions their weights depend on
tests/        stages 3–9; stages 1–2 are in the repo's top-level tests/
```
