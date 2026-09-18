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
`is_terminal` is false everywhere. `ignore_terminations=False` restores the
recording's own semantics, for a trainer configured the same way.

First success and settled success are tracked separately. The flag flickers —
PickCube's success wants a static robot as well as a placed cube — so the step
an episode *reaches* success and the step it *keeps* success are different
numbers, and neither is inferred from the other.

## The actor

Read against **lerobot 0.6.1**. The flow methods live on `policy.model`
(`VLAFlowMatching`), not on the policy: `embed_prefix`, `embed_suffix`, and
`denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)`.

Conditioning is two passes. The prefix — tokenized instruction plus one state
token — is embedded and run through the VLM once per observation to produce a
key/value cache; every Euler step of the sampler reuses it. Building the prefix
inside the integration loop would be correct and ten times slower.

The state token replaces SmolVLA's raw-state slot. No images are supplied: the
observation already went through the world model, and re-encoding the frame
with the frozen vision tower would encode it twice. `state_token_mode` decides
the width — `embedding` (default) emits `vlm_hidden_size` and bypasses
`state_proj`; `state_proj` emits `max_state_dim` and uses the frozen
projection, which puts a 32-wide bottleneck between the world model and the
policy.

LeRobot's flow convention is `x_t = t*noise + (1-t)*actions` with target
`noise - actions`, so `t=0` is the clean action and sampling integrates
**downward** from 1. `flow_sampler.py` follows that, not the textbook
orientation.

Actions pad to the checkpoint's `max_action_dim` (32) before `action_in_proj`
and are sliced back after `action_out_proj`. The loss and the sampler mask the
padding, so the policy is never scored on axes the Panda does not have.

Frozen means `requires_grad=False`, never `detach()`: gradients flow *through*
the frozen transformer into the adapter, which is the only reason conditioning
it works.

The online actor update differentiates the imagined return with respect to the
sampled action. It is **not** `log pi(a) * advantage` — a flow policy has no
tractable log-probability, and the flow-matching loss is a regression, not one.
Both arms use the same objective.

## The adapter

World-model features → LayerNorm → MLP → **one** conditioning token, beside the
fixed task instruction. No context-token block. Only the input width differs
between arms, and `LatentAdapter.parameter_report()` states the resulting
capacity difference.

## Stages

```
demonstrations ──► 1A world-model pretraining (per arm, separate checkpoints)
                   └─► 1B adapter + SmolVLA action expert, frozen world model
                       └─► 2 online model-based RL
```

## Running it

Pin the checkpoint first. The requested revision is resolved to an immutable
commit **before** loading, and that hash is what gets loaded; a branch name is
not a pin. Every caller — training, evaluation, tests — goes through
`sim_vla/models/pretrained.py:load_policy`, so a run cannot train against one
revision and evaluate against another.

```bash
python -m sim_vla.download_pretrained --out data/pretrained
```

By default the **weights go to the Hugging Face cache** and `--out` receives
only the report. Pass `--snapshot` to download the files into `--out` and load
from there instead. Then paste the printed hash into `configs/base.yaml` as
`actor.revision`.

Tests, staged, stopping at the first failure:

```bash
bash sim_vla/run_tests.sh
```

Three outcomes per stage, and only the first is success:

- **passed** — every required module ran and nothing failed
- **INCOMPLETE** (exit 2) — a required module skipped, or nothing ran
- **FAILED** (exit 1) — something failed

Every `sim_vla.tests` module is required, so a stage cannot read green on the
strength of a lightweight test while its integration skipped. A checkpoint that
loads but whose interface has moved **fails**; only an unreachable checkpoint
skips.

Training is separate and the suite never invokes it:

```bash
python -m sim_vla.training.pretrain_world_model --task pickcube --experiment graph --steps 50000
```

## Layout

```
configs/      base + task + experiment yaml; the switch lives here
data/         collection, audit, loader, sequences, normalization, replay
models/       world_model, latent_adapter, smolvla_actor, pretrained,
              flow_sampler, critics, model_config
envs/         the online env, matched to the dataset's recorded contract
training/     pretrain, imitation, imagination, actor_critic, progress, online
evaluation/   world-model diagnostics, policy rollouts, arm comparison
runtime/      checkpoints that carry the decisions their weights depend on
tests/        stages 3-9; stages 1-2 are in the repo's top-level tests/
```
