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

There are two online actor objectives, selected by `online.actor_objective`.
Both arms always use the same one — it is the estimator, not the state.

| | `pathwise` (default) | `flow_reinforce` |
| --- | --- | --- |
| gradient | differentiate the imagined return through the sampler | `advantage * d/dtheta log pi` of a recorded flow path |
| sampler | deterministic Euler | Euler + fixed Gaussian noise per transition |
| rollout graph | the whole chain is retained | none; collected under `no_grad` |
| actor memory bounded by | `imagination_microbatch` | `actor_transition_microbatch` |
| imitation anchor | unavailable | `demo_anchor`, a real flow-matching gradient |

`flow_reinforce` is **not PPO**: no importance ratio, no clipping, no
old-policy copy, no repeated epochs over one imagined batch. Exactly one actor
optimizer step per freshly collected batch, so the parameters that drew the
samples are the parameters that score them.

The deterministic sampler still has no tractable log-probability, and the
flow-matching loss is still a regression, not one. `flow_reinforce` does not
pretend otherwise — it samples a genuinely different, stochastic policy whose
per-transition Gaussian density is exact, and scores that. The injected noise
changes the policy before any training happens, so the starting policy's task
success is a validation gate rather than an assumption.

**Neither objective has a learning signal through identically zero reward and
value heads**, and both start that way. The heads' output layers initialise to
zero, so their output does not depend on the feature and neither does its
gradient. `flow_reinforce` then multiplies `log pi` by a zero advantage;
`pathwise` differentiates a return that is constant in the action. Measured on
a fresh toy model, both give an actor gradient norm of exactly 0.0.

(An earlier version of this file claimed `pathwise` was immune to this. It is
not. Inside `update()` the two differ only in that `pathwise` re-imagines
*after* the critic step and so picks up a ~1e-5 trace from it, while
`flow_reinforce` computed its advantage before that step and stays exactly
zero for one more update. Neither is a usable signal; `critic_warmup` is
load-bearing for both.)

Which is why `advantage_abs`, `advantage_std`, `rl_grad_norm`,
`anchor_grad_norm` and `anchor_grad_ratio` are logged. A zero actor gradient
during warm-up is expected; a zero one *after* warm-up is a broken run, and
the loss value alone cannot tell them apart. Note also that asserting
`p.grad is not None` does not demonstrate a live gradient — both objectives
populate `.grad` with zeros in that state.

## The adapter

World-model features → LayerNorm → MLP → **one** conditioning token, beside the
fixed task instruction. No context-token block. Only the input width differs
between arms, and `LatentAdapter.parameter_report()` states the resulting
capacity difference.

## Stages

```
demonstrations ──► 1A world-model pretraining (per arm, never shared)
                   └─► 1B adapter + SmolVLA action expert, frozen world model
                       └─► 2 online model-based RL
```

### The action timeline

A window's row `t` holds the observation `o_t`. What sits beside it:

| array | is | read by |
|---|---|---|
| `action` | `a_(t-1)`, executed before arriving at `o_t` | `RSSM.obs_step` |
| `action_target` | `a_t`, executed *at* `o_t` | the actor's flow loss |
| `reward` | `r_(t-1)`, earned arriving at `o_t` | the reward head |

Because the reward head predicts the reward that *arrived*, the reward earned by
imagined transition `t` is the **successor's**: `heads["reward"][1:]`, and the
same shift for continuation. Reading `[:-1]` takes a reward no imagined action
caused and drops the one the last action earned.

Two masks, because they answer two questions. `action_valid` is *availability* —
a real action was loaded at this row. `loss_mask` is *eligibility* — this row is
scored and may be conditioned on. The final row of every window has no action of
its own unless action-only lookahead supplied one.

A window has **two axes**. Observation-axis arrays have `burn_in + length + 1`
rows; `action_target` and `action_valid` live on a *target axis* that is
`lookahead` rows longer. That is what lets the last eligible row of a full
interior window still be supervised on a whole chunk — a full interior window
has no padding, so squeezing the lookahead into the observation axis left one
slot no matter how long the chunk was. The lookahead is `chunk_size`, not
`chunk_size - 1`: the last eligible row is the window's final observation, and
the action taken *there* is already the first lookahead action.

### Action units

| coordinates | where |
|---|---|
| **raw** | the dataset, the replay, and `env.step` |
| **normalized** | the actor: flow targets, sampled actions |
| **dynamics** | `RSSM.obs_step` / `img_step` only |

`rssm.py` projects any action outside the unit ball onto it, so standardized 2
and 4 would reach the dynamics as the same value. `rssm.py` is left untouched;
`sim_vla/models/action_space.py` maps normalized to dynamics with
`tanh(x / 3)` instead, which is injective, differentiable and always inside the
ball — so the projection is the identity, as it is for the original Dreamer
pipeline's environment-unit actions.

Clipping to the controller's bounds happens once, in the actor's coordinates,
inside the policy — so the command sent to the environment is the one fed back
as the next `a_(t-1)`. It is straight-through: clipped forward, identity
backward, because a saturated dimension is the one the actor most needs pushing
back from.

### Stage handoff

The three stages run in **one process** and pass Python objects, not files:
Stage 1B trains against the very world model Stage 1A just produced, and
Stage 2 continues with that model and that actor. **Checkpoints are off by
default** — nothing is written and nothing is reloaded. `--save-checkpoints`
turns writing on for a run long enough that losing it would matter; it changes
what is persisted, never what is trained.

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

Training is separate and the suite never invokes it. All three stages, one
process, nothing written:

```bash
python -m sim_vla.training.pipeline --task pickcube --experiment graph --world-steps 50000 --imitation-steps 20000 --online-steps 200000
```

Add `--save-checkpoints` to keep each stage's weights under `--out`. A single
stage can still be run on its own, but with checkpoints off it trains a model
and then drops it, which it says on startup:

```bash
python -m sim_vla.training.pretrain_world_model --task pickcube --experiment graph --steps 50000 --save-checkpoints
```

### Online training and the original Dreamer loop

The original `dreamer.py` imagines with frozen modules, detaches the rollout,
and trains a distribution actor with `log_prob(action) * advantage` and an
entropy bonus. SmolVLA's *deterministic* flow sampler does not expose that
action likelihood, which is why `pathwise` differentiates the imagined return
through the sampler and world-model transitions instead; detaching that
rollout would remove its learning signal.

`flow_reinforce` recovers the original shape by changing the policy rather
than the loss. Adding fixed Gaussian noise to every Euler transition makes
each one a Gaussian with an exact log density, so the rollout *can* be
detached and the actor trained with a detached advantage times a trainable log
probability — the Dreamer form, on the denoising path rather than on a single
action distribution. The advantage is normalized once by the same
`networks.ReturnEMA` spread (floored at 1) that `dreamer.py` uses.

Entropy regularization stays at zero and is not a placeholder. With a fixed
sigma, the Gaussian transition entropy does not depend on the velocity mean,
so an entropy bonus here would have exactly zero gradient, and a sampled
negative log probability is not Dreamer's action entropy either. Learnable
exploration noise is a separate experiment.

The demo mixture, critic warm-up, separate AdamW optimizers, and VLA
progress-shaping design are retained under both objectives. Progress shaping
is unchanged: potential-based shaping is advantage-estimator-agnostic, and
this change does not switch to the original Dreamer's separate progress
critic.

The online pipeline now follows the original ManiSkill Dreamer's **64 replay
timesteps per environment step** and uses **bfloat16 autocast**. Horizon,
discount (`1 - 1 / horizon`), and lambda come from the original model config;
the default imagined horizon is 15. Collection still happens two episodes at
a time. Fractional update budgets carry across collections: with batch 16 and
length 64, 300 environment steps earn 18.75 updates, instead of the old fixed
eight. The ratio includes demonstration rows and burn-in rows. This increases
training work per environment step; it is not a throughput optimization.

Replay batch size stays 16. The pipeline selects up to 256 valid imagination
starts and processes them in **microbatches of 16**. Each rollout is consumed
by backward before the next is built, with gradients weighted by group size
and accumulated before one optimizer step. A shorter final group is weighted
correctly. This keeps the same mean actor objective and total start count;
random draws and numerical results need not match a single large batch.

Both `runs/sim_vla/server1/*_online.sh` scripts restore their own
`world_model.pt` and `imitation.pt`, then start a fresh online phase. They do
not restore an interrupted online replay, critic, optimizer, or step counter.
Deploy the updated Python files with these scripts. Their relevant flags are:

```bash
--batch-size 16 --imagination-batch 256 --imagination-microbatch 16 \
--imag-horizon 15 --train-ratio 64 --online-precision bfloat16
```

If memory is still tight, lower `--imagination-microbatch` to 8 or 4 while
keeping replay batch and total imagination starts unchanged. The YAML settings
are under `online`. `--train-ratio 0` selects the old fixed eight updates per
collection, `--online-precision float32` disables autocast, and
`--imagination-microbatch 0` processes all starts together. Startup logs and
the run summary record the resolved online settings.

### Running `flow_reinforce`

`runs/sim_vla/server1/*_online_flow_reinforce.sh` resume the same Stage 1
checkpoints into a separate output directory. Their flags:

```bash
--actor-objective flow_reinforce --flow-noise-std 0.03 \
--actor-transition-microbatch 16 --demo-anchor 1.0 \
--anchor-rows 64 --anchor-microbatch 16 --actor-lr 1e-5 \
--eval-sampler stochastic
```

**None of those four values is validated.** Noise scale, actor learning rate
and anchor weight all need the pilot in
`docs/SMOLVLA_DREAMER_POLICY_GRADIENT_PLAN.md` §11 first: evaluate the restored
Stage 1B policy under the deterministic sampler and under noise scales 0.01 /
0.03 / 0.1 on matched seeds before committing to a long run. A descending
actor loss is not evidence — a score-function loss's magnitude depends on the
density scale and is not a performance metric.

`--flow-noise-std` is refused unless the objective is `flow_reinforce`, and
`flow_reinforce` is refused without it; a nonzero `--demo-anchor` without a
demonstration sampler is refused too, rather than silently ignored. Failures
surface at config validation, before Stage 1 is restored.

Memory moves from the actor's backward graph to detached trajectory storage:
the backward is bounded by `--actor-transition-microbatch`, while the recorded
rollout still grows with horizon × imagination batch × flow steps. Prefix
recomputation during scoring costs additional compute, so no wall-time
improvement is claimed.

#### The anchor's two budgets

`--anchor-windows` is how many demonstration *windows* are drawn and encoded
(one world-model forward each, over the whole window); `--anchor-rows` is how
many eligible *positions* survive to be conditioned (one actor forward each).
One window yields many eligible rows, so these must not be tied together.
`--anchor-window-microbatch` bounds the encoding itself; windows are grouped
along the window axis only, never sliced in time, because a window cut in time
would give row `t` a posterior that never consumed rows `0..t`.

**Do not assume `demo_anchor=1.0` is balanced.** The RL and anchor gradients
can differ by many orders of magnitude — in one toy measurement the RL term
was 1.3e8 and the anchor 6.4, which in float32 means the anchor vanished
entirely when the two were summed, while still being applied and still being
nonzero. `anchor_grad_ratio` exists to make that visible; check it before
trusting an anchor weight.

#### Profiling

`--profile-online` reports per-phase wall time and CUDA peak memory —
`collect`, `targets`, `critic`, `score`, `anchor` — for each actor update.
GPU timing requires a synchronize per phase boundary, so this is a profiling
setting and not a training one. Use it to decide
`--actor-transition-microbatch` from measurement rather than from guesswork;
16 is a starting point chosen for safety, not for speed.

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
tests/        stages 3-12; stages 1-2 are in the repo's top-level tests/
```
