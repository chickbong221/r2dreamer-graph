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

The online actor objective is the **pathwise return of an executed chunk**, and
it is the only one. Both arms use it: it is the update rule, not the state.

From every eligible replay latent the actor is conditioned once and generates
one action chunk. The first `actor.execute` actions of that chunk are stepped
through the frozen world model -- exactly what the online policy does between
two replans -- and the rollout's return is

    G = V_slow(s_E)
    for t = E-1 .. 0:   G = r_t + gamma * c_t * G

the `lambda = 1` return over the executed chunk. The actor maximises `mean(G)`
pathwise: the gradient runs back through the bootstrap value, the reward and
continuation heads, every imagined transition and every Euler step of the
sampler, into the adapter and the action expert. The critic regresses `V(s_0)`
onto the same `G`, detached. The states inside a chunk are not training points
for a state-only critic, because their next actions are already committed, and
none of them is evaluated merely to form the return.

One rollout per microbatch serves both losses. There is no second rollout for
the critic, no scored flow transitions, and no online imitation term -- Stage
1B is where the policy imitates. `imagination_microbatch` bounds the memory an
update uses; the number of starts is not a setting, because every scored row
of the replay batch is one.

**The objective has no learning signal through identically zero reward and
value heads**, and it starts that way: their output layers initialise to zero,
so the imagined return does not depend on the action and the actor gradient is
exactly 0.0. `critic_warmup` is therefore load-bearing rather than a safety
margin, and during it the rollout is built with no actor graph at all --
conditioning included -- so only the critic trains.

`actor_grad_norm` and `actor_params_with_grad` are logged so that "warming up"
and "broken" can be told apart: a zero actor gradient during warm-up is
expected, and a zero one after it is a broken run. Asserting `p.grad is not
None` does not demonstrate a live gradient -- the buffers are populated with
zeros in that state.

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
entropy bonus. SmolVLA's *deterministic* flow sampler exposes no such action
likelihood, which is why the imagined return is differentiated through the
sampler and the world-model transitions instead; detaching that rollout would
remove its learning signal.

What differs from that loop:

| | `dreamer.py` | sim_vla |
| --- | --- | --- |
| actor | distribution, `log pi * advantage` | flow policy, pathwise `mean(G)` |
| rollout | `imag_horizon` single actions | one chunk, `actor.execute` of its actions |
| return | lambda-return, lambda from the config | `lambda = 1` over the executed chunk |
| starts | replay positions, subsampled | every *eligible* replay position |

The demonstration mixture for the world model, the critic warm-up, the
separate AdamW optimizers and the progress-shaping design are unchanged.
Shaping stays potential-based and is `gamma * cont[t] * phi(s_(t+1)) - phi(s_t)`
per imagined transition: a transition that ends an episode must not be credited
with the potential of a state the agent never occupies. The actor and the
critic are formed from the same shaped reward, so they cannot optimise
different things.

`actor.execute` is the single setting behind all of this. Imagination executes
that many actions of one generated chunk and `LatentPolicy` replans after the
same number, so the policy being optimised is the policy collecting. It may not
exceed the checkpoint's chunk size; the shipped value is 5.

The online pipeline follows the original ManiSkill Dreamer's **64 replay
timesteps per environment step** and uses **bfloat16 autocast**. The discount
is still `1 - 1 / horizon` from the model config -- that horizon sets the
discount and has nothing to do with how long a rollout is. Fractional update
budgets carry across collections: with batch 16 and length 64, 300 environment
steps earn 18.75 updates. The ratio includes demonstration rows and burn-in
rows.

Replay batch size stays 16, and every scored row of that batch is an
imagination start -- about `16 x 65` with the shipped window. They are
imagined in **microbatches of 32**: each rollout is consumed by backward before
the next is built, gradients are weighted by group size, and one actor step and
one critic step follow all of them. A shorter final group is weighted
correctly. Nothing -- actor, critic or slow target -- moves until every
microbatch has been through backward.

The launch scripts' relevant flags:

```bash
--batch-size 16 --imagination-microbatch 32 --train-ratio 64 \
--online-precision bfloat16 --critic-warmup 150 --actor-lr 1e-5
```

If memory is tight, lower `--imagination-microbatch`; `0` imagines every start
together. `--train-ratio 0` selects the old fixed eight updates per collection
and `--online-precision float32` disables autocast. Startup logs and the run
summary record the resolved online settings, including the objective and the
executed chunk length.

`runs/sim_vla/server1/*_online.sh` restore their own `world_model.pt` and
`imitation.pt` and start a fresh online phase. They do not restore an
interrupted online replay, critic, optimizer or step counter, and
`online_latest.pt` is never read back into a run: a critic trained for a
different execution policy would not be a continuation of this experiment.

#### Removed settings

`actor_objective`, `flow_noise_std`, `flow_noise_schedule`,
`actor_transition_microbatch`, `imagination_batch`, `imag_horizon`,
`demo_anchor`, the `anchor_*` budgets, `grad_report_every`, `advantage_scale`
and `eval_sampler` went with the score-function objective and the online
anchor. A config or a launch script that still sets one is refused by name,
with what replaced it, rather than quietly running a different experiment.

#### Profiling

`--profile-online` reports per-phase wall time and CUDA peak memory --
`imagine`, `actor_backward`, `critic_backward`, `step` -- for each
actor-critic update. GPU timing requires a synchronize per phase boundary, so
this is a profiling setting and not a training one. Use it to choose
`--imagination-microbatch` from measurement rather than from guesswork.

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
