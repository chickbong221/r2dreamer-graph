# Latent-conditioned SmolVLA for TD-MPC2 and SOLD

Two more backends beside the Dreamer arm, built the same way and sharing its
imitation contract. Neither backend's algorithm is replaced. Each keeps its own
world model, its own latent, its own losses, its own reward convention and its
own online model-based RL; what is added is an adapter from that latent to one
SmolVLA conditioning token, a chunk-imitation stage against a frozen world
model, and a handful of narrow hooks where the native algorithm asks its policy
for an action.

```
demonstrations ──► 1  the backend's own world-model pretraining
                   │     TD-MPC2: encoder + dynamics + reward + Q ensemble
                   │              + the auxiliary Gaussian prior, coupled
                   │     SOLD:    1a SAVi, then 1b slot dynamics + reward
                   └─► 2  adapter + SmolVLA action expert, frozen world model,
                       │  chunk-based flow matching
                       └─► 3  the backend's own model-based RL, with SmolVLA
                              supplying the actions
```

With `smolvla.enabled: false` each backend runs exactly as it did before.

## Layout

```
vendor.py        importing two upstream trees that both claim `envs`
params.py        counting what was built, without double counting
observations.py  recorded cameras into each backend's image contract
action_space.py  the one boundary between actor units and native units
chunking.py      the shared imitation contract (causality, masks, chunks)
latent_actor.py  SmolVLA behind sampling hooks, and no fake log-probability
checkpoint.py    metadata a set of weights cannot be interpreted without
configs/         tdmpc2.yaml, sold.yaml
tdmpc2/          config, data, agent, policy, stages, online, run
sold/            config, data, adapter, model, policy, stages, env, online, run
tests/           native-behaviour, integration, gradient and real-dependency tests
```

Changes inside the vendored trees are small and additive, and every one of them
is inert with no policy attached:

| file | change |
|---|---|
| `sim_vla/tdmpc2/tdmpc2.py` | `latent_policy` hook, `attach_policy`, `pi_action` at the five learned-policy call sites, `update_pi_latent` beside `update_pi_gaussian`, policy weights in `save`/`load`, device read from the config |
| `sim_vla/tdmpc2/common/scale.py` | device read from the config (was hard-coded `cuda`) |
| `sim_vla/sold/sold/train_sold.py` | `latent_policy` hook, `attach_policy`, `_actor_step`, optional log-prob/entropy through `imagine_ahead` and `compute_actor_loss`, policy branch in the actor optimizer step, `select_action` hook, `on_save_checkpoint`/`on_load_checkpoint` |

## Importing two upstream trees in one process

Both checkouts are meant to be run with their own directory as the working
directory and both import by bare top-level name. `tdmpc2` owns `common`,
`envs`, `trainer`, `tdmpc2`; `sold` owns `modeling`, `datasets`, `utils`,
`envs`, `train_sold`. `envs` is claimed by both *and* by this repository, whose
`sim_vla.models.world_model` imports the repository's `networks` and `rssm`.

`vendor.py` owns those names for the duration of a `with` block and hands them
back afterwards, caching the module objects so classes created inside a block
keep working outside it and re-entering reinstalls the same objects rather than
a second copy.

```python
from sim_vla.integrations.vendor import TDMPC2, SOLD
WorldModel = TDMPC2.get("common.world_model", "WorldModel")
Predictor  = SOLD.get("modeling.sold.prediction", "GaussianPredictor")
```

## The shared imitation contract

The rules are properties of chunked imitation, not of any one world model, and
they are the Dreamer integration's, reused rather than restated:

* At time `t`, conditioning may use observations through `o_t` only.
* Row `t` is supervised on the whole chunk `[a_t, …, a_(t+H-1)]`, taken from
  `action_target`. `action` is `a_(t-1)` — a model input, never the target.
* **Availability is not eligibility.** `loss_mask` says a row is scored;
  `action_valid` says a real action was loaded there. A row may be conditioned
  on only if both hold, and it lives on the observation axis while
  `action_valid` lives on a target axis `lookahead` rows longer — which is what
  lets the last scored row of a full interior window still be supervised on a
  complete chunk.
* Eligible rows are selected **before** the actor runs.
* Chunks are masked at the episode boundary, cumulatively: once a window runs
  out of loaded targets everything after that is padding.
* Action normalization is one object, used for imitation targets and for policy
  outputs alike.
* Conversion into a backend's native action coordinates happens at exactly one
  boundary, `ActionConverter.to_env`.
* Chunk length, executed-action count, planning horizon, imagination horizon
  and context length are five different numbers and none of them is derived
  from another.

### Action units — two systems, not three

The Dreamer arm needs a third `dynamics` system because `rssm.py` projects any
action outside the unit ball onto it, so standardized 2 and 4 arrive as the
same value. **Neither backend here has that problem and neither gets that
squashing.** TD-MPC2 concatenates the action into its MLPs and clamps plans to
`[-1, 1]`; SOLD embeds it linearly and its actor emits `tanh(mean)` clamped to
the action space. Both native spaces *are* the controller's normalised
`[-1, 1]`, so `action_normalization: identity` is the default and is honest.
`mean_std` and `range` are supported, go through `FieldScaler` (which floors the
divisor, so a joint the task never moves does not become a division by zero),
and are recorded in the checkpoint.

### Image contract

Recorded frames are `uint8 (T, H, W, 3)` per camera. They cross into each
backend as **bytes**, because both backends scale their own: TD-MPC2's encoder
starts with `ShiftAug` then `PixelPreprocess` (`/255 - 0.5`), and `train_sold`
divides by 255 itself. One function resamples, for stored windows and live
observations alike, and it never writes into its input.

| | cameras | layout | size |
|---|---|---|---|
| TD-MPC2 | all recorded, stacked on channels | `(T, 3·cams, H, W)` | 64 or 128 (`layers.conv` asserts it) |
| SOLD | one (SAVi is a single-view 3-channel autoencoder) | `(T, 3, H, W)` | the SAVi encoder's own grid, 64×64 |

## TD-MPC2

### The five policy sites

`tdmpc2.py` asked `model.pi` for an action in five places. They are now named,
and each can be served by SmolVLA or fall back to the Gaussian prior:

| site | what it is |
|---|---|
| `act` | the executed action when `mpc=false`. With MPC on — the default, and what Stage 3 keeps — the planner is the final selector and this is not reached. |
| `plan_proposals` | the `num_pi_trajs` trajectories that seed MPPI |
| `estimate_value` | the terminal bootstrap `Q(z_H, π(z_H))` inside the planner |
| `td_target` | the Q-learning bootstrap |
| `update_pi` | policy optimization |

**Trajectory generation follows the planner, not the chunk.** At each latent the
policy samples a chunk and its *first* action is the proposal; the latent is then
advanced by that action and the next proposal comes from the resulting latent.
The chunk is not laid out along the planning horizon.

### Cost

With every site served and the shipped planning settings, `estimate_value`
alone asks for `num_envs · num_samples · iterations` flow samples **per
environment step** — about 10⁵ for 32 envs. The run prints this at startup:

```
[stage3] flow-sampler cost per environment step: 106,368 sampled chunks,
         1,063,680 denoising passes ({'plan_proposals': 2304, ...})
```

`smolvla.sites` drops a site back to the Gaussian prior. Doing so is a real
deviation, is recorded in the checkpoint, and turns the prior's training back on
(`needs_gaussian`) — a site bootstrapping off a policy frozen at the end of
pretraining is a stale target, not a fixed one. `smolvla.pi_trajs` and
`smolvla.proposal_flow_steps` are the two cost knobs that do not change which
policy is used; `pi_trajs` changes a *planner* setting and is reported as such.

### The Gaussian prior after handoff

Stage 1 is upstream's coupled update, and the prior is not optional there: the Q
targets are `r + γ·Q_target(z', π(z'))`, so it has to be learning at the same
time or the targets bootstrap off a random policy. What trains in Stage 1:

| module | optimizer |
|---|---|
| `_encoder` | `optim`, at `lr · enc_lr_scale` |
| `_dynamics`, `_reward`, `_Qs` | `optim` |
| `_pi` (Gaussian) | `pi_optim`, via `update_pi` |
| `_target_Qs` | none — Polyak, `soft_update_target_Q` |

After handoff, with SmolVLA serving all five sites, the prior is **dormant**: no
gradient, no role in any target, retained so the world model's state dict keeps
its native shape and so `smolvla.enabled: false` restores the upstream agent
exactly. If any site falls back to it, it keeps training.

### Commands

```bash
# what would be built, measured
python -m sim_vla.integrations.tdmpc2.run params --task pickcube

# 1: upstream's coupled update, on the demonstrations
python -m sim_vla.integrations.tdmpc2.run stage1 --task pickcube \
    --steps 50000 --save --out runs/tdmpc2_smolvla

# 2: adapter + action expert, frozen world model
python -m sim_vla.integrations.tdmpc2.run stage2 --task pickcube \
    --steps 20000 --save --out runs/tdmpc2_smolvla \
    --world-checkpoint runs/tdmpc2_smolvla/tdmpc2_world_model.pt

# 3: upstream's online TD-MPC2, with SmolVLA as the policy
python -m sim_vla.integrations.tdmpc2.run stage3 --task pickcube \
    --steps 1000000 \
    --imitation-checkpoint runs/tdmpc2_smolvla/tdmpc2_imitation.pt

# all three in one process, objects handed on in memory
python -m sim_vla.integrations.tdmpc2.run pipeline --task pickcube \
    --world-steps 50000 --imitation-steps 20000 --online-steps 1000000 \
    --save --out runs/tdmpc2_smolvla

# the native baseline, same data, same stages
python -m sim_vla.integrations.tdmpc2.run pipeline --task pickcube --no-smolvla
```

`--set a.b=c` overrides any config key, so a sweep needs no new file:

```bash
--set world_model.model_size=19 --set 'smolvla.sites=[plan_proposals,td_target,update_pi]'
```

## SOLD

### The two actor sites

| site | what it is |
|---|---|
| `imagine_ahead` | one action per imagined step, on the detached slot context. The imagined return is then differentiated *through* it — `actor_gradients: dynamics`. |
| `select_action` | one action per environment step, on the episode's slot history |

### The slot-history adapter

The conditioning feature for a slot world model is a set of slots per frame over
a history of frames. Turning that into one token is exactly what SOLD's own
actor does, so the adapter **is** `modeling.sold.prediction.Predictor` —
upstream's class, its ALiBi causal mask, its cls token — with the output width
set to SmolVLA's conditioning width. No new dynamics module, no second encoder,
no extra modality.

**The context is bounded, and this is an intentional difference.** `AlibiMask`
builds `mask[h, i, j] = slope_h · j` under a causal triangle, and the mask is
sliced from the bottom-right corner, which subtracts a per-row constant that
softmax ignores. The bias a query puts on a key is therefore `slope · j`
*absolute within the window* — so the attention profile depends on how long the
window is. Upstream lives with that: `imagine_ahead` calls the actor on a
context growing from `num_context` to `num_context + imagination_horizon`, while
`select_action` calls it on a history growing to the whole episode. A policy
pretrained elsewhere and conditioned through one token cannot absorb that, so
the adapter reads the last `context` frames everywhere — imitation, imagination
and inference — and the three see the same profile over the same slots.

It does not truncate slot *identity*: SAVi's recurrence carries the whole
episode into each slot; what is bounded is how many frames the conditioning head
attends over. `context` defaults to `num_context`, the largest value that is the
same at every imagined step, and imitation uses `burn_in = context - 1` so every
eligible row has a full history.

### The target dataset, not the shipped checkpoints

`sold/checkpoints/` holds SAVi and SOLD models for `reach_red` and `push_red` —
multi-object-fetch scenes with a different robot, action space and camera.
Nothing here loads them. Stage 1a pretrains SAVi on the target demonstrations.

### Teacher forcing

`dynamics_predictor.teacher_forcing` is off upstream and stays off. In its
batched form it feeds the ground-truth future slots into the rollout, so the
dynamics loss would be computed against a model that saw the frames it was asked
to predict while imagination at run time has none. Turning it on is refused with
that explanation.

### Commands

```bash
python -m sim_vla.integrations.sold.run params --task pickcube

# 1a: SAVi on the target demonstrations
python -m sim_vla.integrations.sold.run stage1a --task pickcube \
    --steps 20000 --save --out runs/sold_smolvla

# 1b: slot dynamics and the reward head, on top of it
python -m sim_vla.integrations.sold.run stage1b --task pickcube \
    --steps 50000 --save --out runs/sold_smolvla \
    --world-checkpoint runs/sold_smolvla/sold_autoencoder.pt

# 2: adapter + action expert, frozen world model
python -m sim_vla.integrations.sold.run stage2 --task pickcube \
    --steps 20000 --save --out runs/sold_smolvla \
    --world-checkpoint runs/sold_smolvla/sold_world_model.pt

# 3: upstream's online SOLD, with SmolVLA as the actor
python -m sim_vla.integrations.sold.run stage3 --task pickcube \
    --steps 1000000 \
    --imitation-checkpoint runs/sold_smolvla/sold_imitation.pt

# all four in one process
python -m sim_vla.integrations.sold.run pipeline --task pickcube \
    --autoencoder-steps 20000 --world-steps 50000 \
    --imitation-steps 20000 --online-steps 1000000 \
    --save --out runs/sold_smolvla

python -m sim_vla.integrations.sold.run pipeline --task pickcube --no-smolvla
```

## Intentional deviations from the native algorithms

These are differences, not equivalences. Nothing below claims the algorithm is
mathematically unchanged.

**Both backends**

1. **No entropy bonus, and no replacement.** A flow policy has no analytic
   entropy. TD-MPC2's `entropy_coef · log π` term and SOLD's
   `-actor_entropy_loss_weight · mean(discounts · entropy)` are dropped in
   SmolVLA mode. No substitute regularizer is added.
2. **No log-probability.** `LatentActor.log_prob`, `.entropy` and `.rsample`
   raise. The flow-matching loss is a regression onto a velocity and is never
   substituted for a density.
3. **The return term is preserved.** TD-MPC2 keeps the scaled Q objective with
   its `RunningScale` and `rho` weighting; SOLD keeps the normalised
   lambda-return advantage under dynamics gradients with the same discounting.
4. **A new module gets a new learning rate.** `smolvla.online_lr` defaults to
   `1e-5` for the adapter and action expert; the native heads keep their own
   rates. The Gaussian prior keeps `cfg.lr` on the runs where it still trains.
5. **One action per chunk is executed.** `execute` must be 1. Executing more
   would make the imagined/planned policy a different policy from the executed
   one; anything else is refused rather than approximated.

**TD-MPC2 only**

6. Turning a site off in `smolvla.sites` hands it back to the Gaussian prior and
   turns the prior's training back on. Default is all five on.
7. `smolvla.pi_trajs` overrides the planner's `num_pi_trajs`. It is a cost knob,
   it changes a planning setting, and it is reported as a deviation when set.
   Default 0 = upstream's value.

**SOLD only**

8. **`actor_gradients: reinforce` is refused** with a flow policy.
   `attach_policy` raises.
9. **The adapter's slot-history context is bounded** to a fixed number of frames
   in imitation, imagination and inference, where the native actor sees a
   growing window. Rationale above.
10. The environment is built from the dataset's metadata rather than through
    `sold/envs/make_env`, which has no ManiSkill suite. The SOLD tree's own
    suites are unchanged and still reachable through upstream's entry point.

## Defects fixed along the way

These are fixes, not deviations. Each has a regression test.

1. **`sim_vla/sold/sold/utils/training.py`** — `on_load_checkpoint` called
   `replay_buffer.load_from_files`, which does not exist;
   `RingBufferDataset` defines `load_from_disk_storage`. Resuming a run with
   `save_replay_buffer: True` therefore raised `AttributeError` from inside the
   load hook, after Lightning had already restored the weights, so the failure
   named neither the buffer nor the resume. Stage 3 writes exactly those
   checkpoints. Regression: `test_sold_native.ReplayResume`.
2. **`sim_vla/tdmpc2/common/scale.py`** and `tdmpc2.py` hard-coded
   `torch.device('cuda')`, so neither the agent nor its running scale could be
   built on CPU. Both now read `cfg.device` and default to `'cuda'`, so a run
   that sets nothing is unchanged. Without this none of the TD-MPC2 tests could
   run anywhere but on a GPU box.
3. **`sim_vla/integrations/tdmpc2/config.py`** — `lr: 3e-4` in
   `sim_vla/tdmpc2/config.yaml` is a *string* under `yaml.safe_load`: YAML 1.1's
   float pattern wants a dot or a signed exponent. Upstream never notices
   because Hydra reads the file through OmegaConf, whose resolver accepts it.
   Reading it with plain PyYAML gives `cfg.lr * cfg.enc_lr_scale` →
   "can't multiply sequence by non-int of type 'float'", from a line that
   mentions neither YAML nor the learning rate. The loader here applies
   OmegaConf's float rule.

And one test-design fix, which is the earlier review's "gradient tests need
nonconstant heads" finding in a new place: `common/init.py` zeroes the last
layer of TD-MPC2's Q heads, so a freshly built agent has `dQ/da == 0`
*exactly*. The first version of the Q-objective gradient test reported no
gradient into the adapter and was right about the number and wrong about the
conclusion. `common.unzero_critic` perturbs those weights first, and the test
now fails if the conditioning path is cut.

## Parameter report

Measured, on CPU, from instantiated models — not derived from config widths. The
guideline is ~50M for the world-model side, excluding SmolVLA's ~450M.

```bash
python -m sim_vla.integrations.tdmpc2.run params --task pickcube --json tdmpc2.json
python -m sim_vla.integrations.sold.run   params --task pickcube --json sold.json

# without the dataset or the checkpoint, for an architecture-only audit
python -m sim_vla.integrations.tdmpc2.run params --no-dataset --world-only --cameras 2
python -m sim_vla.integrations.sold.run   params --no-dataset --world-only --action-dim 8
```

The adapter rows below are measured at `token_dim = 960`, which is
`smolvla_base`'s `vlm_hidden_size`; the run reads that off the loaded
checkpoint, so `--world-only` reports the backend without it (7,800,231 for
TD-MPC2 and 12,232,017 for SOLD).

### TD-MPC2 — `model_size: 5`, `obs: rgb`, `include_state: false`, 64×64, 2 cameras, `action_dim: 8`

| component | total | trainable |
|---|---:|---:|
| encoder | 53,568 | 53,568 |
| dynamics | 795,136 | 795,136 |
| reward | 583,269 | 583,269 |
| policy prior (Gaussian) | 535,568 | 535,568 |
| critic (Q ensemble, `num_q: 5`) | 2,916,345 | 2,916,345 |
| critic target (deep copy) | 2,916,345 | 0 |
| adapter (512 → 1024 → 1024 → 960) | 2,564,032 | 2,564,032 |
| **world-model side, distinct** | **10,364,263 (10.36M)** | **7,447,918 (7.45M)** |
| SmolVLA (`lerobot/smolvla_base`) | ~450M, excluded | adapter + action expert |

`true_latent_dim = 512`. With `include_state: true` and a 25-wide proprioception
vector it is 576 and the total is 10,890,663 (10.89M) / 7,810,478 trainable.

### SOLD — `sold/configs/train_sold.yaml`, 64×64, 7 slots × 128, `action_dim: 8`, `max_episode_steps: 150`

| component | total | trainable |
|---|---:|---:|
| autoencoder — encoder | 85,824 | 85,824 |
| autoencoder — decoder | 181,124 | 181,124 |
| autoencoder — corrector | 165,888 | 165,888 |
| autoencoder — predictor | 131,968 | 131,968 |
| autoencoder — initializer | 896 | 896 |
| dynamics (OCVP-Seq, 4 layers) | 4,285,056 | 4,285,056 |
| reward head | 1,875,967 | 1,875,967 |
| policy prior (Gaussian actor) | 1,753,360 | 1,753,360 |
| critic | 1,875,967 | 1,875,967 |
| critic target (deep copy) | 1,875,967 | 1,875,967 |
| adapter (7×128 → 256, 3 layers → 960, context 3) | 2,237,632 | 2,237,632 |
| **world-model side, distinct** | **14,469,649 (14.47M)** | **14,469,649 (14.47M)** |
| SmolVLA (`lerobot/smolvla_base`) | ~450M, excluded | adapter + action expert |

### The sizing decision

**Nothing was resized.** Both backends are already far below the guideline at
their own default settings, and the instruction is to leave smaller models
alone. For reference, the measured cost of the other TD-MPC2 presets at this
observation contract is 3.95M (preset 1), 29.25M (preset 19) and 66.62M
(preset 48) — preset 48 overshoots the budget, and choosing preset 19 *in order
to* approach 50M would be picking an architecture by its parameter count, which
is the thing to avoid. `model_size: 5` is also what this project's own TD-MPC2
run script uses, so the comparison against the native baseline stays like for
like.

Counting notes: target critics are deep copies, so they are distinct tensors and
count (they are not trainable); a component's parameters are attributed to the
first component that claims them, so the catch-all `world_model(rest)` /
`autoencoder(rest)` rows read 0 unique and exist to prove nothing was missed.

## Checkpoints

A tensor file is not self-describing. Every stage checkpoint records backend,
stage, env id, whether SmolVLA was the policy, the full architecture block,
the action-normalization mode and statistics fingerprint, the dataset identity,
the pretrained revision and chunk size, and the policy settings. A `.json`
sidecar sits beside the weights so a run can be identified from a directory
listing.

Loading refuses rather than coerces: a different backend, env id, architecture,
normalization mode or statistics, dataset identity, actor revision or SmolVLA
flag. An *empty* normalization record means `none`, not "unknown" — treating the
two as the same is how weights fitted with `mean_std` once loaded into a run
with normalization off.

TD-MPC2's own `agent.save` stays additive: a native run writes exactly what it
always wrote, and a run with a policy attached writes the policy beside it.
SOLD's Lightning checkpoints carry the policy's parameters (it is a submodule)
plus its optimizer state and settings through `on_save_checkpoint`.

## Tests

```bash
bash sim_vla/integrations/run_tests.sh          # every stage
bash sim_vla/integrations/run_tests.sh 1 3      # stages 1 through 3
```

Three outcomes per stage, and only the first is success: **passed**,
**INCOMPLETE** (exit 2 — a required module skipped, so nothing was verified) and
**FAILED** (exit 1). Stages 6 and 7 are required: they are upstream's own
`SOLDModule` and the real pretrained checkpoint, dataset and simulator. A suite
that read green because lerobot or Lightning was missing would be worse than no
suite.

Three things are stubbed and nothing else: the demonstrations (countable fake
arrays, so an off-by-one in the timeline shows up as a wrong integer), the 450M
action expert (a small linear velocity field with the same signature), and
nothing else — the world models, planners, losses and trainers under test are
the real ones.

One fixture is worth knowing about. `common.unzero_critic` perturbs the last
layer of TD-MPC2's Q heads, because `common/init.py` zeroes them and a fresh
agent therefore has `dQ/da == 0` *exactly*. A gradient test run against that
reports no gradient into the adapter and is right about the number and wrong
about the conclusion — and would keep reporting zero if the conditioning path
really were cut.
