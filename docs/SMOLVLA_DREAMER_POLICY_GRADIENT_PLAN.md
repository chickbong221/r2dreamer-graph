# SmolVLA Stage 2: Dreamer-style policy gradient with an imitation anchor

Implementation handoff for Claude. This is a plan, not an implemented feature.

## 1. Objective and scope

Keep the existing pretrained SmolVLA actor and latent conditioning adapter. Replace the optional Stage 2 pathwise actor update with a Dreamer-style score-function update: generate imagined experience without retaining its autograd graph, calculate detached advantages, and differentiate the log probability of the sampled behavior. Add a separate flow-matching imitation loss on demonstrations.

**Do not implement PPO.** Do not introduce importance ratios, ratio clipping, an old-policy network, multiple optimization epochs over the same imagined batch, or a PPO replay buffer. Use one actor optimizer step per newly generated imagined batch.

Preserve the current `pathwise` objective as the existing default and comparison baseline. Add an explicit `flow_reinforce` objective. Both the `dreamer` and `graph_progress` experiment arms must use the same selected actor objective and hyperparameters.

This is a Dreamer-style update for a stochastic flow policy, not an exact reproduction of original DreamerV3 or RL-100. The important shared property is a detached advantage multiplying a trainable log probability. The flow actor requires a different probability representation than Dreamer's action-distribution MLP.

Do not change Stage 1A/1B training, replace SmolVLA with an MLP, change the graph representation, implement action chunk execution beyond `execute=1`, or submit training jobs as part of implementation.

## 2. Current code to build on

Repository-relative paths below resolve inside the checked-out `r2dreamer-graph` repository; on the current workstation the root is `E:/Code/r2dreamer-graph`.

| File | Relevant current behavior |
| --- | --- |
| `dreamer.py` | Actor uses detached imagined features/actions and detached normalized advantages in a log-probability objective. |
| `networks.py` | `ReturnEMA` implements return-spread scaling with a floor of 1. |
| `sim_vla/models/flow_sampler.py` | `sample_actions` uses initial Gaussian noise followed by deterministic Euler flow steps; `flow_matching_loss` supplies Stage 1B imitation. |
| `sim_vla/models/smolvla_actor.py` | `condition` builds an adapter-conditioned VLM prefix; `expert_velocity` executes one action-expert step. |
| `sim_vla/training/imagination.py` | Selects fresh posterior start states, advances RSSM/semantic state, computes lambda returns. |
| `sim_vla/training/actor_critic.py` | Current actor loss is negative differentiable imagined return. Includes microbatching and a currently unimplemented `demo_anchor` setting. |
| `sim_vla/training/train_imitation.py` | Correct causal demo features, action targets, chunk alignment, lookahead, and masks. |
| `sim_vla/training/online.py` | World-model/progress updates, real collection, replay, actor scheduling, and checkpoints. |
| `sim_vla/training/pipeline.py` | Restores Stage 1 checkpoints and constructs online/actor-critic configs. |
| `sim_vla/runtime/checkpoint.py` | Saves modules, optimizer state, and compatibility metadata. |

Retain the existing memory controls, mixed precision, coordinate transformations, and replay train-ratio semantics. Do not undo the earlier fixes while introducing the new objective. Update comments that currently describe pathwise gradients as the only supported actor objective.

## 3. Give the flow policy a valid score-function gradient

The present deterministic Euler sampler does not expose a tractable final-action log probability. Initial Gaussian noise alone does not provide the required log probability of the actor's output. Flow-matching MSE is not a substitute for `log_prob`.

Implement a separate, explicitly named stochastic Euler sampler for `flow_reinforce`. For a first implementation, use fixed, positive Gaussian noise at every flow transition. Do not claim this is the exact ReinFlow or RL-100 noise schedule.

For a detached conditioning feature `f`, a chunk with `C` actions of task width `D`, and `K` flow steps:

```text
u_0 ~ Normal(0, I), shape [C, D]
t_k = 1 - k / K
dt = 1 / K
mu_theta(f, u_k, t_k) = u_k - dt * velocity_theta(u_k, t_k, condition_theta(f))
u_(k+1) ~ Normal(mu_theta, sigma_k^2 I),  k = 0 ... K-1
output chunk = u_K
```

Use `sigma_k = flow_noise_std / sqrt(K)` as a simple, explicit initial schedule. `flow_noise_std` is a nominal aggregate injected-noise scale in normalized action coordinates, not a guarantee of the final output variance. Require it to be strictly positive and finite for this objective, including at the final flow transition. Keep it fixed during a run initially.

This changes the policy's sampling distribution even before training. It is mathematically a valid discrete stochastic policy, but it is not guaranteed to preserve the pretrained deterministic flow's action distribution or task success. Its initial performance is an explicit validation gate, not an assumption.

The conditional log probability for a recorded transition is:

```text
ell_k = sum over ALL C * D sampled coordinates of
        -0.5 * ((u_(k+1) - mu_theta) / sigma_k)^2
        -log(sigma_k) -0.5*log(2*pi)
```

The trainable part of the full denoising-path log probability is `sum_k ell_k`. The initial `u_0` density can be omitted from the actor loss because it does not depend on actor parameters.

Requirements:

- Use the same negative Euler direction as the existing LeRobot-convention sampler.
- Keep recorded flow states, sampled noise, means used in sampling, and density arithmetic in float32. Transformer operations may use the configured autocast precision; cast their output before the probability arithmetic.
- Use the same evaluation/training module mode for collecting paths and scoring them. Disable dropout for these operations without disabling autograd during scoring. Restore prior modes afterward. Test this with the actual SmolVLA integration.
- Implement sampling and log-probability evaluation using one shared transition-mean function.
- Score raw flow transitions, before environment clipping or coordinate conversion. Environment clipping is a downstream deterministic mapping, not a Gaussian observation to score.
- Sum over sampled coordinates; do not average over dimensions or use a product of densities in ordinary probability space.
- The stochastic policy lives in task action width `D`. Internal padding used by SmolVLA is not an extra sampled random variable.
- Include the complete sampled chunk in the transition density, even though only its first action is executed. Intermediate chunk entries can affect the first action through subsequent expert attention. Do not introduce heuristic masks for unexecuted chunk rows.
- A deterministic transition with zero variance must not be scored as a Gaussian. Keep deterministic sampling solely in the existing pathwise/reference evaluation path.

The transition log probabilities belong to an augmented policy whose internal actions are denoising transitions. They are not the marginal log probability of the final robot action. A score-function gradient of the full sampled path can nevertheless optimize the expected final-action return.

## 4. Collect an ephemeral imagined-rollout record without gradients

Add a dedicated `imagine_flow_reinforce` entry point or an equally explicit branch. Retain the current `imagine` path for the existing objective.

After the world-model and progress updates, recompute posterior start states using the existing `start_states`. During collection, hold actor parameters and all world-model/progress parameters fixed. Enclose the complete imagined rollout in `torch.no_grad()` and ensure no helper re-enables gradients internally.

For each of `H` imagined environment transitions:

1. Get the current RSSM feature, including semantic state for the graph arm.
2. Build actor conditioning once and sample the stochastic flow chain.
3. Record the detached feature and the `K+1` flow states.
4. Take the final chunk's first action, apply `coords.executed`, then `coords.to_dynamics` as in current imagination.
5. Advance the RSSM once. Preserve the existing semantic-state update inside `img_step`; do not extract a scene graph or advance the semantic state a second time.

Suggested record fields:

```text
features:          [H+1, B, feature_dim]
flow_states:       [H, B, K+1, C, D]
executed_actions:  [H, B, D]
flow_times/sigmas: exact collection schedule
instruction:      exact conditioning identity
reward/cont:      [H, B], derived from successor features
returns/values/advantages/weights: populated under no_grad
```

Do not store VLM key/value caches or hidden activations for actor learning. Keep flow records on CPU if necessary and transfer only a scoring microbatch to the GPU. Store them losslessly in float32 initially. Release each no-grad prefix cache after its environment step.

This is an ephemeral imagined batch, not long-term environment replay. Discard it after one actor optimizer step. Existing demonstration/online replay still trains the world model and supplies start states; replay actions are not retrospectively treated as current-policy flow trajectories.

## 5. Detached returns, advantages, and critic update

Preserve the current timeline exactly:

```text
reward[t] = reward_head(features[t+1])
cont[t]   = continuation_head(features[t+1])
value[t]  = critic(features[t])
```

Use the current lambda-return implementation, configured discount/lambda, and slow target critic at the final bootstrap. Compute all targets and actor advantages before any critic optimizer step, then keep them fixed for the entire update.

For the progress arm, preserve its current shaped reward and beta schedule. Do not also switch to the original Dreamer's separate progress critic in this change. That would add another experimental variable. Log environment and shaping rewards separately.

Define:

```text
R[t] = lambda_return(...)[t]
A_raw[t] = R[t] - V_before_update(features[t])
S = ReturnEMA scale of the combined returns, floored at 1
A[t] = detach(A_raw[t] / S)
w[0] = 1
w[t] = product over j < t of (discount * cont[j])
```

Use `networks.ReturnEMA`, updating its statistics once per full newly collected batch, not once per microbatch. Use float32. Do not add per-microbatch advantage normalization or separately normalize progress and environment contributions. Checkpoint its running statistics.

The `w[t]` convention is an explicit survival/discount weight for the outgoing action at `features[t]`. Verify its indexing against the successor-indexed continuation head. For a terminal transition, its own reward remains eligible while subsequent transitions receive zero weight.

Train the existing distributional critic on detached features and detached returns, with these fixed weights. Keep the existing slow-target update cadence and critic warm-up. During warm-up update the critic and skip the actor, including the anchor, to preserve the current policy warm-up behavior. The first implementation need not change critic architecture or add slow-value regularization.

## 6. Dreamer-style actor loss: no PPO

Use this exact objective for the new branch:

```text
L_RL = -(1 / (B*H)) * sum_(b,t) [ w[t,b] * A[t,b] * sum_k ell[t,b,k] ]
L_actor = L_RL + demo_anchor * L_FM_demo
```

All recorded inputs, advantages, and weights are detached. Only recomputed actor conditioning and transition means receive gradients. Sum over `k`; silently averaging over `K` changes the actor-loss scale relative to the imitation anchor. Keep `K` fixed during the first experiments.

For each scoring microbatch of `(environment-step, start-state, flow-step)` tuples:

1. Load detached feature `f`, `u_k`, `u_(k+1)`, time, sigma, weight, and advantage.
2. Recompute `actor.condition(f)` WITH gradients enabled.
3. Run one expert velocity evaluation on detached `u_k`.
4. Compute the transition log probability of the fixed recorded `u_(k+1)`.
5. Accumulate its globally normalized contribution to `L_RL` and backpropagate immediately.
6. Release that microbatch's computation graph.

Crucial: detaching the feature does not mean detaching the adapter or prefix. The adapter still learns through the frozen VLM's operations. Reusing the no-grad collection prefix would sever this gradient. Frozen VLM weights must remain frozen, but the recomputed prefix must participate in autograd.

Never use newly reparameterized sampled targets during scoring: both sides of each recorded transition are fixed samples for this score-function loss. Never backpropagate through RSSM transitions or from one recorded flow step into another.

Zero actor gradients once before all RL and imitation microbatches, clip gradients once after both terms, and perform exactly one actor optimizer step. Until that step, all actor parameters remain equal to those that collected the fresh imagined batch. No old-policy copy or importance correction is needed.

Initially set entropy regularization to zero. Fixed-sigma Gaussian transition entropy has no gradient with respect to the velocity mean and is not the final-action entropy used by Dreamer. Do not add a misleading entropy bonus, use sampled negative log probability as an unexplained entropy surrogate, or claim exact Dreamer entropy equivalence. Learnable exploration is a separate future experiment.

## 7. Implement the demonstration anchor without a second trainer

Refactor a small shared helper from `ImitationTrainer.loss` that prepares detached features, aligned action-chunk targets, and validity masks. Reuse it from Stage 1B and Stage 2. Do not instantiate `ImitationTrainer` inside online training: its constructor freezes the world model and creates a second actor optimizer.

Supply a separate demonstration-only batch to the actor update through `OnlineTrainer`. It must use the current world model to encode features under no-grad, with the existing normalization and coordinate conversion. Supervise `action_target` and its future chunk, not the previous action consumed by the posterior.

Preserve `loss_mask`, `action_valid`, lookahead, episode boundaries, and chunk masks. Sample a bounded number of eligible anchor rows before building the expensive actor conditioning. Do not use arbitrary failed online trajectories as imitation targets.

Use the existing flow-matching convention:

```text
z ~ Normal(0, I), tau ~ Uniform(0, 1)
x_tau = tau*z + (1-tau)*demo_action_chunk
velocity_target = z - demo_action_chunk
L_FM_demo = masked mean squared velocity error
```

This is one expert evaluation per selected demonstration row. Use a separately capped row budget and anchor microbatch size, independent of the world-model replay batch size.

For variable valid chunk lengths, normalize the accumulated anchor loss by the total number of valid scalar targets in the selected anchor batch, not by microbatch count or a mean of unequal masked means. Its contribution must be independent of how that batch is partitioned.

A nonzero anchor weight requires valid demonstration data. If a sampled batch has no eligible rows, retry a bounded number of times and then report/fail explicitly rather than silently disabling the requested anchor for the whole run. A zero anchor weight must not trigger demo work.

## 8. Integrate collection, configuration, and experiment arms

For `flow_reinforce`, `LatentPolicy` must use the same stochastic flow sampler and noise schedule during environment collection. Otherwise the real behavior distribution and the imagined policy being improved would differ unnecessarily.

Evaluate the stochastic training policy as the primary policy. An optional deterministic Euler evaluation may be reported separately, but do not silently evaluate it as if it were the same policy. Keep `execute=1`, existing action normalization/clipping, and recurrent reset behavior.

Expose the settings through YAML, config validation, `online_configs`, resolved logs, and CLI overrides:

| Setting | First implementation choice |
| --- | --- |
| `actor_objective` | `pathwise` or `flow_reinforce`; old behavior remains default |
| `flow_noise_std` | Required positive value for `flow_reinforce`; chosen by the pre-training sampler pilot |
| `flow_noise_schedule` | Explicit versioned `constant_per_step_scaled_by_sqrt_k` |
| `actor_transition_microbatch` | Start at 16 scored flow transitions; independent of imagination batch |
| `demo_anchor` | Configurable nonnegative coefficient; no universal best value assumed |
| `anchor_rows` | Initial engineering cap: 64 eligible demonstration rows per actor update |
| `anchor_microbatch` | Initial engineering cap: 16 rows |
| `actor_lr` | Expose separately; pilot lower values before using the existing 3e-5 |
| `advantage_scale` | Dreamer return EMA, floor 1 |
| `eval_sampler` | Explicit stochastic primary; deterministic reference optional |

Keep replay `batch_size=16`, `imagination_batch=256`, imagined horizon 15, resolved checkpoint flow-step count, and world-model train ratio 64 for the intended comparison. Existing imagination microbatching can bound collection memory; the new transition microbatch bounds actor backward memory. Preserve the current precision setting, with float32 density calculations.

Do not label all of these numbers as validated training hyperparameters. In particular, noise scale, actor learning rate, and anchor weight require a short pilot. Expose `actor_every` and evaluation cadence if needed through existing config construction; do not silently hard-code new scheduling.

Add new launch scripts, leaving the current baseline scripts available:

```text
runs/sim_vla/server1/slurm_peginsertion_baseline_online_flow_reinforce.sh
runs/sim_vla/server1/slurm_peginsertion_graph_progress_online_flow_reinforce.sh
```

Resume Stage 1A/1B from the corresponding original directories:

```text
dreamer:
/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260919_140933/peginsertion/dreamer

graph_progress:
/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260919_140921/peginsertion/graph_progress
```

Check that both `world_model.pt` and `imitation.pt` exist. Start a new Stage 2 run in a new output directory. Preserve the cluster resource/environment setup; obtain credentials from the environment. Do not copy API keys. Do not submit the scripts automatically.

## 9. Checkpoint contract

Record actor objective, exact sampler/schedule version, noise scale, flow-step count, anchor coefficient, action-coordinate identity, precision, resolved hyperparameters, actor-update count, and world/environment-update counts.

Save the return EMA as an additional module. Save actor/critic/world/progress optimizer state where applicable. Existing Stage 1 checkpoints need no new stochastic-policy weights for the fixed-noise version and must remain loadable. Switching actor objectives from an existing online checkpoint is an explicit new experiment, not an indistinguishable resume.

The current pipeline restores Stage 1 checkpoints and starts online training; do not claim it supports exact Stage 2 continuation merely because `online_latest.pt` exists. If implementing online resume, make it a separately tested path that restores the required counters, normalizer, schedule state, replay/sampler state and RNG state, and document any environment-state limitation. Discard unfinished ephemeral imagined batches at restart; resume from an update boundary.

## 10. Required tests and acceptance gates

Implement focused tests alongside the existing `sim_vla/tests` suite.

1. **Probability correctness.** Compare transition log probabilities with `torch.distributions.Independent(Normal(...), 2)` in a toy actor. Verify summation across chunk/action coordinates, sigma handling, Euler sign, and rejection of zero/invalid sigma.
2. **Collection/scoring consistency.** With unchanged weights and deterministic module behavior, recomputed means and log probabilities match collection-time diagnostic values, including the actual mixed-precision SmolVLA path.
3. **Score-gradient direction.** For fixed Gaussian samples and fixed positive/negative advantages, verify the gradient increases/decreases the sampled transition's likelihood as expected. Add a one-step analytic toy expected-reward gradient check over many samples.
4. **Gradient boundaries.** Actor RL loss updates the adapter and intended expert parameters while producing no world-model, progress-head, or critic gradients. Recorded flow states/features have no `grad_fn`. Frozen VLM weights stay unchanged while adapter gradients pass through its recomputed prefix.
5. **One update, fresh data.** Assert exactly one actor optimizer step per imagined batch and that no stale batch is used after that step. Assert no actor step during critic warm-up.
6. **Microbatch equivalence.** On fixed recorded trajectories and fixed anchor noise/times, compare full-batch gradients with several partitions, including an uneven final group. Check both the full-path score sum and the globally masked anchor denominator.
7. **Timeline correctness.** Toy dynamics with hand-computable rewards test successor reward indexing, final bootstrap, continuation weights, lambda returns, detached advantages, and no reward credited before its action.
8. **Anchor correctness.** Same prepared demo rows and noise/times give the same loss as Stage 1B. Burn-in, terminal padding, lookahead, and target-versus-previous-action alignment are preserved. Check nonzero anchor actually contributes actor gradients.
9. **Both experiment arms.** Run small updates with graph disabled and graph/progress enabled; preserve latent shapes, semantic advancement, shaping schedule, and action-coordinate behavior.
10. **Checkpoint/config compatibility.** Old Stage 1 checkpoint initialization still works. New objective/sampler settings and EMA round-trip. Invalid objective/noise/anchor combinations fail before expensive training.
11. **Real memory smoke test.** On the server's real SmolVLA, measure synchronized CUDA peak allocated/reserved memory and time separately for world-model learning, no-grad rollout collection, critic learning, flow-score microbatches, and the anchor. Do not infer memory improvements from toy tests alone.

For memory testing, run past critic warm-up so the actor actually executes. Confirm that backward graph size is controlled by the transition microbatch, while total detached trajectory storage can still grow with horizon/batch. Do not promise a fixed memory-reduction factor or lower wall time: prefix recomputation introduces additional compute.

## 11. Pilot and experiment order

Deliver implementation in reviewable stages: sampler and probability tests; detached imagination/advantages; score-function actor and gradient tests; imitation anchor; pipeline/collection/checkpoint integration; server smoke test.

Before long fine-tuning, evaluate the restored imitation checkpoint using its original sampler and candidate stochastic samplers on matched seeds. Example exploratory noise scales are 0.01, 0.03, and 0.1 in normalized coordinates; these are pilot candidates, not proven settings. Log task success, return, action clipping, and output shifts. If all candidates seriously degrade the starting policy or produce unusably noisy gradients, revisit the stochastic sampler before proceeding.

Pilot actor learning rate and anchor coefficient using bounded runs. Judge the anchor by retention of demonstrated behavior and task success, not merely by the numerical ratio of two losses. Raw score-function loss values depend on density scale and are not a performance metric.

For a clean algorithm comparison, include:

- Existing pathwise objective without anchor: current reference.
- Pathwise objective with the same anchor: isolates the gradient-estimator change from the anchor change.
- `flow_reinforce` with anchor: proposed method.

Apply each chosen configuration consistently to both task arms. Report environment steps, actor optimizer steps, compute time, peak memory, success-once, success-at-end, environment return, demo flow loss, normalized/raw advantage statistics, flow noise, clipping frequency, and separate RL/anchor gradient diagnostics on occasional sampled updates.

Use multiple seeds for the final comparison and held-out evaluation seeds. Do not declare the method better from lower memory or a descending actor loss alone. Keep the initial test small; launch full 500,000-step runs only after correctness, initial-policy quality, and real GPU memory checks pass.

## 12. Deliverables and final report from the implementer

Deliver the two selectable actor objectives, stochastic sampler, ephemeral rollout record, genuine demonstration anchor, config/CLI integration, backward-compatible Stage 1 initialization, two new submission scripts, tests, and updated README.

Report exactly which tests ran, whether actual SmolVLA/CUDA testing ran, measured peak memory/time, any unimplemented acceptance gates, and all chosen pilot hyperparameters. Do not claim mathematical equivalence to the previous pathwise estimator, identical samples after changing noise, exact original DreamerV3 reproduction, or performance improvements without evidence.

The key acceptance condition is: **SmolVLA receives a valid log-probability gradient and a real imitation gradient, while neither gradient traverses imagined world-model transitions or an entire sequential denoising chain.**

Background: ReinFlow demonstrates score-function RL for stochastic flow policies, but the simple fixed-noise sampler above is an explicitly specified prototype rather than its complete learned-noise algorithm: https://arxiv.org/abs/2505.22094. RL-100 is useful background for denoising-policy RL but its PPO objective is intentionally excluded here: https://arxiv.org/abs/2510.14830v4.
