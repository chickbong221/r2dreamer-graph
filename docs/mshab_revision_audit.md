# MS-HAB Pick: agreed plan and remaining work

Restored from the completed audit retained in this conversation. This is a
concise reconstruction, not a byte-for-byte recovery of the deleted document.
No new code audit, server run, collection, or mining was performed to restore it.
Treat the findings below as the state at that audit, not verification of any
subsequent changes made elsewhere.

## Scope and comparison

The audit compared the post-ManiSkill reference commit `8fd3819` with the
MS-HAB revision at `e18656e`, then checked the eight-node/FIFO changes made in
the local working tree. The FIFO changes had not been committed or validated
in a live server training run at the end of that audit.

The intended experiments remain:

- **A:** train five Pick objects on one named scene; evaluate those objects
  on that same scene.
- **B:** train `004_sugar_box` on that scene; evaluate across all 63 scenes.
- **C:** later evaluate B's checkpoint under changed lighting, without
  training a third model.
- **Transfer:** a later run using A's checkpoint.

Training scene: `v3_sc0_staging_00.scene_instance.json`.

## What follows the agreed plan

| Area | Verified implementation at the audit |
| --- | --- |
| A's objects | `002_master_chef_can`, `003_cracker_box`, `004_sugar_box`, `007_tuna_fish_can`, `024_bowl`. |
| B's configuration | One training scene; 63 evaluation environments with even scene distribution. Complete coverage also depends on the installed dataset containing the expected 63 configurations. |
| Language and target identity | Language input and the explicit target flag are retained. |
| Node features | The shared graph encoder actually consumes entity identity, bounding boxes, centroid, and target flag; these are not merely packed and ignored. |
| Per-episode membership | The runtime uses the target's `target-supporters` whitelist. Mining nine objects does not insert all nine target categories into every episode. Multiple instances of an admitted category and its admitted furniture can still appear. |
| Protected nodes | EE is row 0, active target row 1, rest site row 2. Target seeding supports presence before camera coverage; packing checks protected rows. |
| Dynamic target | `$active_target` resolves to row 1, rather than searching for a category ID that may match several instances. |
| Rest site | Its live position is derived from the robot base pose and `ee_rest_pos_wrt_base`; tolerance comes from `pick_cfg.ee_rest_thresh`. |
| Calibration | Separate EE height families and EE-site scales are implemented. Virtual sites are excluded from ordinary object-family classification. Asset completeness remains a separate requirement below. |
| Schedule | Three phases, weighted `0.35 / 0.20 / 0.45`, with seven distinct facts. Return-to-rest requires grasp and completes on grasp AND reached. No contact milestone or temporal latch. |
| Terminal rungs | The retained weighted spatial clauses passed the existing terminal-rung validator at rest tolerance `0.05`. |
| Object-object relations | A/B enable the independent switch disabling all object-object relations, in addition to `object_object_spatial: false`. Scheduled EE-target and EE-rest relations remain enabled. |
| Checkpoints | Best-only selection uses `eval/success_once`, starts at 8M steps, and saves only on eligible improving evaluations. No extra final, milestone, or cancellation checkpoint. |
| Surface approximation | V1 deliberately keeps the mined supported-object-origin proxy with the corrected object-frame/outward-normal conversion. Exact physical-surface reconstruction is deferred, not silently enabled. |

The schedule's potential reaching 1.0 is not a replacement for environment
success. The inspected MS-HAB success predicate additionally checks robot
rest, static state, and cumulative force limits. Environment `success_once`
remains the checkpoint metric; the schedule does not terminate the episode.

## Eight-node/FIFO correction completed locally

The launchers now request `n_max=8`, `e_max=168`:

- Three protected nodes: EE, target, rest site.
- Five context slots managed by first-arrival FIFO.
- Older unprotected nodes are removed when capacity is needed.
- Evicted nodes are purged from retained history and temporal buffers.
- A still-visible rejected node does not receive a new arrival timestamp every
  frame and repeatedly displace the same retained nodes.
- Selection runs before live pose refresh and relation generation.
- Context drops are reported through the existing node-drop metric.
- The packer still rejects a graph that violates its final capacity contract.

This bounded policy is scoped to protected MS-HAB Pick. The normal ManiSkill
path retains its existing strict overflow behavior.

With at most six physical object nodes, the inspected relation emission bound
is 39 edges in EE-only mode, or 129 if object-object physical/compatibility
relations are restored while object-object spatial relations stay disabled.
The existing 168-edge allocation covers both cases.

The reported overflow was five bowl instances plus six furniture nodes, the
rest site, and EE: 13 rows. Objects visible in one image were not the same set
as everything retained during an episode.

Historical correction: eviction was already absent at `8fd3819`. Its removal
predated the current MS-HAB port (`4fc088f`). The old ReLDreamer copy still had
FIFO, but it was not the same baseline revision.

## Not corrected: long-run experiment issues

### 1. A balances objects before filtering the training scene

In `envs/maniskill.py`, `balance_objects` is applied before
`select_named_build_configs`. Equal object counts over all scenes need not
remain equal after filtering to the pinned scene.

A read-only synthetic reproduction using the actual helpers produced counts
`a: 4, b: 1` after filtering an initially balanced set. This proves the missing
guarantee, not the actual imbalance of the server dataset.

**Required correction:** filter each object's plans to the requested scene,
then balance the final eligible sets. Report final per-object counts. Not
implemented in the audited changes.

### 2. A's fixed five-per-object evaluation is not enforced

The configuration comments promise 25 episodes, five per object, using the
same plans at each evaluation. The reset path does not provide a fixed,
object-stratified set of task-plan indices. The inspected upstream environment
can sample plans during reset. Keeping the scene fixed does not fix the object
mixture or initial task plans.

**Required correction:** explicitly select and reuse the intended evaluation
panel. Not implemented in the audited changes.

### 3. Local runtime assets differ from the validated server assets

The local asset check failed on:

```text
missing ['ee-structural-surface-height-offset']
```

The local assets otherwise had nine target whitelists, 17 union members,
required entity vocabulary 19, declared rest-site membership, and a compiling
schedule. Earlier server logs passed the calibration check.

**Next action:** synchronize the validated server runtime asset bundle before
using it locally. Do not replace good server assets with the older local JSON,
invent a calibration value, or bypass the check. No assets were changed by the
FIFO repair.

## Deferred or incomplete, not a reason to redesign the trial

- **Experiment C:** lighting settings are still unselected. The inspected
  adapter has no illumination override, and there is no complete eval-only
  checkpoint-loading path wired to C. Launching `train.py` with C's config
  is not a lighting evaluation. C remains work after B.
- **Metric summaries:** success and step logs exist, but there is no completed
  summary for steps to 50%, 70%, and 80% success. A final evaluation exactly at
  the matched training endpoint is not guaranteed by the current loop, which
  evaluates before stepping and may exit after crossing the budget.
- **Transfer:** deliberately postponed until A produces a checkpoint.
- **Exact physical-surface anchors:** deliberately deferred under V1. The
  mined proxy is approximate even when mining and runtime agree.

No schedule-weight change, new experiment arm, or additional checkpoint policy
was introduced by this audit.

## Performance: established facts versus open questions

The serial per-environment graph loop, privileged-state readers, node builder,
projection path, and shared graph encoder were already present at the
post-ManiSkill reference. The defaults of 126 training environments, batch
size 32, sequence length 64, and train ratio 64 also predated this port.

The new site provider has a concrete possible inefficiency: its helper converts
a batched tensor to CPU before selecting an environment row, and the provider
runs per environment. This can cause repeated full-batch transfers. It was
identified by code inspection, not isolated by a runtime profile, and remains
unfixed. A possible optimization is one conversion per vector step, without
caching stale world positions across frames or resets.

The recorded first training update did not account for the entire initial
delay. Initial evaluation and replay warm-up also occur before training logs;
logging is step-based, not a ten-second heartbeat. Eight-node FIFO bounds
subsequent work, but no live throughput improvement was measured locally.

## Verification already performed

- Full local suite: 1,215 tests run; 1,209 passed, two skipped, and four import
  errors from unavailable `gymnasium`, `omegaconf`, or `tensordict` dependencies.
  No executed test failed in that final run.
- New FIFO integration coverage exercised the real builder/packer with
  simulation I/O mocked: protected rows, overflow selection, eviction cleanup,
  stable arrivals, reset behavior, and strict non-MS-HAB overflow.
- Terminal-rung validation passed four weighted spatial clauses at `0.05`.
- Script syntax and Git whitespace checks passed.
- A live server training trial with the FIFO change remained pending.

## Practical conclusion

A short server trial is appropriate after the updated code and server assets
pass validation. It should exercise optimizer updates, graph protection, FIFO,
and finite losses; early success is not a requirement.

Do not treat that trial as long-run experimental sign-off. Correct A's final
object balancing and fixed evaluation panel before the full experiment, and
verify the actual server assets rather than relying on the stale local copy.
