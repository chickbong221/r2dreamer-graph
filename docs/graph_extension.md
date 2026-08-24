# Compact graph extension

This repository is an independent PyTorch implementation. It does not import
files from ReLDreamer. Only the observation contract and the model behavior
needed by DreamerV3 are retained.

## Observation contract

When `model.graph.enabled=true`, the environment must emit these tensors per
frame. Index zero is padding for entity and relation vocabularies.

| Key | Shape | Storage dtype |
|---|---:|---|
| `graph_node_ent` | `[8]` | `uint8` |
| `graph_node_app` | `[8,2,384]` | `float16` |
| `graph_node_bbox` | `[8,2,4]` | `float16` |
| `graph_node_target` | `[8]` | `uint8` |
| `graph_edge_src` | `[168]` | `uint8` |
| `graph_edge_dst` | `[168]` | `uint8` |
| `graph_edge_rel` | `[168]` | `uint8` |
| `graph_edge_abs` | `[168]` | `uint8` |
| `graph_edge_temp` | `[168]` | `uint8` |

The replay remains fixed-width. `compact_graph()` selects
`graph_edge_rel != 0` on the GPU and offsets endpoints into one disconnected
batch. Relation embeddings, message MLPs, aggregation, and relation decoding
therefore execute only for real edges. The eight node slots remain dense because
their cost is small and retaining them makes target and reconstruction labels
unambiguous.

## Method switch

- Graph semantic method: `model=size50M_graph_simple` (`size100M_graph_simple`
  for MS-HAB).
- Matched DreamerV3 baseline: `model=size50M` (`size100M`).

`model.graph.enabled` is the only switch that selects the method, and the two
presets differ only in setting it. There is one graph contract; the full,
graph-only and slot variants and `model.graph.state_mode` were removed, along
with `graph_simple` and `graph_only_latent` as separate switches. Graph mode
requires `model.rep_loss=dreamer`, and the constructor refuses anything else.

## The graph contract

Nodes carry entity id, target flag, per-camera boxes and a world centroid. The
graph builder constructs no DINO and reads no RGB: `graph_node_app` is absent
from the observation space, the transition, replay and the sampled batch --
not zeroed -- which removes about 12.3 KB per environment step. Boxes cost
8 x 2 x 4 float16 = 128 bytes per frame and are extracted with one lookup table
from segmentation id to node row plus two boolean projections per camera.
Patch coverage is never computed.

`graph_node_uid` stays in the model's reserved graph key set even though
nothing packs it, so a stale wrapper that still exposes it cannot feed it to
the ordinary MLP encoder and quietly train an identity-conditioned model. A
run that is handed the key raises instead.

### Pooled graph-simple

The replay contract:

| Key | Shape | Storage dtype |
|---|---:|---|
| `graph_node_ent` | `[8]` | `uint8` |
| `graph_node_bbox` | `[8,2,4]` | `float16` |
| `graph_node_target` | `[8]` | `uint8` |
| `graph_edge_*` | `[168]` | `uint8` |

Boxes arrive already normalised to `[0,1]` as `[xmin, xmax, ymin, ymax]` with
exclusive maxima, so nothing normalises them a second time. Per-camera validity
is derived on device as `xmax > xmin and ymax > ymin` and never stored. It masks
the box terms; it is not a predicted visibility objective.

The latent model changes in four ways:

- `g` is a deterministic 512-d vector, not a categorical sample. Its prior is
  `P(h_t)` and its posterior `Q(h_t, c_t)`; neither reads `g_{t-1}`, which
  already reached `h_t` through the transition.
- `g` no longer gates `z`. The `z` branch is stock DreamerV3 -- `q(z | h, o)`,
  `p(z | h)`, `stoch x discrete` rather than `hybrid_stoch x discrete`. Graph
  information reaches `z` only through `g_{t-1} -> h_t -> z_t`.
- `g` still enters the transition and the feature `[z, g, h]`, so every
  downstream head sees it. Image reconstruction reads `g` with a stopped
  gradient, so pixels cannot reshape the semantic state into a second visual
  latent; state reconstruction, reward, continuation, actor and value all keep
  normal gradients.
- `semdyn`/`semrep` are replaced by `graphdyn`/`graphrep`, an RMS-normalised
  stop-gradient predictability regularizer. Both take the same forward value
  (logged once as `graph_align_mse`); only the gradient routing and the
  weights differ. Nothing in that objective resists collapse -- the graph and
  task losses do -- so watch `graph_align_cos` against
  `graph_sem_post_var`.

The pooled token `c_t` is a masked mean over admitted nodes concatenated with
the normalised node count, then one projection: every admitted node gets the
same `1/n` coefficient, and the count keeps two sets with the same mean and
different cardinality apart. Attention pooling exists only in `full` mode. Note
what this does and does not claim: the aggregation is permutation invariant with
uniform coefficients, but a 512-d bottleneck is still lossy, and it is the node
and relation reconstruction objectives -- not the pooling -- that push `g` to
retain the whole bounded graph.

The decoder reads `g`, never the encoder's node vectors. Each node is recovered
by querying `g` with a narrow signature of that node's current box, which makes
`node`, `nodetgt`, `relabs` and `reltemp` a measure of what `g` retained. A box
is a cheap content address: it works the frame a node first appears and keeps
episode-random identity codes out of the global dynamics. It is a *content*
key, not an identity one, so two nodes with identical boxes are not separable.
That mattered little when only currently segmented objects were emitted: the end
effector was the one node that could have an empty box. Under unconditional
retention every node without pixels has an all-zero box, so more than one node
can share a query. See "Retention and the decoder query" below.

`loss/node` here averages entity cross entropy and box SmoothL1, reported
separately as `node_ent_loss` and `node_bbox_loss`. It is **not** comparable
with full mode's `loss/node`, which averages appearance, boxes and visibility.
Box error is averaged over the four coordinates rather than summed, and every
term is reduced per valid node and then per frame, so loss magnitude does not
track graph size. Box IoU is an evaluation metric, not an update-loop one.

### Progress in pooled graph-simple

`progress.enabled=true` works in both simple modes. Pooled has two
implementations of the potential, selected by `progress.mode` (`ee_target`
or `task_schedule`), and they read the same stage table so their numbers are
comparable.

**`ee_target` (default).** The potential is a scalar. `progress_head` is a
bounded head beside the environment reward head -- `phi = sigmoid(MLP(feat))`
on the same `[z, g, h]` the policy and the ordinary critic read -- and it is
trained by regression, not classification:

```
progress_model = masked_huber(phi, progress_target)
```

`progress_target` is computed from the *observed* labels in replay.
`ProgressScorer.replay_potential` reads the end-effector-to-target block --
row 0 to row 1, which packing now guarantees -- scatters the six labels
one-hot and contracts them through the same `[R, n_abs]` matrix the predicted
path uses. One-hot in, so the scalar is exactly the hard cumulative-stage sum;
the observed and predicted potentials agree by construction rather than by a
second table.

Validity is strict. The stage weights sum to one only when every relation the
scorer names is present, so a frame missing one would score low through no
fault of the robot, and regressing on that teaches the head that losing a fact
is losing progress. A frame is supervised only when every relation the
scorer names is present and none is duplicated; anything else is *masked*,
never scored zero. `contact` is not one of them: a grasped object is also
in contact, so scoring both paid twice for one event. The graph still emits
it and the scorer ignores relations it does not name, so the edge neither
earns nor invalidates. The schedule miner drops a `contact` milestone on a
pair that already has `grasp`, `support` or `contain`.
`train/progress/valid_fraction` is that mask, and a low value is a persistence
or relation bug rather than a hard task.

Imagination reads the frozen head directly: `phi = progress_head(imag_feat)`.
The reward is the potential *difference*,
`F_t = gamma * c_{t+1} * phi_{t+1} - phi_t`, built after `imag_cont` is
predicted so it uses the same discount and continuation as `_lambda_return`.
Index 0 is zero because the return consumes `reward[:, 1:]`. Nothing is
clipped and negatives are kept: a floor at zero would pay to undo and redo
the same progress. The progress critic input is `imag_feat` and
nothing else -- the potential is already a function of it, so appending a
second view would hand the critic a shortcut the actor does not have. Note the
consequence: the head is fitted on posterior features and applied to imagined
ones, exactly as the reward head is, which makes `graph_align_cos` and
`graph_align_mse` load-bearing for progress quality.

Progress never runs inside a loop: after the rollout one
pass covers all `B x H` states at once.

### The retained target

Both sources depend on the target existing to have relations with. Retention is
now unconditional -- every whitelisted object a camera has seen stays a vertex
until reset -- so the target is no longer a special case for *existence*. It
remains one for *eligibility*: under `visibility_policy: projected_camera` an
ordinary node outside both frustums emits no relations, while the protected
target keeps its end-effector facts and pairs with any in-frame object. While
retained it keeps row 1 and its entity id, its bounding boxes go to zero (the
only per-camera visibility signal the packed observation carries), its centroid
stays live, and all six end-effector relations keep being recomputed -- the four geometric ones from the current end-effector pose
against its centroid, `contact` and `grasp` from live simulator queries through
`state.active_obj`. So an occluded target's facts still change as the robot
moves, and supervision is continuous after first observation rather than
punched full of holes.

Its pose is re-read from the simulator every frame, retained or not. A grasped
object rides with the gripper, and a frozen centroid would report it still lying
where it was picked up while `grasp` reads `holds` on the same frame -- the two
would contradict each other and the ladder reads both. The old snapshot froze
the pose while the target was ungrasped *and* invisible; nothing freezes now.

### Retention and the decoder query

The decoder addresses a node by its geometry, so under retention that signature
has to include the centroid. Every node without pixels has an all-zero box and
all-zero per-camera visibility bits; a box-only query would be identical for all
of them, and the decoder would be asked for a different entity id and a different
relation set for each from one input. `SimpleGraphDecoder` therefore queries on
`bbox_feature` concatenated with `centroid_feature`, on the same fixed bounds the
encoder uses.

`graph_node_centroid` `[N, 3]` carries the world-frame position that survives
the boxes going dark, normalised on fixed bounds
(`graph.centroid_origin`, `graph.centroid_scale`) rather than batch statistics,
so the same object in the same place encodes identically in every episode.

Packing pins row 0 to the end effector and row 1 to the target; the vertex
registry cannot supply this, because it hands the target whichever index was
free when it was first admitted. Row 1 stays padding until the target is first
observed. Reserving it costs one object row in a frame with more visible
whitelisted objects than rows, which is counted, not swallowed --
`log_graph_node_drops`. A persistently nonzero value means `n_max` is too small
for the scene, not that the reservation is wrong.

### Schedule and metrics

`beta` is the only scheduled quantity. It is zero until
`progress.beta_warmup_start` environment steps, linear to `progress.beta` by
`progress.beta_warmup_end`, and constant after; the progress head and
`progress_value` run from step 0 throughout. The critic reads a detached input,
so training it early moves neither the world model nor the actor. The head does
update the world-model latent, deliberately -- it is an auxiliary prediction
task like the reward head, with its own loss scale if that needs turning down.
Turning beta on together with a cold critic would instead let an unfitted
critic steer, and an immature world model can hallucinate progress under
imagined actions long after the real arm is still.

Five primary metrics, each answering a question the others cannot:

| Metric | Reads wrong as |
|---|---|
| `progress/valid_fraction` | target persistence or six-relation bug |
| `progress/target_std` | behaviour produces near-constant progress |
| `progress/head_mae` | world-model progress prediction not ready |
| `progress/critic_mae` | progress critic not ready |
| `progress/influence` | normalisation or beta problem |

`influence` is `beta * E|A_progress| / E|A_env|`: roughly 5-20% at the plateau,
under 1% means beta is too small to matter, well over 25% means progress is
doing the steering. `train/progress_beta` and
`train/progress_potential_horizon_std` sit beside them -- influence cannot be
read without knowing which beta produced it, and horizon_std separates "beta is
too small" from "acting does not change predicted progress".

The two critics are never mixed. The environment critic learns the
environment return, the progress critic learns the shaping return, and the
actor combines their **raw** advantages before a single normalisation:

```
A_actor = (A_env_raw + beta * A_progress_raw) / EMA_IQR(G_env + beta * G_progress)
```

Normalising each advantage by its own return spread destroyed their relative
magnitude, so beta stopped meaning "this fraction of the task advantage" and
a nearly flat potential was amplified to task scale. Lambda return is linear
in `(reward, value, bootstrap)`, so the raw combination equals forming one
combined return for the actor while each critic keeps its own target. At
`beta = 0` the actor loss and the EMA update are identical to the baseline.
Read `progress/influence_raw` for the weight beta actually buys; it is not
comparable with the older separately-normalised `progress/influence`.

The clean paper comparison holds every loss fixed and varies only the schedule:
`beta: 0.0` throughout for the control, the warm-up for the treatment.

`episode/score` and `eval/score` stay environment-only: the shaping reward
exists only inside imagined rollouts and never reaches a transition, the
replay reward, or the reward head.

No checkpoint is written or resumed. Corrected relation labels keep their ids
and change geometric meaning, so mixing old and new transitions would train
one label to mean two things.

### Scoped spatial calibration

`planar-distance` and `height-offset` stay two relations in the vocabulary and
are calibrated separately per endpoint scope. The mined asset carries eight
keys -- `ee-object-*`, `object-object-*`, and their `-change` forms -- and each
spatial edge records which one labelled it in `Edge.bin_key`. `TemporalBuffer`
reads that key rather than deriving one from the relation name, so the change
bins split with the absolute bins.

Object-object geometry is measured between **mined surface anchors**, not link
origins: a table's origin sits ~0.9m below its own top, which reported a bin
resting on it as `far-above`. Anchors are pair-specific -- a table carrying two
things has one component per partner -- and keep describing the pair after
physical support ends, so a lift changes height without changing what height
means. Spherical objects store a radius instead of a local bottom point, so an
unmoved but spinning ball keeps its relations. Pairs with no mined anchor fall
back to origins.

The miner measures the same way: the collector ships a reservoir of raw object
pose pairs at `t` and `t - K`, and the asset builder reprojects them through
the anchors it just mined. Calibrating on origins while labelling on surfaces
is what made a `level` band span +/-22cm.

A pair neither endpoint can move is dropped entirely -- no spatial, physical
or compatibility edges, and no pose samples. PlaceSphere's bin and table are
both kinematic, so PhysX solves no contact between them, no anchor is ever
mined, and the pair would report link origins every frame while its fixed
~0.9m offset set the height scale for every pair that does move. A body type
the adapter cannot read counts as dynamic, so a missing field never deletes a
fact silently.

MS-HAB emits no object-object spatial edges, so it requires the object-object
*planar* scale only -- its obj-obj compatibility near gate reads it -- and
`required_bin_keys` is scope-aware for that reason. The whitelist schema
version is descriptive; the missing-bin check is what rejects a pre-split
asset.

### Initial physical pairs

A physical object-object relation already true in the first graph of an
episode is scene layout, not an affordance to pursue. Those unordered pairs
are captured once (`GraphBuilder._initial_captured`, cleared on reset) and
their compatibility edges are suppressed for the rest of the episode. Physical
and spatial facts are kept, EE-object affordances are never suppressed, and a
pair that first becomes physical later is not affected.

The matched command keeps the same Dreamer reconstruction objective and eager
execution but constructs no graph or semantic parameters. Graph observation
keys may still be present and are ignored. This makes `model.graph.enabled` the
single method switch while allowing the environment configuration to stay the
same.

Graph mode requires `model.rep_loss=dreamer`. Whole-update `torch.compile` is
skipped in graph mode because the compact edge count is dynamic; the graph-free
arm retains the base repository's existing compile behavior.

## Server checks

Run correctness tests:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

Check that padding width no longer controls graph compute:

```bash
python runs/benchmark_graph.py \
  --edge-widths 96 168 \
  --real-edges 72 \
  --batch 8 \
  --length 32 \
  --units 512 \
  --layers 2
```

The important result is the ratio between 96 and 168. It should be close to
one and no more than about 1.20x. Absolute time determines how much overhead
the unchanged two-layer, 512-wide method adds to a full Dreamer update.

## Offline assets

The graph runtime reads three mined assets, and two of them are namespaced by
MS-HAB task group:

```
scenegraph/configs/
  affordances/<group>.json              per-object grasp/contact/support geometry
  subtask_whitelists_raw/<group>/       every entity the rollouts touched
  subtask_whitelists/<group>/           the pruned runtime gate + pick_all.json
  instructions.npz                      shared: keyed by <subtask>/actor:<object>
```

Namespacing is not bookkeeping. The same object rests on different furniture in
each task -- set_table's bowl starts inside a counter drawer, prepare_groceries'
on the counter -- so a whitelist mined under one task names supporters the other
never produces. Such a file loads and validates perfectly; nothing but the group
label distinguishes it. `env.mshab_task` selects the directory, every asset
records the group it was mined for, and the builder refuses one that disagrees.

Affordances are per group for the same reason, which is less obvious because
grasp anchors are expressed in the object's own frame. Three things make the
file scene-bound anyway: over a third of its entries are furniture
(`link:fridge-0/body`, `link:kitchen_counter-0/body`, scene backgrounds) whose
support and contact components exist only because the rollouts touched them;
grasp components are policy-conditioned, and each group trains its own
per-object policy, so a bowl lifted out of a drawer is not sampled like a bowl
lifted off a counter; and `build_subtask_whitelists` reads this file to mine the
compatibility-change bins, so a shared one would let one group's geometry set
another group's relation scales. A single file would also make the result
depend on which group was mined last. Mining affordances is a post-processing
pass over rollouts already on disk, so the separation costs minutes, not
sim-hours. For more samples, raise `--n-success` within a group rather than
merging across groups.

Mine one group per invocation. Passing several would collapse each shared object
onto whichever task sorted first:

```bash
python -m scenegraph.tools.prepare_assets --mshab-task set_table --subtask pick --clean
```

`--dry-run` prints the coverage table and every subcommand without running any
of them; collection is measured in sim-hours, so start there. `--clean` deletes
the selected group's artifacts and nothing else.

The pipeline runs five stages, each usable standalone:

| stage | tool | writes |
| --- | --- | --- |
| collect | `collect_robot_success_states` | `$MS_ASSET_DIR/data/robot_success_states/fetch/<group>/<subtask>/` |
| affordances | `build_affordances` | `configs/affordances/<group>.json` |
| mine | `build_subtask_whitelists` | `configs/subtask_whitelists_raw/<group>/` |
| prune | `prune_whitelists` | `configs/subtask_whitelists/<group>/` |
| instructions | `build_instruction_embeddings` | `configs/instructions.npz` |

Mining and pruning are separate on purpose. Raw whitelists keep every entity the
robot interacted with (`--membership-policy full-evidence`), so changing what
the runtime admits costs one re-prune instead of another collection run, and the
evidence a rule discarded stays on disk. The default runtime policy,
`target-supporters`, keeps the target plus whatever directly supports it: a
rollout contacts whatever is in the way, so admitting every contacted entity
fills a pick-the-bowl graph with the groceries the arm brushed past.

Relation bin edges come from `<group>/pick_all.json` and from nowhere else.
There is no scale profile to fall back on, because a hand-written one makes a
relation token mean a distance the task's own demonstrations never produced.
A union asset that fails to calibrate an absolute relation raises at bind time.

Only `pick`, `open` and `close` have collectors. `place` has none, so these are
assets for the pick training environment, not complete long-horizon task-group
assets -- inventory `mshab_checkpoints/rl/<group>/` before assuming otherwise.

## MS-HAB runtime smoke

The repository contains only the runtime scene-graph slice. It uses externally
installed ManiSkill and MS-HAB packages and their ReplicaCAD task plans; it
does not import code from ReLDreamer.

Before a long run, exercise reset, named RGB cameras, segmentation, frozen
DINOv2 features, graph packing, the semantic RSSM, and policy inference:

```bash
python tests/probes/smoke_mshab.py \
  --num-envs 8 \
  --steps 4 \
  --build-configs 4 \
  --mshab-task prepare_groceries \
  --mshab-obj all
```

Add `--graph-only` to exercise the graph-only RSSM without a pixel CNN or
stochastic `z` state.

The smoke performs no replay writes and no optimizer update. Use CPU replay
storage for training so fixed-width RGB and graph records do not consume VRAM.

## W&B logging

Enable the optional scalar-only backend with `wandb.enabled=true`. It sends
episode return/length/success, losses, optimizer state, throughput, and graph
health diagnostics. Videos, histograms, gradients, observations, and the full
resolved Hydra configuration remain local.

Use the same project and group for matched comparisons, and a different name
for each method:

```bash
wandb.enabled=true \
wandb.project=RelRL \
wandb.entity=letuanhf-hanoi-university-of-science-and-technology \
wandb.group=na \
wandb.name=graph
```
