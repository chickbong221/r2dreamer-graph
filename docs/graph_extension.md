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

- Graph semantic method: use `model=size50M_graph`.
- Matched graph-free DreamerV3: use `model=size50M_graph model.graph.enabled=false`.
- Graph-only latent: add `model.graph_only_latent=true` to the graph method.
- Pooled graph-simple: use `model=size50M_graph_simple`, or add
  `model.graph_simple=true` to the graph method.
- Slot graph-simple: use `model=size50M_graph_slots`.

## Simple graph modes

`model.graph_simple=true` swaps the whole graph path for a relation-only one.
It is mutually exclusive with `graph_only_latent` and needs `graph.enabled`.

`graph_simple` alone does not name a contract. `model.graph.state_mode` picks
between two, and the environment is told which through `env.graph.state_mode`:

| Schema | Selected by | Node fields |
|---|---|---|
| `full` | `graph_simple=false` | appearance, bbox, target |
| `simple_pooled_bbox` | `graph_simple=true`, `state_mode=pooled` | bbox, target |
| `simple_slot_uid` | `graph_simple=true`, `state_mode=slots` | uid, target |

Pooled graph-simple addresses a node by the box it currently occupies; slot
graph-simple aligns nodes across frames by UID. Emitting both columns to both
would put a key in replay that one of them must never read, so the packer emits
exactly one. `graph_node_uid` nevertheless stays in the model's reserved graph
key set, so a stale wrapper that still exposes it cannot feed it to the ordinary
MLP encoder and quietly train an identity-conditioned model; a pooled run that
is handed the key raises instead.

Neither simple schema constructs DINO or reads RGB in the graph builder.
`graph_node_app` is absent from the observation space, the transition, replay
and the sampled batch -- not zeroed -- which removes about 12.3 KB per
environment step. The pooled schema keeps boxes, at 8 x 2 x 4 float16 = 128
bytes per frame, and extracts them without any appearance machinery: one lookup
table from segmentation id to node row and two boolean projections per camera,
instead of one `np.isin` sweep and one patch-grid reduction per node. Patch
coverage is never computed.

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
key, not an identity one, so two nodes with identical boxes are not separable --
under `staleness_enabled: false` only currently segmented objects are emitted,
so the end effector is the one node that can have an empty box, and it is the
only node of its type. Turning staleness on would break that argument.

`loss/node` here averages entity cross entropy and box SmoothL1, reported
separately as `node_ent_loss` and `node_bbox_loss`. It is **not** comparable
with full mode's `loss/node`, which averages appearance, boxes and visibility.
Box error is averaged over the four coordinates rather than summed, and every
term is reduced per valid node and then per frame, so loss magnitude does not
track graph size. Box IoU is an evaluation metric, not an update-loop one.

### Progress in pooled graph-simple

`progress.enabled=true` works in both simple modes. Pooled reads one fused head
`H(g_hat) -> [R, n_abs]` covering exactly the relations the scorer names -- six
for the built-in Pick table -- rather than decoding a distribution per stage or
per candidate target. The head lives inside `SimpleGraphDecoder`, so the
optimizer, the checkpoint and `clone_and_freeze` pick it up automatically and
imagination reads the parameters training wrote. Relation ids are mapped to
output rows through the scorer's own relation order; the id is never the row.

`prior_progress_relabs` supervises that head on the observed EE-to-target facts.
The teacher mask is `not is_first and not is_last and target flagged and the
edge exists`: `g_hat` has no preceding episode to predict the reset frame from,
and a target admitted on any later frame is supervised immediately. Nothing
about `target_resolved` reaches replay -- when the target is unresolved,
unadmitted or occluded, the flag is dark and the edges are gone, so the frame
drops out on its own.

That last point is recurrent, not one-off. With `staleness_enabled: false` the
target vertex is only emitted while a camera sees it; the registry holds its
*index* through occlusion but the flag itself goes dark every time it is
occluded. The design assumes one fixed target per episode and relies on `h` to
carry that role through absence. There is no target-null class and no
target-presence head, so a task where the target can change or be absent within
an episode would need one.

Progress never runs inside a loop. During imagination `g` is already a
contiguous slice of the stacked `imag_feat` (`[z, g, h]`), so after the rollout
one linear, one softmax and one contraction cover all `B x H` states at once.
The potential itself is a single precomputed `[R, n_abs]` matrix contraction,
exactly equal to the cumulative-stage sum for `soft: true`. The progress critic
input is `[imag_feat, vec(p)]` and nothing else -- no masks, no counts, no
boxes, no observed labels. It is trained and normalised separately and enters
the actor as `A = A_env + beta * A_progress`; the reported task return, the
reward head, the continuation head and the environment critic are untouched.

`beta` is the only scheduled quantity. It is zero until
`progress.beta_warmup_start` environment steps, linear to `progress.beta` by
`progress.beta_warmup_end`, and constant after; `prior_progress_relabs` and
`progress_value` run from step 0 throughout. That split is possible because the
progress critic reads a detached input, so training it early moves neither the
world model, the graph decoder nor the actor -- it only fits the baseline for
the stationary relations of a robot that has not learned to move yet. Turning
beta on together with a cold critic would instead let an unfitted critic steer,
and an immature world model can hallucinate relation changes under imagined
actions long after the real arm is still. `train/progress_beta` logs the current
value and `train/progress_influence` logs
`beta * E|A_progress| / E|A_env|`, which is the ratio worth watching during the
ramp: roughly 5-20% at the plateau, under 1% means beta is too small to matter,
well over 25% means progress is doing the steering. The clean paper comparison
holds every loss fixed and varies only this schedule: `beta: 0.0` throughout for
the control, the warm-up for the treatment.

`r_progress = (1 - gamma) * Phi` is a bounded progress-*potential* return, not
classical potential-difference shaping: it rewards reaching and holding high
progress states rather than improving between them.

### Slot graph-simple

The replay contract keeps `graph_node_uid` and has no boxes:

| Key | Shape | Storage dtype |
|---|---:|---|
| `graph_node_ent` | `[8]` | `uint8` |
| `graph_node_uid` | `[8]` | `uint8` |
| `graph_node_target` | `[8]` | `uint8` |
| `graph_edge_*` | `[168]` | `uint8` |

UIDs are episode-scoped, allocated on first sight, kept when an object leaves
the view, and never handed to a second object before reset. Codes are permuted
per episode so a UID means "the same object as before" and nothing more.
Overflow raises; size `model.graph.uid_vocab` above the peak
`episode/graph_episode_entities` rather than letting two objects alias. The
ceiling is 256: UIDs are packed as `uint8`, and the replay buffer's
`index_put` has no `uint16` kernel. Going wider means moving to `int32`, not
`uint16`. Nothing assigns or packs a UID under the pooled schema.

`SlotAligner`, slot occupancy, births and deaths, candidate target decoding and
per-candidate mixing belong to this mode alone and never execute under
`state_mode: pooled`.

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
python -m unittest test_graph test_graph_simple test_semantic_rssm \n  test_slot_dynamics test_dreamer_graph test_entity_registry \n  test_mshab_contract
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
python runs/smoke_mshab.py \
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
