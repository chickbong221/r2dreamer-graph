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
- Simple graph: add `model.graph_simple=true` to the graph method.

## Simple graph mode

`model.graph_simple=true` swaps the whole graph path for a relation-only one.
It is mutually exclusive with `graph_only_latent` and needs `graph.enabled`.

The replay contract loses appearance and boxes and gains an identity column:

| Key | Shape | Storage dtype |
|---|---:|---|
| `graph_node_ent` | `[8]` | `uint8` |
| `graph_node_uid` | `[8]` | `uint16` |
| `graph_node_target` | `[8]` | `uint8` |
| `graph_edge_*` | `[168]` | `uint8` |

`graph_node_app` and `graph_node_bbox` are absent from the observation space,
the transition, replay and the sampled batch -- not zeroed. DINO is never
constructed, RGB is never read by the graph builder, and per-node boxes and
patch coverage are never computed. That removes about 12.4 KB per environment
step, less 16 bytes for the UID column.

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

The graph decoder no longer reads the encoder's node vectors. It reconstructs
each node by querying `g` with that node's UID, which makes `nodetgt`,
`relabs` and `reltemp` a measure of what `g` retained. This is conditional
graph-attribute reconstruction: nodes, edge endpoints and relation families
are supplied, and only the target flag and the absolute/temporal labels are
recovered. `node`, appearance, bbox and visibility losses are gone.

UIDs are episode-scoped, allocated on first sight, kept when an object leaves
the view, and never handed to a second object before reset. Codes are permuted
per episode so a UID means "the same object as before" and nothing more.
Overflow raises; size `model.graph.uid_vocab` above the peak
`episode/graph_episode_entities` rather than letting two objects alias.

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
python -m unittest test_graph test_semantic_rssm test_dreamer_graph test_mshab_contract
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
