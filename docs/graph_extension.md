# Compact graph extension

This repository is an independent PyTorch implementation. It does not import
files from ReLDreamer. Only the observation contract and the model behavior
needed by DreamerV3 are retained.

## Observation contract

When `model.graph.enabled=true`, the environment must emit these tensors per
frame. Index zero is padding for entity and relation vocabularies.

| Key | Shape | Storage dtype |
|---|---:|---|
| `graph_node_ent` | `[10]` | `uint8` |
| `graph_node_app` | `[10,2,384]` | `float16` |
| `graph_node_bbox` | `[10,2,4]` | `float16` |
| `graph_node_target` | `[10]` | `uint8` |
| `graph_edge_src` | `[270]` | `uint8` |
| `graph_edge_dst` | `[270]` | `uint8` |
| `graph_edge_rel` | `[270]` | `uint8` |
| `graph_edge_abs` | `[270]` | `uint8` |
| `graph_edge_temp` | `[270]` | `uint8` |

The replay remains fixed-width. `compact_graph()` selects
`graph_edge_rel != 0` on the GPU and offsets endpoints into one disconnected
batch. Relation embeddings, message MLPs, aggregation, and relation decoding
therefore execute only for real edges. The ten node slots remain dense because
their cost is small and retaining them makes target and reconstruction labels
unambiguous.

## Method switch

- Graph semantic method: use `model=size50M_graph`.
- Matched graph-free DreamerV3: use `model=size50M_graph model.graph.enabled=false`.

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
  --edge-widths 96 270 \
  --real-edges 72 \
  --batch 8 \
  --length 32 \
  --units 512 \
  --layers 2
```

The important result is the ratio between 96 and 270. It should be close to
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

The smoke performs no replay writes and no optimizer update. Use CPU replay
storage for training so fixed-width RGB and graph records do not consume VRAM.
