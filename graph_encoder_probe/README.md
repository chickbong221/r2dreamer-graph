# Graph encoder sensitivity probe

**The question.** After reconstruction training, does changing a small part of a
graph produce a measurably different encoder latent?

Nothing else. No claim about unseen graphs, no claim about the RSSM, the actor
or imagination. The measured vector is `GraphEncoder`'s pooled token, before any
dynamics touch it.

## What is wired to what

```text
packed graph G
      |
  GraphEncoder            (imported unchanged from graph.py)
      |
  z = encoding.token      512-d pooled readout
      |
  SimpleGraphDecoder      (imported unchanged from graph.py)
      |
  node attributes + relation labels
```

`z` is handed to the decoder as its semantic input, alongside the decoder's own
geometry queries. Nothing sits in between -- an adapter would be a third module
whose gradients also shaped the thing being measured.

This wiring has a property worth knowing: the decoder never reads the encoder's
per-node vectors. It addresses each node with that node's own box and centroid
and conditions on `z` alone, so **every gradient the encoder receives arrives
through the pooled token**. Whatever the token does not carry, the reconstruction
loss cannot ask for.

Architecture is the repository's, unchanged:

| Setting | Value |
|---|---:|
| Message-passing layers | 2 |
| Encoder width | 512 |
| Pooled token | 512 |
| Decoder width | 256 |
| Node capacity | 8 |
| Edge capacity | 168 |
| Cameras, entity vocabulary | resolved from the collected data |

## The four edits

Each pair holds everything fixed but one thing.

| Group | What changes | What is pinned |
|---|---|---|
| `absolute` | one edge's sigma becomes another label that relation may legally take | nodes, geometry, endpoints, relation type |
| `temporal` | one non-padding delta becomes another change label | everything else |
| `geometry` | one node's centroid moves 1-5 cm along one axis | other nodes, all edge labels |
| `assignment` | two edges of one relation exchange labels | geometry, topology, counts, label histogram |

`assignment` is the group that cannot be answered by counting: both graphs carry
the same labels over the same topology, and only the pairing differs.

Plus unchanged **controls** -- the same frame twice. Whatever distance they show
is what "no change at all" costs in this precision on this hardware, and it is
the floor the tolerance is built from.

Some edited graphs are physically inconsistent, deliberately: a centroid moves
while the boxes that would have followed it stay put. The point is to isolate
what the encoder is sensitive to, not to simulate a plausible frame.

Every generated pair is verified against the packed tensors before it is saved:
the edited row is one the encoder actually consumes (never padding), the written
label is legal for its relation, and no other cell moved.

## Running it

Collection needs ManiSkill and the mined assets; training needs torch. Both live
on the server.

Pre-flight first -- a short collection, twelve updates, and the six plumbing
checks, into `outputs/dataset_smoke` and `outputs/pairs_smoke` so the real cache
is untouched:

```bash
python -m graph_encoder_probe.run --config graph_encoder_probe/config.yaml --smoke
```

Then the full run:

```bash
python -m graph_encoder_probe.run --config graph_encoder_probe/config.yaml
```

The runner reuses a cache that matches the configured collection, so the
simulator starts once. A probe set built from that exact cache is reused too --
the pairs must be identical across any runs that get compared.

Useful flags: `--stage collect|pairs|train`, `--recollect`, `--rebuild-pairs`,
`--checks`, `--run-name NAME`, and `--set train.max_updates=500` for any single
config leaf.

Tests (the model half needs torch; the rest is pure numpy):

```bash
python -m unittest discover -s graph_encoder_probe/tests -t . -v
```

## Throughput

These graphs are tiny -- 8 nodes, a few dozen real edges, 2.9M parameters -- so a
run is bound by launch latency and host-device round trips, not by arithmetic.
Expect a few hundred MiB of VRAM and no meaningful change to it from any tuning:
the weights plus Adam states are ~33 MiB and the whole pool is ~35 MiB.

Two things the loop does about that, both semantics-preserving:

- **`train.pool_on_device`** (default `auto`, on for an accelerator under
  `pool_device_max_gib`): the entire packed pool lives on the training device, so
  a batch is one `index_select` per field rather than nine pageable
  host-to-device copies -- and a pageable copy is synchronous. The epoch's
  shuffled order is uploaded once per epoch, not once per batch.
- The batch loss is kept as a device tensor and reduced only at the probe
  interval, instead of forcing a synchronisation every update to write a number
  that is read every 250.

`progress.csv` carries `updates_per_sec` per interval and the final line reports
the overall rate, so a change here is visible rather than believed.

What is deliberately *not* done: `torch.compile` (`compact_graph` strips padding
with a data-dependent boolean mask, so shapes change every batch and it would
recompile continuously -- the repository sets `compile: False` for this model
too), and mixed precision (the experiment is specified as float32). Raising
`train.batch_size` would use more of the device per update, but batch 128 and
10,000 updates are part of the experiment's definition -- changing either changes
what the run measures, so it is a decision, not a tuning knob.

## What comes out

Console, one line per probe:

```text
update    250 | recon   1.8342 | absol 25/25 tempo 24/25 geome 25/25 assig 22/25 contr 0/8 | dist 4.7e-02
```

In `outputs/runs/<name>/`:

| File | Contents |
|---|---|
| `report.md` | the headline tables: per group, and every pair before vs after |
| `probe_rows.csv` | one row per pair per probe -- distance, cosine, both norms, detected flag |
| `progress.csv`, `history.json` | loss and per-group distance against updates |
| `loss.png`, `latent_distance.png` | the two plots |
| `decoder_*.csv`, `decoder_*.png` | what the decoder recovered (below) |
| `checkpoint_init.pt`, `checkpoint_final.pt` | weights before the first update and at the end |
| `resolved_config.yaml` | exactly what this run was given |
| `result.json` | stop reason, tolerance, per-group summary, check verdict |

## Does the decoded label match the true one?

The latent probe asks whether two graphs land on different tokens. The decoder
readout asks the other half: what comes back out of one token. Every head that
matters here is discrete, so the readout is argmax against ground truth, under
the decoder's own masks -- the target among admissible rows, the absolute label
among those its relation may legally take, the temporal label among the
non-padding classes. Scoring against anything wider would report a number the
loss never optimised.

`SimpleGraphDecoder.forward` returns losses and metrics, not predictions, so
`GraphProbe.predict` taps the two output projections with forward hooks while
the decoder runs normally. What is scored is exactly what was scored by the
loss, and a test asserts the accuracies computed here equal the decoder's own
`node_ent_acc`, `relabs_acc` and `reltemp_acc` to five places.

| Output | Contents |
|---|---|
| `decoder_eval.csv` | per probe: agreement per head, per relation, and the box MAE |
| `decoder_predictions.csv` | one row per node and per fact: true label, decoded label, match |
| `decoder_confusion.png` | which label gets mistaken for which, per head |
| `decoder_accuracy.png` | agreement against updates |
| `decoder_examples.png` | a few frames item by item, true beside decoded |

**On "test set":** this experiment has none by design -- both members of every
pair are trained on, and the probe set is drawn from the training pool. The
readout therefore defaults to the fixed monitor subset and is labelled `train`
everywhere it appears, so no figure can imply generalisation the run did not
test. Setting `train.holdout_frames: N` carves a genuine held-out set out of the
collected frames instead (never probe-pair members, which must stay in
training); the label then becomes `holdout` and the report says so.

A head with nothing to score reports *no items*, not zero accuracy. On
PlaceSphere `node_target` is in that state -- see the caveat above -- and
"nothing to score" and "never right" must not print the same.

## How a difference is decided

Per pair, on `z_A` and `z_B` with gradients off and identical preprocessing:

- **RMS distance** `sqrt(mean_j (z_A,j - z_B,j)^2)` -- how far apart the vectors are.
- **Cosine similarity** `z_A . z_B / (|z_A| |z_B| + eps)` -- whether they point the
  same way. Two tokens can sit at cosine 0.9999 and still differ in length, so
  cosine never decides on its own.

A changed pair counts as **detected** only above a tolerance built from two
measured things: the largest distance any control or repeated encoding shows
(the numerical floor -- the encoder aggregates with `index_add`, which has no
fixed summation order on a GPU), and a small fraction of the token's own RMS
magnitude. The raw distance is always in the table, so a weak response cannot
hide behind a binary flag. A token whose norm is effectively zero is flagged
rather than given a cosine.

## Training, and what it deliberately does not do

Both members of every pair are in the training pool, shuffled in each epoch.
There is **no train/validation/test split**: the question is whether the encoder
separates graphs it has seen, and the probe set is a measurement set drawn from
the training pool.

The loss is the decoder's own four reconstruction terms and nothing else:

```text
L = L_node + L_nodetgt + L_relabs + L_reltemp
```

No contrastive or latent-separation term. Adding one would train the behaviour
the experiment is trying to observe.

One honest caveat about `L_nodetgt`: the target flag names the active *subtask*
object, which is an MS-HAB concept. A plain ManiSkill task like `PlaceSphere-v1`
never sets it, so on this task the term has only negatives to learn from and its
value is not a target-recovery score. The decoder's own loss definitions are left
untouched -- the objective is the one the plan names -- and the run prints a note
and records it in the report's flags when the pool carries no target at all.

Adam, lr `3e-4`, batch 128 graphs, float32, at most 10,000 updates, probed every
250 and once **before the first update** -- a randomly initialised encoder
already separates these pairs, and the run's real subject is whether
reconstruction training preserves, sharpens or erodes that.

Stopping: at least 2,000 updates, then stop if the fixed-subset reconstruction
loss fails to improve by 1% over eight consecutive checks; otherwise stop at the
budget. `stop_reason` records which happened, and reaching the budget is never
reported as convergence.

## Collection notes

`PlaceSphere-v1` driven by its own scripted motion-planning solution, one CPU
env, 200 reset seeds by default. Graphs are built on **every** control step --
temporal labels difference over the last K frames, and a builder fed one step in
five would report a change spanning five times the horizon it was mined for --
and history is reset at every episode boundary. Frames from failed attempts are
kept; only the success label is missing from them, and the probe never reads it.

Only the encoder's own inputs are cached, in compressed npz shards, at the
repository's field names and dtypes. RGB and simulator state are the expensive
half and nothing here reads them.

The end-of-collection summary prints episode and graph counts, the absolute and
temporal labels actually observed, and warns about anything dropped: packer
rejections, node or edge capacity overflow, unresolved targets, and relations
that never changed label. It also prints how many of the kept graphs are
*distinct* -- a long stationary stretch inflates the frame count without adding
variety, so the graph count alone should not decide whether the dataset is
enough.

## Layout

```text
graph_encoder_probe/
    config.yaml     one file drives every stage
    collect.py      simulator -> packed graph cache (the only stage that needs ManiSkill)
    dataset.py      shards, the frame table, and the batch conversion
    pairs.py        the four edits, their verification, and the fixed probe set
    model.py        encoder -> decoder, and the config resolved from the data
    train.py        reconstruction training with the probe read off at intervals
    evaluate.py     distances, tolerance, tables, plots
    decoder_eval.py what the decoder recovered: agreement, confusions, figures
    run.py          orchestration, the six plumbing checks, the report
    tests/          all of it without a simulator
    outputs/        dataset/, pairs/, runs/  (git-ignored)
```
