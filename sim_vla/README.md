# PickCube imitation with `graph_progress`

## 1. Clone and install

```bash
git clone https://github.com/chickbong221/r2dreamer-graph.git
cd r2dreamer-graph
bash sim_vla/install.sh
conda activate dreamer
```

## 2. Collect PickCube demonstrations

```bash
python -m sim_vla.data.collect \
  --env-id PickCube-v1 \
  --num-traj 1000 \
  --num-procs 8 \
  --out-dir data/sim_vla_demos \
  --name demos

test -f data/sim_vla_demos/PickCube-v1/demos.h5
test -f data/sim_vla_demos/PickCube-v1/demos.json
```

## 3. Train and evaluate imitation

```bash
python -m sim_vla.training.pipeline \
  --task pickcube --experiment graph_progress \
  --world-steps 25000 --imitation-steps 25000 --online-steps 0 \
  --world-lr 1e-4 --world-warmup-steps 1000 --world-final-lr 1e-5 \
  --imitation-lr 1e-4 --imitation-warmup-steps 1000 --imitation-final-lr 2.5e-6 \
  --eval-episodes 20 --seed 0 --device cuda \
  --save-checkpoints --out logdir/sim_vla/pickcube/graph_progress_seed0
```

The run has three stages:

1. **World model training:** Learn from the collected PickCube images, robot states, actions, and scene graphs. With `graph_progress`, the world model also learns a head that predicts task progress from the graph.
2. **Imitation learning:** Freeze the trained world model and train SmolVLA to reproduce the demonstrated actions from its state features.
3. **Evaluation:** Run the imitation policy on new PickCube episodes in the simulator and report its success rate. `--online-steps 0` stops after this stage; there is no online reinforcement learning or progress reward shaping.

The output directory contains the world model checkpoint, imitation checkpoint, and evaluation results.
