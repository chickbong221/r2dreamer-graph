# sim_vla setup

```bash
bash sim_vla/install.sh
```

## PickCube imitation

```bash
conda activate dreamer
python -m sim_vla.training.pipeline \
  --task pickcube --experiment graph_progress \
  --world-steps 25000 --imitation-steps 25000 --online-steps 0 \
  --world-lr 1e-4 --world-warmup-steps 1000 --world-final-lr 1e-5 \
  --imitation-lr 1e-4 --imitation-warmup-steps 1000 --imitation-final-lr 2.5e-6 \
  --eval-episodes 20 --seed 0 --device cuda \
  --save-checkpoints --out logdir/sim_vla/pickcube/graph_progress_seed0
```
