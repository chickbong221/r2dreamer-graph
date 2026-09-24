# sim_vla: world model + SmolVLA imitation

A world model is trained on ManiSkill demonstrations (Stage 1A). Then the
pretrained SmolVLA action expert learns to reproduce the demonstrated action
chunks, conditioned on that world model's state (Stage 1B). Finally the policy
runs in the simulator on initial states no demonstration used. One command
runs all three, in one process:

```
demos.h5 ──► 1A world model ──► 1B adapter + SmolVLA action expert ──► evaluation
             (one per arm)      (world model frozen)                  (20 episodes, new seeds)
```

This README covers that path only, which is what `--online-steps 0` runs.
The code also has an online RL stage (Stage 2); it is not used or described
here.

## The arms

| `--experiment` | `model.graph.enabled` | `model.progress.enabled` | policy state |
|---|---|---|---|
| `dreamer` | false | false | `(h, z)` |
| `graph` | true | false | `(h, z, g)` |
| `graph_progress` | true | true | `(h, z, g)` |

The launch scripts compare `dreamer` (the `*_baseline.sh` files) against
`graph_progress`. `progress.enabled` without `graph.enabled` is refused at
config load: its targets come from the scene-graph schedule.

## 1. Install

Linux x86_64, an NVIDIA GPU, and conda. From the repo root:

```bash
bash sim_vla/install.sh
```

This creates a conda env named `dreamer`, the env `runs/sim_vla/imitation/*.sh`
activate. It uses Python 3.11 from conda-forge, installs the pins in
[`requirements.txt`](requirements.txt), and ends with the checks of `--verify`
(below). Expect about 9 GB on disk and 20 minutes, most of it downloading. The
script never modifies an existing env:

| | |
|---|---|
| `ENV_NAME=simvla bash sim_vla/install.sh` | build under another name, then change `conda activate dreamer` in the script you submit |
| `RECREATE=1 bash sim_vla/install.sh` | delete the existing `ENV_NAME` env first, then build |
| `TORCH_CUDA=cu126 bash sim_vla/install.sh` | the CUDA 12.6 build of torch, if the driver is too old for 12.8 |

What gets installed, and why:

| packages | version | needed for |
|---|---|---|
| torch, torchvision, torchcodec | 2.8.0, 0.23.0, 0.7.0 | the models. lerobot requires the last two, and all three must be built for the same torch |
| numpy | 1.26.4 | what the demos were collected with; ManiSkill's motion planner (mplib, used to collect) segfaults on NumPy 2. lerobot's rerun-sdk declares numpy>=2 but is never imported here, so NumPy is installed in a second step and `pip check` reports that one line |
| mani_skill, sapien, gymnasium | 3.0.1, 3.0.3, 1.2.0 | the evaluation env. Every `demos.json` records mani_skill 3.0.1 |
| lerobot[smolvla], transformers, huggingface-hub | 0.4.4, 4.57.6, 0.35.3 | SmolVLA in Stage 1B. 0.4.4 is the lerobot release this code is verified against on Python 3.11 |
| h5py, PyYAML, omegaconf, tensordict, tensorboard, wandb | pinned | imported by this repo on the pipeline's path |
| opencv-python, opencv-python-headless | 4.10.0.84 | sapien requires one and lerobot the other. Both install into the same `cv2/` directory, so they get the same version, and the headless build is reinstalled last so that is the one on disk. 4.10 works with NumPy 1.26 |

Nothing else is installed, because nothing on this path imports it: no JAX or
ReLDreamer `dreamer` package, and no metaworld, mujoco, dm_control, mshab,
robosuite, torchrl or hydra-core. The repo itself is not pip-installed. Run
from the repo root, where `python -m` finds `sim_vla`, `envs`, `scenegraph`
and the root modules.

This replaces the earlier recipe (ReLDreamer's `install_maniskill.sh`, then
`pip install -e ".[metaworld]"`, then lerobot):

- `pip install -e .` no longer works here: `pyproject.toml` was removed from
  this repo in commit 1af4733.
- Installed in that order, each step overrode the pins of the one before.
  lerobot moved torch off 2.8, the version tensordict 0.9.1 and torchrl 0.9.2
  are built for, and pulled in NumPy 2, which then had to be put back to 1.26
  by hand.
- sapien and lerobot each brought an OpenCV package, at different versions,
  into the same `cv2/` directory.
- The rest (JAX, metaworld, mujoco, mshab) belongs to other projects.

### Check an env

```bash
conda activate dreamer
bash sim_vla/install.sh --verify
```

This checks: the Python version, every pin, `pip check` (allowing only
rerun-sdk's numpy>=2), importing each module
the pipeline uses (lazy ones included), a bf16 matmul on the GPU, building and
rendering one `StackCube-v1` env the way evaluation builds it, and loading the
pinned SmolVLA checkpoint. Exit 0 means passed. Exit 1 means something is
broken. Exit 2 means something could not be checked on this node, such as CUDA
on a login node without a GPU. Run it on a GPU node before trusting a new
machine.

### Vulkan

SAPIEN renders the camera observations through Vulkan, so evaluation needs
the NVIDIA Vulkan driver on the node (`vulkaninfo --summary` lists the GPU).
Where the system driver does not give SAPIEN a working Vulkan ICD, the launch
scripts point Vulkan at a user-space copy of the NVIDIA 570.133.20 driver:
see the `NVIDIA_USERSPACE_*` block at the top of any
`runs/sim_vla/imitation/*.sh`. It expects the extracted driver in
`~/nvidia-userspace`:

```bash
mkdir -p ~/nvidia-userspace && cd ~/nvidia-userspace
wget https://us.download.nvidia.com/tesla/570.133.20/NVIDIA-Linux-x86_64-570.133.20.run
sh NVIDIA-Linux-x86_64-570.133.20.run --extract-only
```

If rendering still fails, compare that version with the driver `nvidia-smi`
reports on the compute node.

## 2. Data

One dataset per task, shared by every arm:

```
data/sim_vla_demos/<EnvId>/demos.h5     # the episodes
data/sim_vla_demos/<EnvId>/demos.json   # metadata the evaluation env is rebuilt from
```

| `--task` | env | instruction given to SmolVLA |
|---|---|---|
| `pickcube` | PickCube-v1 | pick up the cube and lift it to the goal |
| `stackcube` | StackCube-v1 | stack the red cube on top of the green cube |
| `peginsertion` | PegInsertionSide-v1 | insert the peg into the hole in the box |
| `placesphere` | PlaceSphere-v1 | place the sphere into the bin |

The location comes from `data.root` and `data.name` in `configs/base.yaml`
and `task.dataset` in `configs/tasks/<task>.yaml`. The pipeline has no flag
for it, so point `data/sim_vla_demos` at the demos, as the launch scripts do:

```bash
mkdir -p data && ln -sfn /path/to/maniskill_demos data/sim_vla_demos
```

Training reads only `demos.h5` and `demos.json`. The `graphs_jsonl/`
directory next to them is a human-readable export.

What the demos fix:

- **The evaluation env.** It is built from `demos.json`: robot, controller,
  cameras, resolution, proprioception fields and reward mode. The built env is
  then checked against the recording, and a mismatch stops the run.
- **The simulator version.** The demos record `mani_skill 3.0.1`, the version
  `requirements.txt` installs.
- **Paths, for the graph arms.** `graph.whitelist_dir` and
  `graph.thresholds_path` in `demos.json` are absolute paths into the checkout
  that collected the data, for example
  `/home/duongnm2/projects/r2dreamer-graph/scenegraph/configs/...`. Train from
  a checkout at that same path. Evaluation compares their digests with the
  recorded ones, so a changed whitelist is refused rather than used.

## 3. The SmolVLA checkpoint

`configs/base.yaml` pins `actor.pretrained: lerobot/smolvla_base` at
`actor.revision: d9f33c94a60fb382c90dea2164c96845bd955e28`. Stage 1B loads it
from the Hugging Face cache in `$HF_HOME`, which defaults to
`~/.cache/huggingface`, and downloads it on first use. The launch scripts
export their own `HF_HOME`. To fill that cache ahead of time, for example on
the login node:

```bash
export HF_HOME=/the/path/your/launch/script/exports
bash sim_vla/install.sh --verify        # loads the pinned revision, downloading it if missing
```

Because the revision is a commit hash, no Hub call is needed to resolve it.
On compute nodes without internet, also `export HF_HUB_OFFLINE=1`.

## 4. Train and evaluate

From the repo root, in the env:

```bash
python -m sim_vla.training.pipeline \
  --task stackcube --experiment graph_progress \
  --world-steps 25000 --imitation-steps 25000 \
  --world-lr 1e-4 --world-warmup-steps 1000 --world-final-lr 1e-5 \
  --imitation-lr 1e-4 --imitation-warmup-steps 1000 --imitation-final-lr 2.5e-6 \
  --online-steps 0 --eval-episodes 20 \
  --seed 0 --device cuda \
  --save-checkpoints --out logdir/sim_vla/stackcube/graph_progress_seed0
```

These are the settings of every `runs/sim_vla/imitation/slurm_<task>_<arm>.sh`.

**Stage 1A: world model.** Batches of `--batch-size` 16 windows. Each window
has `--burn-in` 8 rows that only rebuild the recurrent state, then
`--sequence-length` 64 scored rows. The RSSM is trained with image and
proprioception reconstruction, reward and continuation heads, using the model
preset `configs/model/size50M_graph_simple.yaml` (`--model-config`). The graph
arms add the graph encoder, the semantic latent `g`, the graph decoder and
their losses. `graph_progress` also trains a progress head jointly on the
schedule's potential. In this imitation-only setup that head only shapes the
representation, because progress shaping is a Stage 2 feature. Normalization
statistics are fitted on the demos here and used by 1B and by evaluation.

**Stage 1B: imitation.** Stage 1B uses the same world-model object, frozen.
Each window is encoded causally into posterior states. At every eligible row,
the adapter turns the state into one conditioning token placed beside the
task instruction, and SmolVLA is trained with flow matching to reproduce the
next chunk of demonstrated actions. The adapter, SmolVLA's action expert and
its action projections are trained. The VLM backbone and the world model stay
frozen.

**Evaluation.** Evaluation runs after 1B when `--online-steps 0` and
`--eval-episodes` > 0. It runs `--eval-episodes` episodes (default 20) of 150
steps each, in one CPU-physics env. Seeds start at 900000, clear of every
collection seed. The policy samples a chunk with 5 flow steps, executes the
first `actor.execute` = 5 actions, then replans, and each executed action is
fed back into the world model's posterior. Nothing terminates early, so an
episode can succeed and then undo it:

| metric | meaning |
|---|---|
| `success_rate` | success at any step |
| `success_at_end_rate` | still successful at the last step |
| `env_return_mean` | the environment's own return, never a shaped one |
| `steps_to_success_median` | over the episodes that succeeded |
| `clipped_fraction` | share of steps with an action component outside [-1, 1]. That is `evaluate_policy`'s default range, not the controller's: `pd_joint_pos` actions are joint angles in radians, so this reads close to 1 for any policy |

**Learning rates.** `--world-lr` and `--imitation-lr` are peak rates. Each
ramps up linearly over `--*-warmup-steps` and decays by cosine to
`--*-final-lr` at the stage's last step. Left unset, the rate is constant: the
preset's 4e-5 for 1A and 1e-4 for 1B. The rate of each step is logged as
`world/lr` and `imitation/lr`.

Other flags: `--seed` sets one seed for initialisation, the demo sampler and
the flow sampler (evaluation seeds come from `eval.seeds_start`);
`--batch-size`, `--sequence-length` and `--burn-in` override
`configs/base.yaml`; `--imitation-steps 0` stops after Stage 1A;
`--eval-episodes 0` skips evaluation.

## 5. Launch scripts: `runs/sim_vla/imitation/`

```bash
SEED=1 sbatch runs/sim_vla/imitation/slurm_stackcube_graph_progress.sh
```

| script | runs |
|---|---|
| `slurm_<task>_baseline.sh`, `slurm_<task>_graph_progress.sh` | 1A, 1B, evaluation. The two differ only in `--experiment` |
| `slurm_pickcube_{baseline,graph_progress}_resume.sh` | restore a world model, retrain 1B, evaluate |
| `slurm_placesphere_eval.sh` | evaluate two saved runs, no training |
| `slurm_collect_data.sh` | collect 1000 demos per task (see section 8) |

What they assume about the machine, which is what to change on a new one:
conda in `~/miniconda3` with the env `dreamer`; the repo in
`$HOME/projects/r2dreamer-graph`; the demos linked from
`/home/tuannl/mnt_data/data/maniskill`; `HF_HOME`, `MS_ASSET_DIR` and
`WANDB_API_KEY` exported; the Vulkan block above. Output goes to
`$HOME/logdir/r2dreamer-graph/sim_vla/<timestamp>/<task>/`, and job logs go
to `$HOME/output/`.

## 6. Outputs

With `--save-checkpoints`, `--out` receives:

| file | written after | holds |
|---|---|---|
| `world_model.pt`, `world_model.json` | 1A | weights, plus the arm, dataset and settings they depend on |
| `normalization.json` | 1A | the statistics fitted on the demos |
| `imitation.pt`, `imitation.json` | 1B | adapter and action expert |
| `imitation_eval.json` | evaluation | the metrics above, plus per-episode results |

The checkpoints are written before evaluation starts, so a simulator failure
cannot cost the weights. For `graph_progress` on PickCube the two `.pt` files
are about 0.75 GB and 1.4 GB. Without `--save-checkpoints` nothing is
written: the stages pass their models in memory, and the evaluation numbers
appear only in the log and on W&B.

**W&B:** the project is `sim_vla`, the group is the task and the run is named
`<task>-<arm>`. Stage 1A logs `world/*` against `world/step`, Stage 1B logs
`imitation/*` against `imitation/step`, and the evaluation lands in the run
summary as `imitation_eval_success_rate`, `imitation_eval_success_at_end_rate`,
etc. Export `WANDB_API_KEY` or run `wandb login`. On nodes without network,
set `wandb.mode: offline` in `configs/base.yaml` and `wandb sync` afterwards;
`wandb.enabled: false` turns logging off. Compare arms on
`imitation_eval_success_rate` and `imitation_eval_success_at_end_rate`.

## 7. Re-evaluate, or retrain only Stage 1B

`--resume-from DIR` restores `DIR/world_model.pt`, and `DIR/imitation.pt` when
it exists, instead of training them. The arm, the env, the feature width, the
pretrained revision and the normalization statistics must all match, and a
mismatch is refused. Restoring `imitation.pt` without its world model is
refused too.

Evaluate a finished run without training:

```bash
python -m sim_vla.training.pipeline --task stackcube --experiment graph_progress \
  --resume-from logdir/sim_vla/stackcube/graph_progress_seed0 \
  --world-steps 0 --imitation-steps 0 --online-steps 0 --eval-episodes 20 \
  --seed 0 --device cuda --save-checkpoints --out logdir/sim_vla/stackcube/graph_progress_seed0_eval
```

To retrain only Stage 1B, link `world_model.pt` and `world_model.json` alone
into a fresh directory. Pass that directory as both `--resume-from` and
`--out`, with `--world-steps 0`. `slurm_pickcube_*_resume.sh` do exactly this.

## 8. Hardware, and collecting new demos

**GPU memory.** Stage 1A at the defaults (16 windows of 8 + 64 + 1 rows of
112×112 images) needs more than 24 GB. On an RTX 4090 the `graph_progress` arm
ran out of memory in its first step: 20.3 GiB was allocated and the image
decoder asked for 3.5 GiB more. The launch scripts run on 80 GB H100s. On a
24 GB card, `--batch-size` and `--sequence-length` make it fit (at
`--batch-size 2 --sequence-length 16` the whole pipeline peaked at 5.2 GB),
but they change the experiment, so use the same values for every arm you
compare.

**Collecting new demos** is not needed while the demos exist, and the same env
does it: `python -m sim_vla.data.collect --env-id StackCube-v1 --num-traj 1000`
(see `slurm_collect_data.sh`). It drives ManiSkill's motion-planning solutions
through mplib 0.1.1, which segfaults in `Planner.__init__` under NumPy 2. That
is the reason NumPy stays at 1.26.

## 9. Troubleshooting

| symptom | cause, fix |
|---|---|
| `cannot import name 'is_offline_mode' from 'huggingface_hub'` | transformers and huggingface-hub come from different generations (4.x pairs with hub 0.x). `python -m sim_vla.doctor` names the pair. Reinstall the pins |
| a Vulkan error when the first env is built | no Vulkan driver for the GPU on this node: see Vulkan in section 1 |
| `libGL.so.1: cannot open shared object file` on `import cv2` | the non-headless OpenCV build is on disk: `pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84` |
| `CUDA out of memory` in Stage 1A | see section 8 |
| `the online environment does not match the recording` | the env resolved a different robot, controller or reward than the demos. Keep mani_skill 3.0.1, or re-collect |
| a graph arm cannot read a whitelist or thresholds path | the path recorded in `demos.json` does not exist here: see section 2 |
| segfault in `mplib.Planner` while collecting | NumPy 2 got installed: `pip install numpy==1.26.4` |
| `[wandb] init failed` | training continues without logging. Set `WANDB_API_KEY`, or `wandb.mode: offline` |

## How it works

**Graph disabled means disabled everywhere.** A `dreamer` run has no graph
encoder, no `g`, no graph decoder, no graph losses and no graph-derived
progress, and its batches carry no graph arrays: `data/dataset.py` does not
open them. The acceptance test corrupts and then deletes the stored graphs and
asserts a baseline's batches are byte-identical.

**An arm is chosen before its world model is trained.** A graph-trained world
model with `g` hidden from the actor is not a baseline, because the graph has
already reached `h` and `z`. `runtime/checkpoint.py` refuses to load one arm's
checkpoint as another's.

**Terminations.** The evaluation env runs with `ignore_terminations=True`, as
`envs/maniskill.py` does: episodes end at the horizon. The demos were recorded
without that wrapper and carry terminal flags from the task's success. The
flags are kept as diagnostics, but the loader reports `is_terminal` false
everywhere, so the continuation head never learns a signal the env cannot
produce. Reaching success and keeping it are tracked separately, because the
flag flickers: PickCube's success also wants a static robot.

**The action timeline.** A window's row `t` holds the observation `o_t`. Beside
it sit three arrays:

| array | is | read by |
|---|---|---|
| `action` | `a_(t-1)`, executed before arriving at `o_t` | `RSSM.obs_step` |
| `action_target` | `a_t`, executed at `o_t` | the actor's flow loss |
| `reward` | `r_(t-1)`, earned arriving at `o_t` | the reward head |

`action_valid` says a real action exists at a row. `loss_mask` says the row
is scored and may be conditioned on. `action_target` lives on a target axis
`chunk_size` rows longer than the observation axis. That way the last
eligible row of a window is still supervised on a whole chunk.

**Action units.** The dataset and `env.step` use raw actions. The actor's flow
targets and samples are normalized. `RSSM.obs_step`/`img_step` get
`tanh(x / 3)` of the normalized action, which is injective and stays inside
the unit ball, so `rssm.py`'s projection onto that ball is the identity. Actions
are clipped to the controller's bounds once, inside the policy. The clip is
straight-through, and the command sent to the env is the one fed back as the
next `a_(t-1)`.

**The actor.** This code was read against lerobot 0.4.4 and 0.6.1, which share
the SmolVLA interface; 0.4.4 is installed because 0.6.1 needs Python 3.12.
The flow methods live on `policy.model` (`VLAFlowMatching`). The prefix, which
is the tokenized instruction plus one state token, is run through the VLM once
per observation to build a key/value cache. Every Euler step of the sampler
reuses that cache. No images are passed, because the observation has already
been through the world model.

`state_token_mode: embedding` (the default) makes the adapter emit the VLM's
hidden width and skip `state_proj`. `state_proj` mode would put SmolVLA's
32-wide state projection between the world model and the policy.

LeRobot's flow convention is `x_t = t*noise + (1-t)*actions`, so `t=0` is the
clean action and sampling integrates down from 1. Actions are padded to the
checkpoint's `max_action_dim` (32), and the loss masks the padding. "Frozen"
means `requires_grad=False`, never `detach()`: gradients still flow through the
frozen transformer into the adapter.

**The adapter.** World-model features go through LayerNorm and an MLP to
produce one conditioning token. Only the input width differs between arms;
`LatentAdapter.parameter_report()` states the resulting capacity difference.

## Layout

```
configs/      base + task + experiment yaml; the arm switch lives here
data/         loader, sequences, normalization, audit; collect.py records demos
models/       world_model, latent_adapter, smolvla_actor, pretrained, flow_sampler
envs/         the evaluation env, matched to the dataset's recorded contract
training/     pipeline (entry point), pretrain_world_model (1A), train_imitation (1B)
evaluation/   policy rollouts and their metrics
runtime/      checkpoints that carry the decisions their weights depend on
install.sh    the environment; requirements.txt holds the pins
tests/        the staged test suite (bash sim_vla/run_tests.sh), not needed to train
```
