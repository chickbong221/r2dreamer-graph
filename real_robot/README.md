# Real-robot offline training

Offline training on `locht131/aloha_placing_kitchen_lerobot`: 150 teleoperated
ALOHA episodes of one right arm putting a banana in a pot and closing the lid,
two RGB cameras, 15 Hz, no simulator.

```
all recordings -> full-episode graphs and dense rewards -> world-model pretraining
               -> frozen latent features -> short imagined transitions -> model-assisted IQL
```

Everything the ManiSkill path reads off privileged state — graph facts,
geometry, reward, termination — is reconstructed once here, saved, and then read
by training as plain arrays. The repository's own modules do the modelling:
`MultiEncoder`, `GraphEncoder`, the semantic `RSSM`, `MultiDecoder`,
`SimpleGraphDecoder`, `pack_graph`, the relation vocabularies, the schedule
compiler and `TaskScheduleReplayPotential` are all imported unchanged. Nothing
outside this directory was modified, and nothing outside it imports from here.

The dataset is read directly from its parquet files and MP4s, so the `lerobot`
package is not a dependency and the v2.1 format needs no conversion.

## What trains on what

* **Every episode trains** — preprocessing, the action normaliser, the reward
  scales, the world model, the latent cache, the behaviour policy, the progress
  head and the policy. Nothing is held out; policy performance is measured on
  the robot.
* **Diagnostic episodes** are a fixed subset of those same episodes: 10,
  spread across episode lengths and, once annotations exist, covering every
  outcome they record. They are saved by id in
  `selections/selection_<version>.json`, not copied, and they stay in
  training. Fitting and pipeline behaviour are watched on them and logged as
  `diagnostic/...`; nothing measured on them is evidence of generalisation, and
  every report says so.
* Windows for the world model are contiguous frames of one episode, with
  burn-in, and never cross into another episode.

## Two environments

Preprocessing needs video, perception and Gemini packages; training needs the
repository's pinned stack. Keep them apart.

```bash
# preprocessing (once)
conda create -n realrobot-prep python=3.11 -y && conda activate realrobot-prep
pip install -r real_robot/requirements-preprocess.txt
pip install git+https://github.com/apple/ml-depth-pro.git
mkdir -p real_robot/outputs/depth_pro && curl -L \
  https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o real_robot/outputs/depth_pro/depth_pro.pt
export GEMINI_API_KEY=...        # never written to any file
```

Training runs in the existing `dreamer` environment. Every entry point is
`python -m real_robot.<module>` from the repository root; no `ManiSkill`
environment is created and no simulator assets are read.

## Order of work

Stage 3 is the first milestone: **three annotated episodes with graph
overlays, distance curves and dense reward plots**. Stop there and look at them
before spending the Gemini budget on 150 episodes.

```bash
# 1  source, and what the recorded action field commands
python -m real_robot.preprocessing.download
#    state the command dimensions, units, representation and gripper values in
#    configs/action_mapping.yaml, with evidence other than measurement, and confirm it
python -m real_robot.preprocessing.audit_dataset            # writes audit/action_spec.json

# 2  one shared bin specification, proposed then frozen by hand
python -m real_robot.preprocessing.prepare_videos --episodes pilot
python -m real_robot.preprocessing.define_bins propose      # read the proposal, edit if needed
python -m real_robot.preprocessing.define_bins freeze

# 3  three episodes end to end  <- first milestone
python -m real_robot.preprocessing.annotate_episode  --episodes pilot
python -m real_robot.preprocessing.track_objects     --episodes pilot
python -m real_robot.preprocessing.estimate_geometry all --episodes pilot
python -m real_robot.rewards.kitchen fit-scales      --episodes pilot   # for the plots only
python -m real_robot.evaluation.inspect_rewards      --episodes pilot
python -m real_robot.evaluation.render_annotations   --episodes pilot
python -m real_robot.evaluation.check_bins           --episodes pilot   # the cm bins against geometry

# 4  correct what the overlays show, then freeze preprocessing

# 5  every episode, the diagnostic selection, and the dataset
python -m real_robot.preprocessing.prepare_videos    --episodes all
python -m real_robot.preprocessing.annotate_episode  --episodes all
python -m real_robot.preprocessing.track_objects     --episodes all
python -m real_robot.preprocessing.estimate_geometry all --episodes all
python -m real_robot.data.selection create                  # 10 diagnostic ids, all episodes train
python -m real_robot.rewards.kitchen fit-scales --episodes all --force
python -m real_robot.preprocessing.build_dataset --episodes all --name full_episode_v1

# 6  world model, with periodic diagnostics
python -m real_robot.training.pretrain_world_model --run-name wm_base
python -m real_robot.evaluation.evaluate_world_model --world-model wm_base

# 7  freeze the selected checkpoint and encode one latent cache
python -m real_robot.training.encode_dataset --world-model wm_base --checkpoint final \
    --name wm_base_final --progress

# 8  behaviour policy and one-step imagined transitions
python -m real_robot.training.generate_rollouts --latents wm_base_final --name h1

# 9  model-assisted IQL
python -m real_robot.training.train_model_assisted_iql --latents wm_base_final --rollouts h1 \
    --run-name maiql_base

# 10 the progress-aware branch (one changed objective)
python -m real_robot.training.train_model_assisted_iql --latents wm_base_final --rollouts h1 \
    --run-name maiql_progress --progress

# 11 the robot
python -m real_robot.evaluation.robot_policy --iql maiql_base --replay 12   # dry run first
```

Preprocessing stages take `--episodes` (`all`, `training`, `diagnostic`,
`pilot`, `diagnostic:3`, `3,17,42`, `0-9`); `train`, `val` and `test` are
refused. Every stage takes `--set config.key=value` for any single setting.
A stage reuses an output only when the inputs it records are unchanged, and
makes it again otherwise; `--force` makes it again regardless.

Past-only graphs (`--mode past_only`) and deployment adaptation are deferred;
the code for them is kept but is not part of this path.

## Annotation

`full_episode` annotation makes **two main Gemini calls** per episode, both
over the whole episode, both cameras, at 15 fps:

1. **events** -- events, the active target, the outcome, and tracking anchors
   (a box and every named point, each point placed or marked hidden);
2. **relations** -- for every fact id, absolute intervals over every frame and
   temporal intervals from frame `K`, as two independent lists (a distance can
   close quickly inside one bin). It reports disagreements with the events
   pass instead of labelling around them.

Every request sends the fixed specification first, then the videos, then the
episode and pass text, so the specification and the videos form a shared
prefix. The prompt defines where things are measured (reference points, height
along the table normal, planar distance parallel to the table), that a label or
event starts at the first frame its condition holds, that attempts can fail or
recover, and when the target returns to the banana.

Validation (`graphs/validate.py`) turns every gap in the answers into an
issue, and nothing is valid until they are resolved:

| Stage | Checks | Repair |
|---|---|---|
| events | well-formed events, target, outcome | the events pass again, whole |
| anchors | frame-0 anchors for every entity and camera; every named point listed; anchors at every event; no gap over 15 frames for the gripper, a carried object, or anything the wrist camera sees; a table plane | only the missing keyframes, clipped when local |
| consistency | grasp/release/drop events against the grasp labels; placement and seating against contain/support; the completion and outcome against all of them; the active target against where the banana is; disagreements the relations pass reported | events, target, outcome and the facts involved, together, over the whole video |
| relations | legal labels, no conflicts, full coverage | only the listed facts over the listed frames, clipped when local |

Repairs run in that order and the annotation is rebuilt after each one, so
every request sees the current answers. Entries outside what a repair asked
for are rejected, not applied. An answer cut off at the output limit is asked
for again in halves; malformed JSON gets one more try; every failed response
is saved under `gemini_cache/failures`. An episode still invalid after
`repair_rounds` is saved with its issues and refused by the build. Each run
prints calls, input/cached/output tokens and the cost per usable episode,
repairs included.

## Geometry and reward

* The gripper's closing point is the midpoint of the two fingertips lifted
  separately with the depth at each finger's own surface; the space between
  open fingertips is table or object, so the depth there is never used.
* One focal length and one camera alignment are shared by all episodes, so
  `estimate_geometry all` checks that the high camera did not move first, and
  alignment and measurement refuse an episode whose camera moved or was never
  checked.
* `reward.lid_seated.unknown_geometry` says what an unmeasurable lid means:
  `not_seated` (default) or `defer_to_label`.
* Frames whose stage score lacks its distance are counted overall, per stage
  and as the longest run. Too many (`reward.fallbacks`), a Gemini success that
  labels and gripper cannot verify, or any other critical reward check keeps
  the episode out of the dataset.
* `check_bins` compares the centimetre bins, which Gemini estimated from RGB,
  with measured distances per label, and flags labels whose measured median
  falls outside their range.

## The action mapping

Numbers cannot certify what the action field commands: a position target and
a hindsight label both track a later state, and a gripper squeezing an object
breaks the similarity of a perfectly good command. So the mapping is declared
in `configs/action_mapping.yaml` — command dimensions, representation, a unit
per dimension, the gripper's action and state indices and its open and closed
values — each claim with its evidence (`metadata`, `documentation`,
`recording_code`, `hardware`, `author`, or `measurement`).

`audit_dataset` measures the recordings and checks them against the
declaration. `action_spec.json` is `verified` only when the declaration is
confirmed by a named person, no unit is `unknown`, the command and gripper
claims each cite evidence other than measurement, the videos and timestamps
are intact, and no measurement contradicts the declaration — unless that
contradiction is listed under `accepted_contradictions` with a reason.
`build_dataset` refuses anything else; there is no override flag.

The file ships with what the dataset itself shows (metadata names, the
frame-by-frame findings) and is not yet verifiable: the metadata records no
units and names the fields "displacement", and nothing but measurement speaks
for the representation or the gripper values. Supply that evidence — the
recording code, the dataset's author, or a hardware replay — and confirm.

## Policy training

`train_model_assisted_iql` is the policy path; no recorded-only policy is
trained first. Each update follows IQL's reference order:

| Step | Learns from |
|---|---|
| 1. value (expectile regression onto the target critic) | recorded transitions |
| 2. values recomputed with the updated value network | — |
| 3. actor (advantage-weighted likelihood, IQL's clipped weights, no other term) | recorded actions |
| 4. critics (onto `r + γ c V(z')` with the updated value) | recorded + imagined (10%) |
| 5. target critic (Polyak) | — |

The behaviour policy in `rollouts.yaml` is an auxiliary imitation model that
proposes actions for imagined rollouts. It is not the IQL actor: it adds no
loss to it, does not initialise it, and is not a benchmark. Imagined
transitions start at recorded latent states, take a sampled behaviour-policy
action (zero *additional* noise; the sample itself is stochastic), and take
their next state, reward and continuation from the frozen world model. No
recorded future graph or geometry is attached to them. One step and a 10%
critic share are starting restrictions, not accuracy guarantees.

The progress branch shares the world model, the latent cache, the task reward
and the imagined transitions; it fits its progress head on every recorded
transition, measures it on the diagnostic rows, uses the same value-before-actor
order, and draws networks, batches and the head from separate seeds so the
base initialisation and batches do not move. The task reward (the dense
replacement for an environment reward) and progress shaping stay separate
signals throughout.

Policy diagnostics — action error, TD error, value statistics, advantage
weights, and recorded and imagined critic errors separately — are logged, not
used for selection.

## Checkpoints

| File | Meaning |
|---|---|
| `final.pt` | the run's output and the default everywhere downstream |
| `latest.pt` | for `--resume` |
| `step_XXXXXXXX.pt` | periodic snapshots, kept |
| `best_diagnostic.pt` (world model only) | lowest diagnostic model loss, on episodes that were trained on; labelled, with `best_diagnostic.json`; not evidence of generalisation |

No policy checkpoint is selected by action error.

## What is fixed, and what refuses

Each artifact records the identity of everything it depends on, and each
consumer compares it before reading arrays:

| Artifact | Records |
|---|---|
| `source.json` | repo id, the resolved commit sha, a checksum per file |
| `selections/selection_v1.json` | every episode as training, the diagnostic ids, how they were chosen |
| `audit/action_spec.json` | the declared mapping and its hash, who confirmed it, the checks, the normaliser over all episodes |
| `spec/graph_spec.json` | the frozen bins, hashed, tied to `graph.yaml` |
| `annotations/<mode>/…json` | its inputs (graph, bins, prompts and schemas, Gemini settings, prepared videos, validation rules), issues, repairs, calls and tokens |
| `tracks/…json` | the annotation file, tracking settings and source videos it was made from |
| `depth/…json`, `camera_check.json` | video, Depth Pro weights, focal length; whether the camera stayed fixed |
| `geometry/…json`, `camera_alignment.json` | annotation, tracks, depth, alignment and settings; scene frame |
| `reward/reward_scales.json` | fitted scales, and the annotation and geometry of every episode they came from (all of them, or the build refuses) |
| `datasets/<name>/manifest.json` | all of the above plus the selection, vocabulary, action transform and image transform; each packed episode's inputs |
| world-model checkpoints | the dataset identity and content, the selection, whether the dataset was partial |
| `latents/<name>/identity.json` | checkpoint file SHA-256 and weight digest, dataset, annotation, reward and scales, action mapping and normaliser, latent convention, selection |
| `rollouts/<name>/rollouts.json` | the latent cache's compatibility record, the behaviour policy's hash, the settings |
| policy checkpoints | the latent compatibility record, the imagined-transition set, IQL settings, seeds |

Loading refuses, naming every field that disagrees, rather than warning.
A corrected annotation therefore makes its tracks, geometry, reward scales and
packed episode stale: each is made again when its stage runs, the build refuses
an episode whose chain is not current, and a world model trained before a
rebuild refuses the rebuilt dataset.
Changing the world model's weights — another checkpoint, or the same run
trained further — requires a new latent cache and new imagined transitions:
the file hash and the weight digest are both checked, so a matching
architecture and configuration are not enough. Cache and rollout directories
are never rewritten under a different contract, and artifacts from the earlier
split-based layout are refused by format.

## Annotation modes

`full_episode` (the default and the mode used throughout this path) gives
**retrospective** annotations: a robot cannot have them, and every record says
which mode produced it. `past_only` asks for the state at every fifth frame
from a clip that ends at that frame. It is deferred, with deployment
adaptation.

## Costs worth knowing before you start

* **Gemini.** Video costs 258 tokens per frame on Gemini 2.5 at default media
  resolution and 66 at `low`; Gemini 3 models use 70 by default. With two
  cameras at 15 fps, one main call over the mean 280-frame episode is about
  145k video tokens on 2.5 (37k at `low`, 198k for the longest episode) plus
  the specification, and an episode takes two main calls: 300 calls for the
  dataset before repairs. Local repairs send clips; reconciliation sends the
  whole episode. Whether prefix caching discounts the second call shows in the
  printed cached-token counts -- check them on the pilot before assuming it.
  `define_bins propose` samples its three episodes at `bins.video_fps` (5), about
  146k tokens on 2.5 at default resolution. Every response is cached on disk by a
  hash of the whole request, so re-running a stage -- including after a daily
  quota runs out -- repeats no call that was already answered.
* **Depth Pro.** One forward pass per high-camera frame, about 42k frames in
  total. Depth maps are cached at 240×320 in float16.
* **The pilot** (three episodes) is minutes of Gemini time and a few minutes of
  depth.

## Tests

```bash
python -m unittest discover -s real_robot/tests -t .
```

175 tests. The 141 that need only numpy run anywhere: the graph contract
against the repository's packer; interval validation; anchor coverage and the
consistency checks (a success whose banana never enters the pot is an issue to
reconcile, not a warning); the annotation flow with a scripted Gemini (two
main calls, repairs in order and on current answers, out-of-scope entries
rejected, cut-off answers split); the client's failure handling; artifact
reuse and dataset content; the reward's required behaviours, unknown geometry
and fallback limits; the episode selection; contiguous windows; the
action-mapping audit; geometry, including the fingertip midpoint; bin
grounding; prompt schemas and the progress schedule. The 34 that need torch and omegaconf
skip without them: the world model (including its diagnostics), IQL's update
order (the actor reads the updated value; imagined transitions reach only the
critics; the progress branch cannot move the base initialisation),
checkpoints, and a small end-to-end run through model-assisted IQL that also
checks the world model is unchanged and that mismatched checkpoints, caches
and rollouts are refused.

## Things the real data still has to settle

* The action mapping's evidence and confirmation (above). Until
  `action_spec.json` says `verified: true`, the build refuses.
* Whether the camera-to-robot alignment is accepted. If its median residual
  exceeds 3 cm, geometry falls back to a table-aligned camera frame and says so
  in every file; distances stay metric but are not robot coordinates.
* Whether Gemini's interval boundaries are frame-exact enough for the milestone
  labels. The overlays in stage 3 are how you check; `reward_checks.json` is
  how you check the consequences.
* Whether two main calls label as well as more, smaller ones would, and what a
  usable episode really costs with repairs. Run 3-5 representative episodes,
  read the overlays, anchors, event boundaries and reward traces, and the
  printed cost per usable episode, before annotating all 150.
* Whether the centimetre bins match the measured geometry (`check_bins`).
* The Gemini model id in `annotation.yaml` (`gemini-2.5-pro`) and the backend.
  The Interactions API has no temperature setting, so that backend needs
  `annotation.gemini.temperature: null`; `generate_content` stays the default.
