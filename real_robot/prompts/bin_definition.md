You are defining the label specification that will be used to annotate an entire robot-manipulation dataset with scene graphs. The same specification will be applied, unchanged, to every episode by many independent annotation calls. It must therefore describe physical situations that look the same in every episode, not ranges relative to one episode's motion.

## Dataset

- Robot: one ALOHA arm (the right arm) with a parallel two-finger gripper, teleoperated.
- Recording rate: {{FPS}} frames per second. The videos you are shown are sampled at {{VIDEO_FPS}} frames per second; judge motion, not exact frames.
- Cameras, synchronised frame for frame:
{{CAMERAS}}
- Every frame has its camera name, frame number and time burned into the top-left corner (for example `HIGH  F 0123/0280  t 008.20s`). Frame numbers are identical across the cameras of one episode.
- Task the operator attempted, identical in every episode: {{TASK}}. Attempts can fail or recover; the specification must cover what happens, not only success.

You are shown {{N_EPISODES}} representative episodes of the dataset (each with all cameras):
{{EPISODES}}

## Entities

{{ENTITIES}}

## Where things are measured

The annotators, and a later geometric check, measure every spatial fact between these reference points:
{{REFERENCE_POINTS}}

- Height is measured straight up from the tabletop, perpendicular to the table surface. Position in an image is not height.
- Planar distance is the distance between the two reference points measured parallel to the tabletop, ignoring height.

## Relations and their fixed label vocabularies

These vocabularies are fixed by the model that will read the graphs. Use exactly these label strings; do not add, rename or merge labels.

{{RELATIONS}}

Direction conventions (these are fixed too):

- A fact `relation(src, dst)` is always read from `src` to `dst`.
- `height-offset(src, dst)` describes the height of `src`'s reference point minus the height of `dst`'s. `height-offset(ee, banana) = above` means the gripper is higher than the banana.
- `support` and `contain` name the holder in the label: `src-holds` means `src` supports/contains `dst`; `dst-holds` means `dst` supports/contains `src`; `not-holds` means neither.
- The compatibility relations score how well two things are positioned to perform an interaction (grasping, contacting, one resting on the other, one inside the other). `match` is ready, `partial-match` close but not yet correct, `poor-match` wrong.

When a label may be `unobserved`:
{{UNOBSERVED}}

Temporal labels describe how a value changed over a fixed window of K = {{K}} frames ({{K_SECONDS}} s): the label at frame t compares the value at frame t with the value at frame t - {{K}}. `decrease-fast`, `decrease-slow`, `stable`, `increase-slow`, `increase-fast`. For distances, decrease means the gap is closing. For height-offset, decrease means `src` is moving down relative to `dst`. For compatibility relations the value is the degree of mismatch, so decrease means the fit is getting better.

## What to produce

1. Reference dimensions. Estimate the real sizes of the objects and the gripper from the videos (banana length and thickness, pot inner diameter and height, lid diameter and handle height, gripper finger length and opening). These are estimates from RGB video, not measurements; say what visual cue each one is read from. They anchor every threshold below, and annotators will use them as visual rulers.
2. Spatial bins, in centimetres, separately for the `ee-object` and `object-object` scopes wherever a relation is used in both, measured between the reference points above. For each label give a meaning an annotator can check by eye and a `lower_cm`/`upper_cm` range. Planar-distance ranges are unsigned, start at 0, are contiguous and increase from `very-near` to `very-far`. Height-offset ranges are signed (`src` minus `dst`), contiguous, and increase from `far-below` to `far-above`; `level` straddles zero. Use -1000 or 1000 for an open end. Choose thresholds so that the task's important situations fall in different bins (for example: fingers around the banana versus hovering above it; banana above the pot opening versus beside the pot; lid resting on the rim versus held a few centimetres above it).
3. Physical rules for `contact`, `grasp`, `support` and `contain`: what visual evidence makes each label true, which camera is decisive (the wrist camera usually decides whether the fingers hold an object), and -- only for the relations whose vocabulary has it -- when a label is `unobserved`. A label starts at the first frame its condition holds.
4. Compatibility bins: the meaning of `match`, `partial-match`, `poor-match` and `unobserved` for each compatibility relation, concretely for this task (grasping the banana or the lid handle; the banana inside the pot; the lid seated on the pot rim).
5. Temporal bins over K = {{K}} frames: for planar-distance and height-offset, the change in centimetres over the window that separates `stable` from `slow` and `slow` from `fast`, as signed ranges from `decrease-fast` to `increase-fast`; for compatibility, a meaning for each change label.
6. General rules that keep annotation consistent across episodes (occlusion, motion blur, which camera wins when they disagree, what to do at the boundary between two bins, failed or repeated attempts).

Do not normalise any threshold to a particular episode. A banana 3 cm from the gripper must receive the same label in every episode.
