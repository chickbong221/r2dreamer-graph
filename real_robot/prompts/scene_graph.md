<!-- SPEC -->
You are labelling one episode of a robot-manipulation dataset with its scene graph at every frame. Your labels become training data, so apply the fixed specification below exactly, and in the same way in every episode.

## Recording

- One SO-101 arm with a two-jaw gripper, teleoperated, recorded at {{FPS}} frames per second.
- Two synchronised cameras:
{{CAMERAS}}
- You are shown one recorded frame in {{STRIDE}}, {{VIDEO_FPS}} per second. Each shown frame has its camera, its recorded frame number and its time burned into the top-left corner, for example `TOP  F 0123/0471  t 004.10s`. Read frame numbers from that overlay; the same number is the same instant in both cameras.
- Every frame number you give is a recorded frame number. Intervals cover every recorded frame, including the ones you are not shown; a label starts at the first shown frame on which it holds.
- The final recorded frame is also included, even when it is off the sampling grid. Its playback spacing may differ from recorded time; use the burned-in frame number and time.
- The task text says what the operator tried to do; it is not evidence that it happened. An attempt can fail, stall, drop an object, retry or recover. Label only what the video shows.

## Entities

{{ENTITIES}}

## Where things are measured

- Spatial facts are measured between the entities' reference points, given above.
- Height is measured straight up from the tabletop. Position in an image is not height: in the top camera an object lower in the image is usually nearer the robot, not lower, and the wrist camera tilts with the gripper.
- Planar distance is measured parallel to the tabletop, ignoring height.
- `height-offset(src, dst)` is the height of `src`'s reference point minus that of `dst`'s.
- Approximate sizes, to use as rulers:
{{SIZES}}

## Facts

Each line is one fact: its id, its orientation `relation(src, dst)`, and the labels it may take.

{{FACTS}}

- `support` and `contain` name the holder in the label: `src-holds` means `src` supports or contains `dst`, `dst-holds` means `dst` supports or contains `src`, and `not-holds` means neither.
- {{UNOBSERVED}}
- If neither view supports a legal label, use JSON `null` for that interval (also for temporal change when either endpoint is unknown). This marks missing supervision, not a negative fact or stable motion. Do not invent metric precision from the approximate sizes.
- Temporal labels use a window of K = {{K}} frames ({{K_SECONDS}} s): the temporal label at frame t is the change of the fact's value from frame t - {{K}} to frame t. Frames 0 to {{K_MINUS_ONE}} have no temporal label. For distances a decrease means the gap is closing; for `height-offset` it means `src` moves down relative to `dst`.
- Absolute and temporal labels are independent: a distance can close quickly while staying inside one bin, so their boundaries need not coincide.
- A label starts at the first frame on which its condition holds. Later frames may make you certain of what happened, but they never move a boundary away from where the condition first held.

## Label definitions

{{LABELS}}

## Boxes

- A box is `[ymin, xmin, ymax, xmax]`, integers from 0 to 1000 relative to the image, tight around the visible part of the entity. For `table`, the visible part of the black mat.
- Give box keyframes for every entity in every camera: at frame 0, at the last frame, never more than {{BOX_EVERY}} frames apart, and on every frame where the entity becomes hidden or visible again in that camera.
- A keyframe is `{frame, visible, box_2d}`. When the entity is not visible in that camera (out of view or fully hidden), `visible` is false and `box_2d` is `[0, 0, 0, 0]`.
- Boxes are interpolated linearly between keyframes, so add keyframes wherever that would drift from where the entity really is.

<!-- EPISODE -->
## This episode

- Episode {{EPISODE}}: {{N_FRAMES}} recorded frames, numbered 0 to {{LAST_FRAME}}; you are shown {{SHOWN}} of them in each camera.
- The videos above are {{CAMERA_ORDER}}.
- What the operator tried to do: {{TASK}}

<!-- JOB -->
## Your job

Report:

{{ITEMS}}

<!-- ITEM target -->
`active_target`: contiguous intervals `{start, end, object}` covering every frame from 0 to {{LAST_FRAME}} exactly once. The active target is {{TARGET_RULE}}

<!-- ITEM facts -->
`facts`: for {{FACT_SCOPE}}, one entry `{fact, absolute, temporal}`. `absolute` is contiguous intervals `{start, end, label}` covering every frame from 0 to {{LAST_FRAME}} exactly once. `temporal` covers every frame from {{K}} to {{LAST_FRAME}} exactly once for the facts that have temporal labels ({{TEMPORAL_FACTS}}), and is an empty list for the others.

<!-- ITEM boxes -->
`boxes`: for {{BOX_SCOPE}}, one entry `{entity, camera, keyframes}` following the box rules.

<!-- ITEM notes -->
`notes`: anything that made this episode hard to label (occlusion, a failed attempt, a camera glitch), or an empty string.

<!-- REPAIR -->
## Corrections needed

Your earlier answer for this episode could not be used as it was. These problems were found:

{{ISSUES}}

Answer again only for the parts listed under "Your job", completely and following every rule above; your answer replaces the earlier one for those parts. Your earlier answer for them:

{{PREVIOUS}}
