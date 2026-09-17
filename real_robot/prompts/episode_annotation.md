<!-- SPEC -->
You are annotating episodes of a robot-manipulation dataset with scene-graph labels. Your labels become training data for a model, so following the fixed specification below exactly, in every episode, matters more than anything else.

## Recording

- One ALOHA arm (the right arm) with a parallel two-finger gripper, teleoperated, recorded at {{FPS}} frames per second.
- Cameras, synchronised frame for frame:
{{CAMERAS}}
- Every frame has its camera name, frame number and time burned into the top-left corner (for example `HIGH  F 0123/0280  t 008.20s`). Always read frame numbers from that overlay. The same frame number is the same instant in every camera.
- Each episode is one attempt at a task. An attempt can fail, stall, drop an object, retry or recover. The task description says what the operator tried to do; it is never evidence that anything happened. Label only what the video shows.

## Entities

{{ENTITIES}}

## Where things are measured

Spatial labels describe the physical scene, not the image.

- Every spatial fact is measured between these reference points:
{{REFERENCE_POINTS}}
- **Height** is measured straight up from the tabletop, perpendicular to the table surface. Position in an image is not height: in the high camera an object lower in the image is usually nearer the camera, not lower, and the wrist camera tilts with the gripper.
- **Planar distance** is the distance between the two reference points measured parallel to the tabletop, ignoring any difference in height.
- `height-offset(src, dst)` is the height of `src`'s reference point minus the height of `dst`'s.
- The sizes and centimetre thresholds below are estimates. Use the reference dimensions as rulers, and judge by the objects' real sizes, not by how large they appear in an image.

## Facts

Each line is one fact: its id, its stored orientation `relation(src, dst)`, and the labels it may take.

{{FACTS}}

Label conventions:

- `support`/`contain`: `src-holds` means `src` supports/contains `dst`; `dst-holds` means `dst` supports/contains `src`; `not-holds` means neither. For example `contain(banana, pot) = dst-holds` means the banana is inside the pot, and `support(lid, pot) = dst-holds` means the lid rests on the pot.
{{UNOBSERVED}}
- Temporal labels use a window of K = {{K}} frames ({{K_SECONDS}} s). The temporal label at frame t describes the change in the fact's value from frame t - {{K}} to frame t, judged over that window only. Frames 0 to {{K_MINUS_ONE}} have no temporal label.
- Absolute and temporal labels are separate: a distance can be closing quickly while it stays inside one bin, so their boundaries need not coincide.
- **Boundaries.** A label, and an event, starts at the first frame on which its condition holds. Later frames may make you certain of what happened, but they never move a boundary away from the frame where the condition first held.

Examples of the distinctions that matter most:

- The fingers moving around the banana, or touching it without closing on it: `grasp(ee, banana) = not-holds`. From the first frame the fingers are closed on the banana and holding it: `holds` -- even if you only become certain when it lifts a few frames later.
- The banana above the pot's opening, or resting on the rim: `contain(banana, pot) = not-holds`. From the first frame the banana is at least partly below the rim, inside the pot: `dst-holds`.
- The lid touching the rim while still tilted or carried: `contact(lid, pot) = holds` but `support(lid, pot) = not-holds`. From the first frame the lid sits on the rim with the pot bearing it: `support(lid, pot) = dst-holds`.
- The gripper closing from 10 cm to 7 cm on the banana while that whole range lies in one distance bin: the absolute label is the same on every one of those frames, and the temporal label reports the decrease.

## Named points

Boxes are `[ymin, xmin, ymax, xmax]` and points `[y, x]`, integers from 0 to 1000 relative to the image. Place each point exactly on the described location:
{{POINTS}}

## Frozen label specification

Apply these definitions exactly, in every frame. Do not adapt them to an episode.

{{BIN_SPEC}}

<!-- EPISODE -->
## This episode

- Episode {{EPISODE}}: {{N_FRAMES}} frames, numbered 0 to {{LAST_FRAME}}.
- The videos above show frames {{VIDEO_START}} to {{VIDEO_END}}, `high` first.
- What the operator tried to do: {{TASK}}

<!-- PASS events -->
## Your job: events, active target, outcome and tracking anchors

Watch the whole episode in both cameras, then report:

1. `events`, each at the first frame its condition holds, with its object and a few words of `evidence` naming the camera that shows it:
   - `grasp`: the fingers are closed on the object and holding it.
   - `release`: the fingers have opened and let go of the object.
   - `drop`: the object falls out of the gripper unintentionally.
   - `place`: the banana is inside the pot. For the lid, report `lid_seated` instead.
   - `lid_seated`: the lid sits on the pot rim with the pot bearing it.
   - `task_complete`: the banana is in the pot, the lid is seated on the pot and the gripper has let go of the lid (object: `lid`).
   Report every occurrence: an object grasped twice has two grasp events.
2. `active_target`: contiguous intervals covering every frame from 0 to {{LAST_FRAME}} exactly once. The target is `banana` until the banana rests inside the pot with the gripper no longer holding it, and `lid` from then on, including after the task is complete. If the banana leaves the pot again before the task is complete, the target is `banana` again until it is back inside and released.
3. `outcome`: `success` is true only if `task_complete` happens within the recording, and `completion_frame` is its frame (-1 otherwise). `banana_in_pot_at_end` and `lid_closed_at_end` describe frame {{LAST_FRAME}}. If the attempt is unsuccessful, give the reason.
4. `keyframes`: anchors that initialise and correct a tracker. Each keyframe gives the frame, the camera, the entity, whether it is `visible` in that camera, its `box_2d` (`[0, 0, 0, 0]` if it is not visible), and every named point of that entity -- each either `visible: true` with its `point`, or `visible: false` with point `[0, 0]` when that part is hidden. Provide keyframes:
   - at frame 0 for every entity in every camera, visible or not;
   - at every event frame, for the event's object and for `ee`, in every camera;
   - for `ee` in every camera, never more than {{KEYFRAME_EVERY}} frames apart;
   - for any entity that moves in an image, never more than {{KEYFRAME_EVERY}} frames apart in that camera. Motion in the image counts whatever causes it: a carried object moves in every camera, and everything the wrist camera sees moves whenever the arm moves;
   - whenever an entity becomes hidden or visible again in a camera;
   - for `table` in the `high` camera, at least one keyframe with three or more surface points visible, placed on bare table far apart from each other and not along a line: they define the table plane.

<!-- PASS relations -->
## Your job: absolute and temporal labels for every fact

The events, active target and outcome already reported for this episode:

{{CONTEXT}}

For {{FACT_SCOPE}}, report under `facts`:

- `absolute`: contiguous inclusive intervals `{start, end, label}` that cover every frame from 0 to {{LAST_FRAME}} exactly once;
- `temporal`: for the facts that have temporal labels ({{TEMPORAL_FACTS}}), intervals that cover every frame from {{K}} to {{LAST_FRAME}} exactly once; an empty list for every other fact.

Put each boundary at the first frame where the new label holds, and set absolute and temporal boundaries independently.

If the video shows that a reported event, the active target or the outcome is wrong, do not label around the error and do not correct it silently: add an entry to `pass1_disagreements` with its kind, the frames concerned and what the video shows. Label the facts as the video shows them.

<!-- REPAIR events -->
## Corrections needed: events, active target, outcome and anchors

Your previous answer to this pass could not be used as it was. These problems were found:

{{ISSUES}}

Return a complete corrected answer, following every rule above. Your previous answer:

{{PREVIOUS}}

<!-- REPAIR anchors -->
## Corrections needed: tracking anchors

These anchors are missing or incomplete:

{{ISSUES}}

Return `keyframes` only for these entities, cameras and frames, following the keyframe rules above:

{{ANCHOR_REQUESTS}}

Within a range, place keyframes so that no two consecutive keyframes of that entity in that camera are more than {{KEYFRAME_EVERY}} frames apart, and include a keyframe wherever the entity becomes hidden or visible again. Keyframes for anything else are ignored.

<!-- REPAIR relations -->
## Corrections needed: labels of some facts

The events, active target and outcome for this episode:

{{CONTEXT}}

These problems were found in the labels:

{{ISSUES}}

Report `facts` only for {{FACT_LIST}}: `absolute` intervals covering exactly frames {{RANGE_START}} to {{RANGE_END}} and, for facts with temporal labels, `temporal` intervals covering exactly frames {{TEMPORAL_START}} to {{RANGE_END}} (an empty list otherwise). They replace every earlier label of these facts on those frames; all other labels are kept, and entries for other facts or frames are ignored. If the video shows that a reported event, the active target or the outcome is wrong, say so in `pass1_disagreements`. The current labels of these facts on those frames:

{{PREVIOUS}}

<!-- RECONCILE -->
## Contradictions to resolve

Parts of the annotation of this episode contradict each other:

{{ISSUES}}

Watch the frames concerned in both cameras and decide from the video which parts are wrong. Correct whichever answer the video contradicts and keep what it confirms. Return:

- `events`, `active_target` and `outcome`: complete answers for the whole episode, following the rules for events above;
- `facts` for {{FACT_LIST}}: `absolute` intervals covering exactly frames {{RANGE_START}} to {{RANGE_END}} and, for facts with temporal labels, `temporal` intervals covering exactly frames {{TEMPORAL_START}} to {{RANGE_END}} (an empty list otherwise). They replace the current labels of these facts on those frames.

The current answers:

{{PREVIOUS}}

<!-- PASS past_only -->
## Your job: the state at frame {{FRAME}}, using only the past

The video you are shown ends at frame {{FRAME}}. You must not use anything that happens after it; label the scene exactly as it is at frame {{FRAME}}, using only what the frames up to {{FRAME}} show.

Your own answer for the previous update (frame {{PREVIOUS_FRAME}}) is below. It was also made without seeing the future. Stay consistent with it unless the new frames show it was wrong.

{{PREVIOUS}}

Report:

- `frame`: {{FRAME}}.
- `active_target`: `banana` until the banana rests inside the pot with the gripper no longer holding it, then `lid`; `banana` again if it has left the pot before the task is complete.
- `facts`: one entry for every fact in the specification, with its absolute `label` at frame {{FRAME}} and its `temporal_label` for the change from frame {{FRAME_MINUS_K}} to frame {{FRAME}} (use `none` if {{FRAME}} is less than {{K}}, or if the fact has no temporal label).
- `objects`: for every entity and every camera, whether it is visible at frame {{FRAME}}, its `box_2d` (`[0, 0, 0, 0]` if not visible) and every named point, visible with its `point` or `visible: false`.
- `events_so_far`: every event (grasp, release, place, drop, lid_seated, task_complete) that has happened at or before frame {{FRAME}}, with its frame number and object.
- `task_complete`: whether the task has been completed at or before frame {{FRAME}}.
- `banana_in_pot`, `lid_closed`: the state at frame {{FRAME}}.
