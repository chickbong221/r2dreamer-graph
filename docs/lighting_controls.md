# Lighting controls for Experiment C — what the stack exposes

Experiment C evaluates Experiment B's checkpoint under changed lighting. No
third model is trained, and **the conditions are not chosen here.** This is an
inventory of what can be varied and what each knob actually does, so the
choice is made against the mechanisms rather than against a guess.

## Why this needs care

`env.shader_dir` is the obvious-looking knob and the wrong one. It selects a
rendering pipeline (`minimal`, `default`, ray tracing), which changes shadow
treatment, material response and antialiasing together. Changing it would move
the pixels for several reasons at once, and an ablation whose independent
variable is "the renderer" cannot support a claim about illumination.

The prediction that makes C an ablation rather than a vague robustness check:
the graph is built from **segmentation and privileged poses**, so its facts
should be *bit-identical* across lighting conditions while the pixel encoder
degrades. That prediction only holds if lighting is the only thing that moved,
which is exactly why the knob has to be the right one.

## What is reachable from this repository's config surface

Confirmed present in `configs/env/mshab.yaml` and the MS-HAB construction path
in `envs/maniskill.py`:

| Control | What it is | Suitable as the C variable? |
|---|---|---|
| `env.shader_dir` | Rendering pipeline (`minimal` today) | **No** — changes more than illumination |
| `sensor_configs` (width/height) | Camera resolution | No — not illumination |
| `human_render_camera_configs` | Separate high-res render cameras | No — not what the policy sees |
| `eval_reconfiguration_frequency` | Whether scenes rebuild per reset | No, but relevant: a lighting change applied at build time needs a reconfigure to take effect |

**Nothing in this repository currently varies illumination.** There is no
lighting field in the env config, and `envs/maniskill.py` passes none.

## What has to be checked on the server

These live in ManiSkill/SAPIEN and ReplicaCAD, not here, so they need
inspecting where those packages are installed:

1. **Scene-builder lighting.** ManiSkill scene builders typically set lights
   in a `_setup_lighting` step. Whether ReplicaCAD's builder exposes intensity
   or colour, and whether MS-HAB's `SequentialTaskEnv` forwards anything, is
   the first question.
2. **Direct SAPIEN light manipulation.** `scene.add_directional_light`,
   `add_point_light` and ambient light are set at build time. Rescaling them
   after `reconfigure` is the most likely mechanism, and it is also the one
   that keeps geometry, physics and camera poses identical.
3. **Whether a light change survives `reconfiguration_freq = 0`.** Evaluation
   pins the scene set for the whole run; a lighting change applied during
   scene build would need either a reconfigure or a post-build adjustment.

Suggested inspection, to be run where mshab is installed:

```bash
python - <<'EOF'
import inspect
import mshab.envs
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder as B
src = inspect.getsource(B)
for key in ("light", "Light", "ambient", "shadow"):
    for i, line in enumerate(src.splitlines()):
        if key in line:
            print(f"{key:8s} {i:5d}  {line.strip()[:110]}")
EOF
```

## Decisions needing confirmation

1. **Which mechanism.** Post-build light rescaling is the only candidate that
   changes illumination alone; anything applied through the scene builder
   risks moving geometry or materials with it. Confirm after the inspection
   above.
2. **Which conditions.** "Dim" and "bright" are not quantities. They need to
   be multipliers on the scene's own light intensities (for example ×0.4 and
   ×2.0 relative to nominal), stated so the result is reproducible.
3. **How many.** Three points (nominal, dim, bright) is the minimum that shows
   a direction rather than a difference.

## Interface, prepared and unset

`configs/env/mshab_pick_c.yaml` exists, reuses B's scene and object, trains
nothing, and carries an explicit `PENDING` note where the conditions belong.
It is deliberately not runnable as a lighting experiment yet: an evaluation
that silently ran at nominal lighting three times would produce three numbers
and no ablation.

**Verification C must include regardless of the mechanism chosen** — with
matched seeds, between any two conditions:

- robot, object and camera poses identical;
- physics state identical;
- target identity identical;
- **symbolic graph facts identical** (this is the claim);
- RGB pixels materially different (this is the manipulation check — without
  it, "the graph was robust" may only mean nothing changed).
