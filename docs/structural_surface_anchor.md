# Structural-surface height: V1 decision and deferred refinement

The stored reference plane is `(anchor, outward_normal)` in the supporter's
object frame. Mining and runtime both use it. Agreement between those two
paths is necessary, but does not establish that the anchor is on the physical
surface.

## Approved V1 scope

For the first MS-HAB version, retain the existing supported-object-origin proxy
and the corrected object-frame/outward-normal conversion that made the assets
runnable. Exact physical-surface reconstruction and verification are deferred,
not a V1 training-readiness gate. Do not revert the coordinate-direction fix or
replace the current assets with a new geometric estimator.

The possible geometry-only recovery method below is retained for future work;
it is not an enabled runtime path or an extra collection requirement. No new
rollouts or re-mine are needed for this scope decision. Schedules, protected
nodes, and per-target membership are unchanged. V1 should describe the height
reference as an approximate mined support plane, not an exact tabletop plane.

## Normal-frame correction

The old MS-HAB miner wrote local `[0, 0, 1]` without checking the supporter's
orientation. For the affected furniture, that axis rotates to a horizontal
world direction. The height calculation then rejects it, leaving no structural
height samples and no corresponding bin scale. An anchor's coordinates alone
do not establish the axis convention of every furniture model.

The corrected miner negates the recorded force on the supporter, transforms it
into the supporter's frame, and declares that the resulting vector points
outward. ManiSkill declares the convention of its own evidence separately.
This fixes the frame/sign mismatch. A force-derived direction is an estimate
from contact evidence, not a proof of an exact geometric surface normal.

## The anchor remains a supported-object-origin proxy

The existing `surface_anchor` averages supported-object origins in the
supporter's frame. An actor origin is not necessarily its mass centre or its
collision-bounds centre. It can therefore sit above the real support surface,
by an amount depending on the object's geometry and orientation. The exact
bias for these assets has not been measured; a universal centimetre range is
not established.

If the proxy plane is displaced outward by `delta`, then
`height_proxy = height_surface - delta`. Calibrating and evaluating against
that same proxy makes the paths consistent; it does **not** cancel the physical
bias, especially for symmetric height bands centred at zero. A passing bin or
potential check does not validate the anchor's physical location.

## Why the recorded half-extents alone are insufficient

For an actual box with body-local centre `c`, half-extents `h`, supported-body
rotation `R`, body origin `p`, and outward unit normal `n` in world coordinates,
the lowest projected coordinate is

```
n dot (p + R c) - sum_i(abs(n dot R[:, i]) * h[i])
```

The existing evidence supplies poses and half-extents, but not a verified `c`.
There is a second limitation in the current geometry reader:

- `_shape_half_size` reduces mesh vertices to half-extents, losing their local
  centre before the shapes are combined.
- `collision_half_extents_status` adds shape translation but does not rotate
  the bounds by the shape's local quaternion.

Consequently, merely returning the final `(lo + hi) / 2` from that helper would
not reliably recover mesh-centre offsets or rotated shapes. An AABB is also
only a conservative approximation for a non-box at arbitrary orientation.
Do not silently repurpose this classification helper as an exact contact-plane
estimator.

## Deferred: geometry-only recovery before recollection

The inspected local ManiSkill source offers a recovery route:

1. `mani_skill/utils/building/actors/ycb.py` loads a model-specific
   `collision.ply`, with scale taken from `info_pick_v0.json`.
2. `mani_skill/utils/scene_builder/replicacad/rearrange/scene_builder.py` builds
   YCB instances through that builder. Its initialization configurations change
   poses; the inspected path does not randomize each instance's mesh.

This supports inspecting geometry once per model/scale and combining it with
the existing recorded poses. It does **not** yet prove that this checkout and
the collection server have identical assets or collision decomposition.

Before changing anchor estimation, use the server's installed version to:

1. Match each supported actor to its model, scale and collision asset. Record
   file hashes and compare recovered dimensions with the saved extents. Do not
   assume every supporting event involves one of the nine training targets.
2. Read the actual collision geometry, including shape scale, local translation
   and rotation. Prefer the simulator's cooked collision shapes when possible;
   the source mesh alone may differ from its convex decomposition.
3. Transform the vertices into the body frame. For mesh vertices `v`, the exact
   directional minimum of that geometry is
   `n dot p + min_v((R.T n) dot v)`. Use analytic support functions for primitives.
4. Combine that minimum with the existing support poses and independently check
   resting contacts against the supporter geometry, allowing for simulator
   contact offsets and penetration. Report the proxy-to-surface displacement
   per supporter and the spread across supported objects, not just an average.

The collision asset files and simulator are not available in this local
workspace, so the geometry recovery remains unverified and outside V1. No
collection, re-mining, or physical-anchor substitution is performed by this repair.
If matched geometry can be recovered, an offline re-mine can use the existing
rollouts; repeating the successful-policy collection is not inherently needed.
If it cannot, state which geometry/provenance is missing before proposing any
additional collection. Any future replacement of V1's proxy requires an explicit
decision and consistent re-calibration of mining and runtime measurements.
