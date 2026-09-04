# Where the structural-surface plane actually sits

A structural surface is measured as a plane, not as an origin. The plane is
`(anchor, outward_normal)` in the supporter's own object frame, mined by
`build_affordances` and read back by `surface_relative_height`.

Two halves, and only one of them is now exact.

## The normal: exact, and it was the blocker

`build_affordances` used to write `surface_normal: [0, 0, 1]` — the
supporter's local +z, on the assumption that furniture is z-up in its own
frame. ReplicaCAD furniture is **y-up**, which the mined anchors say plainly:
`kitchen_counter-0/body` records `[-0.137, 0.833, 0.253]`, and 0.833 m is the
height of a counter, sitting in Y.

Rotating a local +z into the world therefore produced a *horizontal* world
vector. `oriented_normal` refuses a horizontal normal — correctly, since a
vertical surface has no "above" — so `surface_height` returned `None`, every
structural height sample was discarded, no `ee-structural-surface-height-offset`
scale was calibrated, and `GraphBuilder` refused the asset for a bin nothing
had written. No exception, no warning, at any point in that chain.

The normal is now derived from evidence the collector already records: the
support contact force on the supporter points *into* the surface, so its
negation points away, and rotating that into the supporter's frame gives the
local normal. Each miner declares which way its stored vectors face
(`surface_normal_points`), so the shared helper no longer has to guess — and
it never consults world up about an object-frame vector again.

## The anchor: a proxy, self-consistent, and off by the supported object

`surface_anchor` is the **supported object's centre**, expressed in the
supporter's frame, averaged over the support samples. The supported object
rests on the surface, so its centre lies above the surface by roughly its own
half-height — 2–7 cm for the tidy_house YCB objects.

This does not create the failure the classification exists to remove. The same
stored plane is used when the height scale is mined *and* when a height is
labelled at runtime, so the offset cancels: the deadband still means one
thing, and `level` still means the same height everywhere. What it means is
that "level with the table" is centred on where a can sits, not on the bare
tabletop.

### What is missing to place it on the surface

The drop from the supported object's centre to its lowest point along the
normal is

```
drop = Σ_i | R_supported[:, i] · n_world | · h_i
```

where `h` is the supported object's collision half-extents. `h` **is**
recorded (`extents[key]["half_extents"]`), and both poses are recorded, so
this much is available.

What is not recorded is the **centre of that collision AABB in the object's
own frame**. `collision_half_extents_status` computes `lo` and `hi` over the
collision shapes and returns only `(hi - lo) / 2`, discarding `(hi + lo) / 2`.
Without it, `centre_world` has to be assumed equal to the body origin, which
is true only for objects whose collision bounds are centred on their origin.

So the surface anchor can be recovered to within that offset and no closer.
Adding it is one 3-vector per entity at collection time — a collector change
and a re-collection, not something a re-mine can recover.

**Pending decision.** Leave the anchor as the supported-object-centre proxy
(self-consistent, zero offset between mining and runtime, "level" centred a
few centimetres high), or record the AABB centre and re-collect. Nothing here
assumes an answer.
