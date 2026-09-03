"""What the MS-HAB miner now derives from schema-v9 evidence.

Three things the old MS-HAB assets do not carry and the Pick schedule cannot
run without: which members are extended support planes, which end-effector
height family each member is measured on, and the reviewed rest-site
declaration plus the vocabulary row its runtime node needs.

The classification rules are shared with the ManiSkill miner on purpose. Two
copies would drift into two meanings for one token, and ``level`` has to mean
one thing per scale.
"""

import json
import tempfile
import unittest
from pathlib import Path

from scenegraph.core import families as rules
from scenegraph.core.affordance import load_affordance_set
from scenegraph.core.sites import (
    SITE_EE_REST,
    admit_site_members,
    declared_sites,
    parse_site_declarations,
)
from scenegraph.core.spatial_metrics import (
    EE_SITE_HEIGHT_KEY,
    EE_SITE_PLANAR_KEY,
    FAMILY_GOAL_MARKER,
    FAMILY_MANIPULAND,
    FAMILY_RECEPTACLE,
    FAMILY_STRUCTURAL,
)
from scenegraph.core.whitelist import derive_bin_edges
from scenegraph.tools.build_subtask_whitelists import _WhitelistBuilder

BOWL = "actor:024_bowl"
COUNTER = "link:kitchen_counter-0/body"
TABLE = "actor:frl_apartment_table_01"
SITES_DIR = Path("scenegraph/configs/sites")


def _member(itypes, kind="actor"):
    return {"roles": ["interacted"], "interaction_types": sorted(itypes),
            "kind": kind}


def _extent(half):
    return {"half_extents": list(half), "extent_status": "ok"}


class StructuralSurfaceTest(unittest.TestCase):
    """Both conditions necessary: it supports something, and it is wide."""

    MEMBERS = {
        COUNTER: _member(["contact", "support"], "link"),
        BOWL: _member(["contact", "grasp"]),
    }

    def test_a_wide_supporter_is_a_surface(self):
        found = rules.structural_surfaces(
            {COUNTER: _extent([0.9, 0.4, 0.45])}, self.MEMBERS)
        self.assertEqual(sorted(found), [COUNTER])

    def test_the_reason_names_the_measurement(self):
        found = rules.structural_surfaces(
            {COUNTER: _extent([0.9, 0.4, 0.45])}, self.MEMBERS)
        self.assertIn("0.400m", found[COUNTER])

    def test_a_narrow_supporter_is_not(self):
        """A bin and a tabletop carry identical roles; size is the only
        discriminator."""
        self.assertEqual(
            rules.structural_surfaces(
                {COUNTER: _extent([0.9, 0.1, 0.45])}, self.MEMBERS), {})

    def test_a_wide_thing_that_supports_nothing_is_scenery(self):
        members = {TABLE: _member(["contact"])}
        self.assertEqual(
            rules.structural_surfaces({TABLE: _extent([2.0, 2.0, 0.4])},
                                      members), {})

    def test_an_unreadable_supporter_is_left_unclassified_not_small(self):
        """A missing measurement is not evidence of absence, and a table
        quietly demoted reinstates the metre of origin error."""
        extents = {COUNTER: {"extent_status": "no-collision-shapes"}}
        self.assertEqual(rules.structural_surfaces(extents, self.MEMBERS), {})
        self.assertEqual(rules.unclassified_supporters(extents, self.MEMBERS),
                         [COUNTER])

    def test_a_classified_supporter_is_not_reported_as_unclassified(self):
        self.assertEqual(
            rules.unclassified_supporters(
                {COUNTER: _extent([0.9, 0.4, 0.45])}, self.MEMBERS), [])


class FamilyRuleTest(unittest.TestCase):
    """Strict precedence, and no silent fallback."""

    def test_a_structural_surface_wins_over_every_later_rule(self):
        members = {COUNTER: _member(["contact", "support"], "link")}
        families = rules.object_families(
            members, holders={COUNTER}, supported=set(), structural={COUNTER})
        self.assertEqual(families[COUNTER], FAMILY_STRUCTURAL)

    def test_a_grasped_object_is_a_manipuland(self):
        families = rules.object_families(
            {BOWL: _member(["contact", "grasp"])}, set(), set(), set())
        self.assertEqual(families[BOWL], FAMILY_MANIPULAND)

    def test_a_holder_that_is_not_wide_is_a_receptacle(self):
        families = rules.object_families(
            {COUNTER: _member(["contact", "support"], "link")},
            holders={COUNTER}, supported=set(), structural=set())
        self.assertEqual(families[COUNTER], FAMILY_RECEPTACLE)

    def test_something_supported_that_holds_nothing_is_a_manipuland(self):
        """PullCubeTool grasps the tool, never the cube."""
        families = rules.object_families(
            {"actor:cube": _member(["contact", "support"])},
            holders=set(), supported={"actor:cube"}, structural=set())
        self.assertEqual(families["actor:cube"], FAMILY_MANIPULAND)

    def test_a_declared_actor_backed_site_is_a_goal_marker(self):
        """PickCube's ``actor:goal_site``: a real actor with no collision
        geometry, named by a reviewed declaration."""
        families = rules.object_families(
            {"actor:goal_site": _member([])}, set(), set(), set(),
            declared_sites={"actor:goal_site"})
        self.assertEqual(families["actor:goal_site"], FAMILY_GOAL_MARKER)

    def test_an_undeclared_member_with_no_interactions_is_ambiguous(self):
        """The trap: a behaviour-free counter used to reach this rule and be
        labelled a goal marker -- silently, because the rule returned a
        family rather than None."""
        families = rules.object_families(
            {COUNTER: _member([], "link")}, set(), set(), set())
        self.assertIsNone(families[COUNTER])
        self.assertEqual(rules.ambiguous_families(families), [COUNTER])

    def test_an_unclassifiable_member_gets_no_family(self):
        """Took part in interactions but is neither grasped, nor a holder, nor
        a surface. A fallback would lend it another family's deadband."""
        families = rules.object_families(
            {"actor:x": _member(["contact"])}, set(), set(), set())
        self.assertIsNone(families["actor:x"])
        self.assertEqual(rules.ambiguous_families(families), ["actor:x"])

    def test_bucket_names_split_into_the_two_sides(self):
        """The ManiSkill miner keys directed evidence by string."""
        holders, supported = rules.directed_pairs(
            ["support/link:counter/actor:bowl", "contain/actor:bin/actor:ball",
             "contact/actor:a/actor:b"])
        self.assertEqual(holders, {"link:counter", "actor:bin"})
        self.assertEqual(supported, {"actor:bowl", "actor:ball"})

    def test_both_evidence_shapes_read_the_same_extents(self):
        """ManiSkill stores a bare list; MS-HAB a record with a status."""
        members = {COUNTER: _member(["contact", "support"], "link")}
        bare = rules.structural_surfaces({COUNTER: [0.9, 0.4, 0.45]}, members)
        keyed = rules.structural_surfaces(
            {COUNTER: _extent([0.9, 0.4, 0.45])}, members)
        self.assertEqual(bare, keyed)


def _affordance_set(with_plane=True):
    """An asset carrying the counter's mined support plane.

    A structural surface is measured against that plane, so the miner refuses
    to calibrate one without it -- the alternative is the actor origin, ~0.9m
    below a counter's top.
    """
    payload = {"_schema_version": 4, "objects": {
        BOWL: {"grasp_components": []},
        COUNTER: {"support_components": [
            {"surface_anchor": [0.0, 0.0, 0.45],
             "surface_normal": [0.0, 0.0, -1.0], "partner": BOWL},
        ]},
    }}
    if with_plane:
        payload["objects"][COUNTER]["reference_surface"] = (
            rules.reference_surface_from_supports(
                payload["objects"][COUNTER]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aff.json"
        path.write_text(json.dumps(payload))
        return load_affordance_set(str(path))


class MinerPayloadTest(unittest.TestCase):
    """One rollout through the MS-HAB builder, end to end."""

    def _builder(self, sites=None, with_plane=True):
        return _WhitelistBuilder(
            "pick", BOWL, task_group="tidy_house", sites=sites or {},
            affordance_set=_affordance_set(with_plane))

    def _rollout(self):
        return {
            "target_key": BOWL,
            "interacted": [
                {"key": BOWL, "kind": "actor", "name": "env-0_024_bowl-0",
                 "grasped": True, "max_ee_force": 3.0},
            ],
            "supports": [
                {"supported_key": BOWL,
                 "supporter": {"key": COUNTER, "kind": "link",
                               "name": "env-0_body"}},
            ],
            "extents": {
                BOWL: _extent([0.08, 0.08, 0.04]),
                COUNTER: _extent([0.95, 0.42, 0.45]),
            },
            # tcp_pose plus every member's pose, keyed by canonical key.
            # Enough to reproject each height onto its member's family scale
            # after classification, which is why no new collector format is
            # needed for per-family calibration.
            "pose_samples": [
                {"tcp_pose": [0.0, 0.0, z] + [1.0, 0.0, 0.0, 0.0],
                 "entities": {
                     BOWL: {"pose": [0.0, 0.0, 0.46, 1.0, 0.0, 0.0, 0.0],
                            "kind": "actor", "name": "env-0_024_bowl-0"},
                     COUNTER: {"pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               "kind": "link", "name": "env-0_body"},
                 }}
                # More than temporal K snapshots, so the change stream has a
                # window to difference over.
                for z in (1.10, 1.00, 0.90, 0.80, 0.66, 0.55, 0.48)
            ],
            "bin_samples": {
                "ee_object_planar_distance": [0.4, 0.2],
                # One pooled stream for every partner -- the target and the
                # counter alike. That pooling is the gap pinned below.
                "ee_object_height_offset": [0.02, 0.55],
                "ee_site_planar_distance": [0.9, 0.1],
                "ee_site_height_offset": [0.3, 0.02],
            },
        }

    def _payload(self, sites=None):
        builder = self._builder(sites)
        builder.absorb(self._rollout())
        return builder.payload()

    def test_extents_are_absorbed(self):
        builder = self._builder()
        builder.absorb(self._rollout())
        self.assertEqual(sorted(builder.extents), sorted([BOWL, COUNTER]))

    def test_the_wide_supporter_is_flagged_and_reasoned(self):
        members = self._payload()["members"]
        self.assertTrue(members[COUNTER]["structural_surface"])
        self.assertIn("half-extent", members[COUNTER]["structural_surface_reason"])

    def test_each_member_carries_its_family(self):
        members = self._payload()["members"]
        self.assertEqual(members[BOWL]["family"], FAMILY_MANIPULAND)
        self.assertEqual(members[COUNTER]["family"], FAMILY_STRUCTURAL)

    def test_the_manipuland_is_not_flagged_structural(self):
        self.assertNotIn("structural_surface", self._payload()["members"][BOWL])

    def test_the_ee_site_bins_are_derived_on_their_own_scale(self):
        edges = self._payload()["bin_edges"]
        for key in (EE_SITE_PLANAR_KEY, EE_SITE_HEIGHT_KEY):
            with self.subTest(key=key):
                self.assertTrue(edges.get(key))
        self.assertNotEqual(edges[EE_SITE_PLANAR_KEY],
                            edges["ee-object-planar-distance"])

    def test_each_family_gets_its_own_height_scale(self):
        """Reprojected from the existing pose trace after classification --
        no new collector format, because ``pose_samples`` already keys every
        member's pose by its canonical key."""
        edges = self._payload()["bin_edges"]
        for key in ("ee-manipuland-height-offset",
                    "ee-structural-surface-height-offset"):
            with self.subTest(key=key):
                self.assertTrue(edges.get(key))

    def test_the_two_family_scales_differ(self):
        """The point of the split. Pooled, the metre to the counter would set
        the band a two-centimetre lift has to register against."""
        edges = self._payload()["bin_edges"]
        self.assertNotEqual(edges["ee-manipuland-height-offset"],
                            edges["ee-structural-surface-height-offset"])

    def test_the_surface_scale_is_measured_from_the_plane(self):
        """0.45m up in the counter's own frame, so a gripper at z=1.10 is
        0.65m above the top -- not the 1.10m to its origin."""
        builder = self._builder()
        builder.absorb(self._rollout())
        families = {COUNTER: FAMILY_STRUCTURAL}
        samples = builder._mine_family_height_samples(families)
        heights = samples["ee_structural_surface_height_offset"]
        self.assertAlmostEqual(max(heights), 0.65, places=6)

    def test_a_gripper_always_below_still_calibrates(self):
        """``derive_bin_edges`` builds a symmetric band from a positive
        half-width, so a signed stream of only-negative offsets yields no
        edges at all. A gripper reaching into a drawer is below its reference
        for the whole episode, which is not an exotic case."""
        builder = self._builder()
        rollout = self._rollout()
        # Every tcp height under the bowl's centre: dz is negative throughout.
        rollout["pose_samples"] = [
            {"tcp_pose": [0.0, 0.0, z] + [1.0, 0.0, 0.0, 0.0],
             "entities": {BOWL: {"pose": [0.0, 0.0, 0.60, 1.0, 0.0, 0.0, 0.0],
                                 "kind": "actor", "name": "b"}}}
            for z in (0.30, 0.34, 0.38, 0.42, 0.45, 0.49, 0.52)
        ]
        builder.absorb(rollout)
        samples = builder._mine_family_height_samples({BOWL: FAMILY_MANIPULAND})
        heights = samples["ee_manipuland_height_offset"]
        self.assertTrue(heights)
        self.assertTrue(all(v > 0 for v in heights),
                        "the absolute stream must carry magnitudes")
        self.assertTrue(builder.payload()["bin_edges"].get(
            "ee-manipuland-height-offset"))

    def test_the_change_stream_still_measures_signed_movement(self):
        """Sign belongs in the temporal history: the change is how far the
        gripper moved, which magnitudes alone cannot give."""
        builder = self._builder()
        builder.absorb(self._rollout())
        samples = builder._mine_family_height_samples({BOWL: FAMILY_MANIPULAND})
        self.assertTrue(samples.get("ee_manipuland_height_offset_change"))
        self.assertTrue(
            all(v >= 0 for v in samples["ee_manipuland_height_offset_change"]))

    def test_an_unreadable_supporter_stops_the_mine(self):
        """End to end, through the holder rule that used to hide it.

        A supporter with no extent is a holder, so classification labels it
        ``receptacle`` -- a real family -- and every later check that looks
        for a *missing* family then skips it. The only place this can be
        caught is before classification runs.
        """
        builder = self._builder()
        rollout = self._rollout()
        rollout["extents"][COUNTER] = {"extent_status": "no-collision-shapes"}
        builder.absorb(rollout)
        with self.assertRaises(ValueError) as ctx:
            builder.payload()
        message = str(ctx.exception)
        self.assertIn(COUNTER, message)
        self.assertIn("collision extent", message)

    def test_that_supporter_would_otherwise_have_been_called_a_receptacle(self):
        """Pins why the check has to run first, not last."""
        families = rules.object_families(
            {COUNTER: _member(["contact", "support"], "link")},
            holders={COUNTER}, supported=set(), structural=set())
        self.assertEqual(families[COUNTER], FAMILY_RECEPTACLE)

    def test_a_structural_surface_with_no_mined_plane_is_refused(self):
        """The runtime raises on it, and calibrating against the actor origin
        instead is the error the classification exists to remove."""
        builder = self._builder(with_plane=False)
        builder.absorb(self._rollout())
        with self.assertRaises(ValueError) as ctx:
            builder.payload()
        self.assertIn("reference_surface", str(ctx.exception))


class RestSiteAdmissionTest(unittest.TestCase):
    """The declaration is reviewed; the member row is derived from it."""

    def _declared(self):
        return declared_sites("tidy_house", SITES_DIR, "pick")

    def test_the_experiment_group_ships_a_declaration(self):
        self.assertIn(SITE_EE_REST, self._declared())

    def test_the_declaration_parses_and_validates(self):
        parsed = parse_site_declarations(self._declared(), where="tidy_house")
        spec = parsed[SITE_EE_REST]
        spec.validate()
        self.assertEqual(spec.subject_key, "ee")
        self.assertEqual(spec.provider, "mshab_ee_rest")

    def test_the_site_gets_a_vocabulary_row(self):
        """The encoder resolves node identity only through ``members``; with
        no row the site encodes as padding."""
        out = admit_site_members({BOWL: _member(["grasp"])}, self._declared())
        self.assertIn(SITE_EE_REST, out)
        self.assertEqual(out[SITE_EE_REST]["kind"], "spatial")
        self.assertEqual(out[SITE_EE_REST]["interaction_types"], [])

    def test_an_existing_member_is_not_overwritten(self):
        """PickCube's goal marker is a real actor and keeps its mined family."""
        existing = {"actor:goal_site": _member([])}
        out = admit_site_members(existing, {"actor:goal_site": {}})
        self.assertEqual(out["actor:goal_site"], existing["actor:goal_site"])

    def _payload(self):
        builder = _WhitelistBuilder(
            "pick", BOWL, task_group="tidy_house", sites=self._declared(),
            affordance_set=_affordance_set())
        builder.absorb(MinerPayloadTest()._rollout())
        return builder.payload()

    def test_the_miner_emits_the_declaration_and_the_row(self):
        payload = self._payload()
        self.assertIn(SITE_EE_REST, payload["sites"])
        self.assertIn(SITE_EE_REST, payload["members"])

    def test_the_rest_site_never_receives_a_family(self):
        """Rule 5 would call it a goal marker -- silently, because that rule
        returns a family rather than None -- and score gripper heights on a
        goal marker's deadband."""
        self.assertNotIn("family", self._payload()["members"][SITE_EE_REST])

    def test_a_group_with_no_declaration_file_gets_no_sites(self):
        self.assertEqual(declared_sites("no_such_group", SITES_DIR), {})

    def test_declarations_are_scoped_to_one_subtask(self):
        """The rest position is pick's: no other subtask defines an
        ee_rest_thresh, and routing by group alone would inject a pick site
        into a place or open mine."""
        self.assertIn(SITE_EE_REST,
                      declared_sites("tidy_house", SITES_DIR, "pick"))
        for subtask in ("place", "open", "close"):
            with self.subTest(subtask=subtask):
                self.assertEqual(
                    declared_sites("tidy_house", SITES_DIR, subtask), {})

    def test_only_tidy_house_declares_the_rest_site(self):
        """The frozen experiment is tidy_house/pick. SetTable is out of scope
        and must not silently acquire the same design."""
        self.assertEqual(declared_sites("set_table", SITES_DIR, "pick"), {})

    def test_the_maniskill_layout_is_unchanged(self):
        """One file per gym id, no subtask level."""
        self.assertIn("actor:goal_site",
                      declared_sites("PickCube-v1", SITES_DIR))


class VerificationGateTest(unittest.TestCase):
    """Final asset preparation must not succeed on an unclassified supporter.

    Each one is a member the runtime measures from its actor origin, and if
    one is a counter that is the ~0.9m error the classification removes.
    """

    def _dir(self, members):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        (directory / "pick_024_bowl.json").write_text(
            json.dumps({"members": members}))
        return directory

    def test_a_supporter_with_no_family_is_reported(self):
        from scenegraph.tools.prepare_assets import _unreadable_supporters
        found = _unreadable_supporters(
            self._dir({COUNTER: _member(["contact", "support"], "link")}))
        self.assertEqual(len(found), 1)
        self.assertIn(COUNTER, found[0])

    def test_a_classified_supporter_passes(self):
        from scenegraph.tools.prepare_assets import _unreadable_supporters
        entry = dict(_member(["contact", "support"], "link"),
                     family=FAMILY_STRUCTURAL, structural_surface=True)
        self.assertEqual(_unreadable_supporters(self._dir({COUNTER: entry})), [])

    def test_a_non_supporter_is_not_checked(self):
        """Only supporters need the extent, because only they can be an
        extended surface."""
        from scenegraph.tools.prepare_assets import _unreadable_supporters
        self.assertEqual(
            _unreadable_supporters(self._dir({BOWL: _member(["contact"])})), [])

    def test_the_shipped_maniskill_assets_pass_the_gate(self):
        from scenegraph.tools.prepare_assets import _unreadable_supporters
        for env_id in ("PickCube-v1", "PegInsertionSide-v1", "PlaceSphere-v1",
                       "PullCubeTool-v1"):
            with self.subTest(env=env_id):
                self.assertEqual(
                    _unreadable_supporters(
                        Path("scenegraph/configs/subtask_whitelists") / env_id),
                    [])


class UnionPreservationTest(unittest.TestCase):
    """The union is what the runtime binds bins, families and sites from."""

    def _write(self, directory, target, members, sites=None):
        payload = {
            "_schema_version": 4, "subtask": "pick", "task_group": "tidy_house",
            "membership_policy": "target-supporters", "target": target,
            "members": members, "sites": sites or {},
            "bin_edges": {}, "bin_stats_robust": {"ee_object_planar_distance": 1.0},
            "_n_successful_rollouts": 3,
        }
        (directory / f"pick_{target.split(':')[-1]}.json").write_text(
            json.dumps(payload))

    def _merge(self, first, second, sites=None):
        from scenegraph.tools.build_union_whitelist import merge
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write(directory, "actor:a", first, sites)
            self._write(directory, "actor:b", second, sites)
            return merge(directory, "pick")

    def test_families_survive_the_merge(self):
        merged = self._merge(
            {BOWL: dict(_member(["grasp"]), family=FAMILY_MANIPULAND)},
            {COUNTER: dict(_member(["support"], "link"),
                           family=FAMILY_STRUCTURAL, structural_surface=True)},
        )
        self.assertEqual(merged["members"][BOWL]["family"], FAMILY_MANIPULAND)
        self.assertTrue(merged["members"][COUNTER]["structural_surface"])

    def test_site_declarations_survive_the_merge(self):
        sites = declared_sites("tidy_house", SITES_DIR, "pick")
        merged = self._merge({BOWL: _member(["grasp"])},
                             {BOWL: _member(["grasp"])}, sites)
        self.assertIn(SITE_EE_REST, merged["sites"])

    def test_one_member_cannot_carry_two_families(self):
        """Silently keeping either would make one token mean two heights."""
        with self.assertRaises(ValueError) as ctx:
            self._merge(
                {BOWL: dict(_member(["grasp"]), family=FAMILY_MANIPULAND)},
                {BOWL: dict(_member(["grasp"]), family=FAMILY_RECEPTACLE)},
            )
        self.assertIn("family", str(ctx.exception))

    def test_a_site_declared_two_ways_is_refused(self):
        from scenegraph.tools.build_union_whitelist import merge
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write(directory, "actor:a", {BOWL: _member(["grasp"])},
                        {SITE_EE_REST: {"provider": "mshab_ee_rest"}})
            self._write(directory, "actor:b", {BOWL: _member(["grasp"])},
                        {SITE_EE_REST: {"provider": "something_else"}})
            with self.assertRaises(ValueError):
                merge(directory, "pick")


class ManiskillParityTest(unittest.TestCase):
    """The four verified tasks must classify exactly as before the move."""

    def test_the_maniskill_miner_delegates_rather_than_duplicating(self):
        from scenegraph.tools import build_maniskill_assets as ms
        self.assertEqual(ms.STRUCTURAL_SURFACE_MIN_HALF_EXTENT,
                         rules.STRUCTURAL_SURFACE_MIN_HALF_EXTENT)

    def test_the_shipped_assets_still_carry_their_classification(self):
        for env_id, key, family in (
            ("PickCube-v1", "actor:cube", FAMILY_MANIPULAND),
            ("PickCube-v1", "actor:table-workspace", FAMILY_STRUCTURAL),
            ("PickCube-v1", "actor:goal_site", FAMILY_GOAL_MARKER),
            ("PegInsertionSide-v1", "actor:peg", FAMILY_MANIPULAND),
        ):
            with self.subTest(env=env_id, key=key):
                path = Path("scenegraph/configs/subtask_whitelists") / env_id \
                    / "task_all.json"
                members = json.loads(path.read_text())["members"]
                self.assertEqual(members[key].get("family"), family)


class EeSiteScaleTest(unittest.TestCase):

    def test_the_two_site_scopes_are_calibrated_separately(self):
        """A peg head reaching a hole and a gripper returning to base are
        distances of different magnitudes over different bodies."""
        edges = derive_bin_edges({
            "ee_site_planar_distance": 1.2,
            "object_site_planar_distance": 0.05,
        })
        self.assertNotEqual(edges[EE_SITE_PLANAR_KEY],
                            edges["object-site-planar-distance"])

    def test_an_absent_statistic_derives_no_edges(self):
        self.assertNotIn(EE_SITE_PLANAR_KEY, derive_bin_edges({}))


if __name__ == "__main__":
    unittest.main()
