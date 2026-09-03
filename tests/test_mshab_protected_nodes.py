"""The three nodes an MS-HAB Pick episode is scored against, and their rows.

    row 0  ee
    row 1  the active target
    row 2  spatial:ee_rest_site

Every scheduled fact reads one of them, so a frame missing any one scores
nothing. Two of the three are not guaranteed by the ordinary pipeline: under
``projected_camera`` an object becomes a vertex only once a camera has covered
it, and the vertex registry hands out rows in arrival order.

The rows are a contract rather than a convention because the schedule's
active-target role resolves *by position*. It has to: one schedule serves nine
objects, and a scene can hold two instances of one category that share a
whitelist key and therefore an entity id, which no scan can tell apart.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from scenegraph.adapters.graph_pack import _row_assignment, verify_protected_rows
from scenegraph.core.relation_rules import (
    EE_SITE_HEIGHT_KEY,
    EE_SITE_PLANAR_KEY,
    ee_object_spatial_edges,
    ee_site_key,
    goal_edges,
)
from scenegraph.core.schedule import (
    ACTIVE_TARGET_ENTITY_ID,
    ACTIVE_TARGET_ROLE,
    ACTIVE_TARGET_ROW,
    ScheduleError,
)
from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.core.sites import (
    METRIC_EUCLIDEAN,
    PROVIDER_MSHAB_EE_REST,
    SITE_EE_REST,
    SITE_POINT,
    SOURCE_ORIGIN,
    SiteDeclaration,
    SiteSpec,
)

TARGET = "actor:024_bowl"
SIBLING = "actor:024_bowl-1"

# Row 0 is checked against the vocabulary's own end-effector id, not just
# against a node named "ee": a row that encodes something else would pass a
# name check and fail the model.
VOCAB = SimpleNamespace(entity=SimpleNamespace(ee_id=1))


def _pose(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def _ee(pose=None):
    return Node(node_id="ee", node_type="ee", name="ee",
                pose_world=list(pose or _pose(z=1.0)), attributes={})


def _obj(node_id, key=None, pose=None):
    return Node(node_id=node_id, node_type="object", name=node_id,
                pose_world=list(pose or _pose()),
                attributes={"whitelist_key": key or node_id})


def _site(pose=None):
    node = _obj(SITE_EE_REST, SITE_EE_REST, pose)
    node.visible = False
    node.in_frame = True
    node.attributes["is_site"] = True
    node.attributes["body_type"] = "kinematic"
    return node


def _graph(*nodes, target=TARGET, site=SITE_EE_REST, frame=0):
    return Graph(frame=frame, env_id="PickSubtaskTrain-v0", camera="fetch_head",
                 nodes=list(nodes), edges=[],
                 meta={"active_target_node_id": target,
                       "protected_site_node_id": site})


def _decl(subject="ee", key=SITE_EE_REST):
    return SiteDeclaration(
        key=key, site_type=SITE_POINT, subject_key=subject,
        metric=METRIC_EUCLIDEAN, source=SOURCE_ORIGIN,
        provider=PROVIDER_MSHAB_EE_REST,
        provenance="PickSubtask ee_rest",
    )


def _spec(decl=None, pose=None, tolerance=0.05):
    return SiteSpec(declaration=decl or _decl(),
                    pose_world=np.asarray(pose or _pose(), float),
                    tolerance=tolerance)


class RowAssignmentTest(unittest.TestCase):
    """Rows pinned by meaning, not arrival order."""

    def _rows(self, *nodes, target=TARGET, site=SITE_EE_REST, n_max=8):
        assigned, _ = _row_assignment(
            _graph(*nodes, target=target, site=site), n_max, target, True, site)
        return {node.node_id: row for row, node in assigned}

    def test_the_three_take_their_reserved_rows(self):
        rows = self._rows(_obj("actor:counter"), _site(), _obj(TARGET), _ee())
        self.assertEqual(rows["ee"], 0)
        self.assertEqual(rows[TARGET], 1)
        self.assertEqual(rows[SITE_EE_REST], 2)

    def test_arrival_order_does_not_move_them(self):
        """The registry hands out indices as nodes are first admitted, which
        is why the packer cannot simply reuse them."""
        first = self._rows(_ee(), _obj(TARGET), _site(), _obj("actor:a"))
        second = self._rows(_obj("actor:a"), _site(), _ee(), _obj(TARGET))
        for key in ("ee", TARGET, SITE_EE_REST):
            with self.subTest(node=key):
                self.assertEqual(first[key], second[key])

    def test_ordinary_objects_start_at_row_three(self):
        rows = self._rows(_ee(), _obj(TARGET), _site(), _obj("actor:a"))
        self.assertEqual(rows["actor:a"], 3)

    def test_a_task_with_no_protected_site_keeps_the_old_layout(self):
        """ManiSkill declares sites too, and reserving a row for a hole would
        move a layout that is already verified."""
        rows = self._rows(_ee(), _obj(TARGET), _obj("actor:a"), site=None)
        self.assertEqual(rows["actor:a"], 2)

    def test_a_sibling_of_the_target_does_not_take_row_one(self):
        """Two instances of one category share a whitelist key; only the
        flagged node id is the target."""
        rows = self._rows(_ee(), _obj(SIBLING, TARGET), _obj(TARGET), _site())
        self.assertEqual(rows[TARGET], 1)
        self.assertNotEqual(rows[SIBLING], 1)

    def test_overflow_names_both_reservations(self):
        """Reserving two rows costs two object rows, so a scene that fits by
        vertex count can still fail to fit by row -- and has to say why."""
        # Five vertices into five rows, but rows 1 and 2 are reserved and the
        # target has not been admitted yet, so only rows 3 and 4 are free.
        nodes = [_ee(), _site()] + [_obj(f"actor:{i}") for i in range(3)]
        with self.assertRaises(RuntimeError) as ctx:
            self._rows(*nodes, n_max=5)
        message = str(ctx.exception)
        self.assertIn("row 1 reserved for target", message)
        self.assertIn("row 2 for site", message)


class PackInvariantTest(unittest.TestCase):
    """A contract nothing checks is a silent wrong answer: the phase would
    score whichever object happened to land in the row."""

    def _check(self, ent, flags, position, target=TARGET, site=SITE_EE_REST,
               vocab=VOCAB):
        verify_protected_rows(
            _graph(target=target, site=site), np.asarray(ent, np.uint8),
            np.asarray(flags, np.uint8), position, target, site, vocab)

    def test_a_correctly_packed_frame_passes(self):
        self._check([1, 7, 5, 0], [0, 1, 0, 0],
                    {"ee": 0, TARGET: 1, SITE_EE_REST: 2})

    def test_a_target_off_its_row_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 5, 7, 0], [0, 0, 0, 0],
                        {"ee": 0, SITE_EE_REST: 1, TARGET: 2})
        self.assertIn("reserved row 1", str(ctx.exception))

    def test_a_padding_target_row_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 0, 5, 0], [0, 1, 0, 0],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 2})
        self.assertIn("pad entity id", str(ctx.exception))

    def test_two_flagged_rows_raise(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 7, 5, 7], [0, 1, 0, 1],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 2})
        self.assertIn("exactly one row", str(ctx.exception))

    def test_no_flagged_row_raises(self):
        with self.assertRaises(RuntimeError):
            self._check([1, 7, 5, 0], [0, 0, 0, 0],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 2})

    def test_a_missing_site_raises(self):
        """It is derived from the robot every frame, so absence means the
        builder stopped producing it."""
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 7, 0, 0], [0, 1, 0, 0], {"ee": 0, TARGET: 1})
        self.assertIn("not in the packed graph", str(ctx.exception))

    def test_a_site_off_its_row_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 7, 0, 5], [0, 1, 0, 0],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 3})
        self.assertIn("reserved row 2", str(ctx.exception))

    def test_a_flagged_site_raises(self):
        """It is a place, not the object the subtask acts on. Caught by the
        exactly-one-flag rule, which fires first."""
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 7, 5, 0], [0, 1, 1, 0],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 2})
        self.assertIn("exactly one row", str(ctx.exception))

    def test_an_unadmitted_target_is_fatal(self):
        """Seeding puts the target in the graph at reset and retention keeps
        it, so an absent one means the seeding did not run -- and row 1 would
        be padding the schedule reads as the target."""
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 0, 5, 0], [0, 0, 0, 0], {"ee": 0, SITE_EE_REST: 2})
        self.assertIn("not in the packed graph", str(ctx.exception))

    def test_a_protected_site_with_no_named_target_is_fatal(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 0, 5, 0], [0, 0, 0, 0], {"ee": 0, SITE_EE_REST: 2},
                        target=None)
        self.assertIn("named no active target", str(ctx.exception))

    def test_only_a_run_with_neither_bypasses_the_checks(self):
        """ManiSkill: no target flag, no protected site, so no rows are
        pinned and there is nothing here to verify."""
        self._check([1, 5, 0, 0], [0, 0, 0, 0], {}, target=None, site=None)

    def test_a_missing_end_effector_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([7, 1, 5, 0], [1, 0, 0, 0],
                        {TARGET: 0, "ee": 1, SITE_EE_REST: 2})
        self.assertIn("reserved row 0", str(ctx.exception))

    def test_row_zero_must_encode_the_end_effector(self):
        """Position alone is not enough: a node id of "ee" on a row that
        encodes something else would pass a name check and fail the model."""
        with self.assertRaises(RuntimeError) as ctx:
            self._check([7, 7, 5, 0], [0, 1, 0, 0],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 2})
        self.assertIn("not the end", str(ctx.exception))

    def test_a_padding_site_row_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._check([1, 7, 0, 0], [0, 1, 0, 0],
                        {"ee": 0, TARGET: 1, SITE_EE_REST: 2})
        self.assertIn("pad entity id", str(ctx.exception))


class ActiveTargetRoleTest(unittest.TestCase):
    """A role that names whichever object the episode is acting on."""

    def test_the_sentinel_is_outside_the_entity_vocabulary(self):
        """A real id would be looked up in ``node_ent`` and match some row by
        accident."""
        self.assertLess(ACTIVE_TARGET_ENTITY_ID, 0)

    def test_it_resolves_to_the_reserved_target_row(self):
        self.assertEqual(ACTIVE_TARGET_ROW, 1)

    def _compiled(self, relation="grasp", labels=("holds",)):
        """A one-phase schedule whose only role is the dynamic target."""
        from scenegraph.adapters.graph_vocab import (
            EE_TOKEN, PAD_TOKEN, EntityVocab,
        )
        from scenegraph.core.schedule import compile_schedule
        clause = {"relation": relation, "src": "ee", "dst": "target",
                  "labels": list(labels), "weight": 1.0}
        raw = {
            "_schema_version": 1, "env_id": "PickSubtaskTrain-v0",
            "roles": {"target": ACTIVE_TARGET_ROLE},
            "phases": [{"name": "acquire", "weight": 1.0,
                        "clauses": [clause], "completion": dict(clause)}],
        }
        members = {
            TARGET: {"interaction_types": ["contact", "grasp"],
                     "family": "manipuland"},
            "actor:003_cracker_box": {"interaction_types": ["contact", "grasp"],
                                      "family": "manipuland"},
        }
        objects = {key: {"grasp_components": [{}], "contact_components": [{}]}
                   for key in members}
        bins = {"ee-object-planar-distance": [0.1, 0.2, 0.3, 0.4],
                "ee-manipuland-height-offset": [-0.1, 0.0, 0.05, 0.1]}
        vocab = EntityVocab(token_to_id={
            PAD_TOKEN: 0, EE_TOKEN: 1, TARGET: 2,
            "actor:003_cracker_box": 3})
        return compile_schedule(raw, objects, members, bins, vocab)

    def test_a_schedule_naming_only_the_dynamic_role_compiles(self):
        """It names no fixed member by design; requiring one is what made the
        role impossible to express."""
        schedule = self._compiled()
        self.assertEqual(schedule.role_entity_ids["target"],
                         ACTIVE_TARGET_ENTITY_ID)

    def test_the_compiled_clauses_carry_the_sentinel(self):
        """The slot the runtime looks up is ``(relation, src, dst)`` in entity
        ids, so the sentinel has to survive compilation into it."""
        schedule = self._compiled()
        clause = schedule.phases[0].clauses[0]
        self.assertEqual(clause.dst_entity_id, ACTIVE_TARGET_ENTITY_ID)
        self.assertIn(ACTIVE_TARGET_ENTITY_ID, schedule.entity_ids)
        self.assertIn(ACTIVE_TARGET_ENTITY_ID,
                      [slot[2] for slot in schedule.slots])

    def test_the_end_effector_side_keeps_its_real_id(self):
        clause = self._compiled().phases[0].clauses[0]
        self.assertEqual(clause.src_entity_id, 1)
        self.assertGreater(clause.src_entity_id, 0)

    def test_a_relation_one_candidate_cannot_score_is_refused(self):
        """Scorability for the dynamic pair is the intersection over the
        objects it can resolve to."""
        with self.assertRaises(ScheduleError):
            self._compiled(relation="contain", labels=("src-holds",))

    def test_an_object_object_clause_against_it_is_refused(self):
        """Stored edge order for an object pair comes from sorting the two
        keys, and the sentinel sorts against nothing."""
        from scenegraph.core.schedule import _order
        with self.assertRaises(ScheduleError):
            _order(ACTIVE_TARGET_ROLE, "actor:counter")


class EeSiteEmissionTest(unittest.TestCase):
    """The one end-effector pair a virtual site may have."""

    BINS = {
        EE_SITE_PLANAR_KEY: [0.2, 0.4, 0.6, 0.8],
        EE_SITE_HEIGHT_KEY: [-0.4, -0.15, 0.15, 0.4],
        "ee-object-planar-distance": [0.1, 0.2, 0.3, 0.4],
        "ee-manipuland-height-offset": [-0.1, -0.03, 0.03, 0.1],
    }

    def _cfg(self, subject="ee"):
        return {
            "bin_edges": dict(self.BINS),
            "site_declarations": {SITE_EE_REST: _decl(subject)},
            "families": {TARGET: "manipuland"},
            "structural_surfaces": set(),
            "affordance_set": None,
        }

    def _edges(self, subject="ee"):
        graph = _graph(_ee(_pose(z=1.0)), _obj(TARGET, pose=_pose(z=0.5)),
                       _site(_pose(x=0.3, z=0.9)))
        return ee_object_spatial_edges(graph, None, self._cfg(subject))

    def test_the_declared_gripper_site_gets_both_relations(self):
        pairs = {(e.src, e.dst, e.relation) for e in self._edges()}
        self.assertIn(("ee", SITE_EE_REST, "planar-distance"), pairs)
        self.assertIn(("ee", SITE_EE_REST, "height-offset"), pairs)

    def test_it_is_labelled_on_its_own_scale(self):
        """Never an object's: a return-to-base distance would stretch the band
        a two-centimetre approach registers against."""
        for edge in self._edges():
            if edge.dst == SITE_EE_REST:
                with self.subTest(relation=edge.relation):
                    self.assertIn("ee-site", edge.bin_key)

    def test_a_site_declared_against_an_object_gets_no_gripper_edges(self):
        """A hole or a goal region is measured against a manipuland; an
        end-effector distance to one is a number with no meaning."""
        pairs = {(e.src, e.dst) for e in self._edges(subject=TARGET)}
        self.assertNotIn(("ee", SITE_EE_REST), pairs)

    def test_the_ordinary_object_pair_is_unchanged(self):
        pairs = {(e.src, e.dst, e.relation) for e in self._edges()}
        self.assertIn(("ee", TARGET, "planar-distance"), pairs)
        self.assertIn(("ee", TARGET, "height-offset"), pairs)

    def test_the_helper_names_only_gripper_subject_sites(self):
        site = _site()
        self.assertEqual(ee_site_key(site, self._cfg()), SITE_EE_REST)
        self.assertIsNone(ee_site_key(site, self._cfg(subject=TARGET)))
        self.assertIsNone(ee_site_key(_obj(TARGET), self._cfg()))

    def test_reached_is_emitted_for_the_gripper_pair(self):
        graph = _graph(_ee(_pose(z=1.0)), _site(_pose(z=0.98)))
        edges = goal_edges(graph, None, {"site_specs": [_spec(pose=_pose(z=0.98))]})
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation, "reached")
        self.assertEqual((edges[0].src, edges[0].dst), ("ee", SITE_EE_REST))


class ScorabilityTest(unittest.TestCase):
    """Only the declared gripper pair becomes scorable."""

    def _scorable(self, subject="ee"):
        from scenegraph.core.schedule import scorable_relations
        members = {
            TARGET: {"interaction_types": ["contact", "grasp"],
                     "family": "manipuland"},
            SITE_EE_REST: {"interaction_types": [], "kind": "spatial"},
        }
        bins = {
            "ee-object-planar-distance": [0.1, 0.2, 0.3, 0.4],
            "ee-manipuland-height-offset": [-0.1, 0.0, 0.05, 0.1],
            EE_SITE_PLANAR_KEY: [0.2, 0.4, 0.6, 0.8],
            EE_SITE_HEIGHT_KEY: [-0.4, -0.15, 0.15, 0.4],
        }
        return scorable_relations(
            {}, members, bins, sites={SITE_EE_REST: _decl(subject)})

    def test_the_gripper_site_pair_is_scorable(self):
        pair = self._scorable()[f"ee / {SITE_EE_REST}"]
        self.assertTrue(pair["planar-distance"])
        self.assertTrue(pair["height-offset"])

    def test_an_object_subject_site_is_not(self):
        pair = self._scorable(subject=TARGET)[f"ee / {SITE_EE_REST}"]
        self.assertFalse(pair["planar-distance"])
        self.assertFalse(pair["height-offset"])

    def test_the_dynamic_target_pair_exists(self):
        pair = self._scorable().get(f"ee / {ACTIVE_TARGET_ROLE}")
        self.assertIsNotNone(pair)
        self.assertTrue(pair["grasp"])

    def test_the_dynamic_pair_is_the_intersection_over_candidates(self):
        """A clause that works for eight objects and silently scores zero for
        the ninth is worse than one that refuses to compile."""
        from scenegraph.core.schedule import scorable_relations
        members = {
            TARGET: {"interaction_types": ["contact", "grasp"],
                     "family": "manipuland"},
            "actor:ungraspable": {"interaction_types": ["contact"],
                                  "family": "manipuland"},
        }
        out = scorable_relations({}, members, {
            "ee-object-planar-distance": [0.1, 0.2, 0.3, 0.4],
            "ee-manipuland-height-offset": [-0.1, 0.0, 0.05, 0.1]})
        self.assertTrue(out[f"ee / {TARGET}"]["grasp"])
        self.assertFalse(out[f"ee / {ACTIVE_TARGET_ROLE}"]["grasp"])


class DynamicRowResolutionTest(unittest.TestCase):
    """The scorer half: the sentinel resolves by position, not by scanning.

    Needs torch, so it is skipped where the graph stack is not installed and
    runs on the collection server with the rest of the suite.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch is not installed")

    def _scorer(self):
        import torch

        from progress import TaskScheduleReplayPotential
        from scenegraph.adapters.graph_vocab import build_absolute_vocab

        schedule = ActiveTargetRoleTest()._compiled()
        return TaskScheduleReplayPotential(
            schedule, len(build_absolute_vocab())), torch

    def _rows(self, node_ent):
        scorer, torch = self._scorer()
        rows, resolved = scorer.resolve_rows(torch.tensor([node_ent]))
        column = list(scorer.entities).index(ACTIVE_TARGET_ENTITY_ID)
        return int(rows[0, column]), bool(resolved[0, column])

    def test_it_resolves_to_the_reserved_row(self):
        row, resolved = self._rows([1, 2, 5, 0, 0, 0, 0, 0])
        self.assertEqual(row, ACTIVE_TARGET_ROW)
        self.assertTrue(resolved)

    def test_a_padding_row_is_unresolved(self):
        """Defence in depth: the packer refuses to write this frame at all,
        but the scorer must not read padding as a target if one ever slips
        through a different path."""
        _row, resolved = self._rows([1, 0, 5, 0, 0, 0, 0, 0])
        self.assertFalse(resolved)

    def test_two_instances_of_one_category_do_not_defeat_it(self):
        """The reason the role exists. Scanning entity ids finds two rows and
        gives up; the reserved row names exactly one."""
        row, resolved = self._rows([1, 2, 5, 2, 0, 0, 0, 0])
        self.assertEqual(row, ACTIVE_TARGET_ROW)
        self.assertTrue(resolved)

    def test_the_end_effector_still_resolves_by_entity_id(self):
        scorer, torch = self._scorer()
        rows, resolved = scorer.resolve_rows(
            torch.tensor([[1, 2, 5, 0, 0, 0, 0, 0]]))
        column = list(scorer.entities).index(1)
        self.assertEqual(int(rows[0, column]), 0)
        self.assertTrue(bool(resolved[0, column]))


class NoContactInPickTest(unittest.TestCase):
    """Contact stays observable and unscheduled: a grasped object is also in
    contact, so scoring both pays twice for one event."""

    def test_the_relation_still_exists_in_the_vocabulary(self):
        from scenegraph.adapters.graph_vocab import build_relation_vocab
        self.assertIn("contact", build_relation_vocab().token_to_id)


if __name__ == "__main__":
    unittest.main()
