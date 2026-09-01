import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.viz.graph_draw import (
    _group_by_family,
    _node_display_label,
    _paper_display_edges,
    _paper_cluster_fraction,
    _paper_pair_offset,
    _radial_layout,
    render_graph,
    render_graph_array,
)


class GraphDisplayFactsTest(unittest.TestCase):
    def test_predicate_states_are_presented_as_relations(self):
        grouped = _group_by_family([
            Edge("ee", "cube", "contact", "not-holds"),
            Edge("ee", "cube", "grasp", "unobserved"),
            Edge("cube", "table", "support", "dst-holds"),
            Edge("cube", "table", "contain", "holds"),
            Edge("cube", "table", "planar-distance", "near"),
        ])

        self.assertEqual(grouped["physical_state"], ["support", "contain"])
        self.assertEqual(grouped["spatial"], ["near"])
        rendered = " ".join(text for lines in grouped.values() for text in lines)
        self.assertNotIn("holds", rendered)
        self.assertNotIn("unobserved", rendered)

    def test_paper_affordances_include_the_relation_type(self):
        grouped = _group_by_family([
            Edge("ee", "cube", "grasp-compatibility", "match"),
            Edge("cube", "table", "support-compatibility", "partial-match"),
        ], name_affordance=True)

        self.assertEqual(
            grouped["affordance"],
            ["grasp match", "support partial-match"],
        )

    def test_place_sphere_paper_view_shortens_the_table_name(self):
        graph = Graph(0, "PlaceSphere-v1", "base")
        table = Node(
            "actor:table-workspace", "object", "table-workspace",
        )

        self.assertEqual(_node_display_label(graph, table, True), "table")
        self.assertEqual(
            _node_display_label(graph, table, False), "table-workspace",
        )

    def test_paper_view_hides_contact_after_stronger_relation_holds(self):
        graph = Graph(
            frame=0,
            env_id="test",
            camera="base",
            edges=[
                Edge("ee", "cube", "contact", "holds"),
                Edge("ee", "cube", "grasp", "holds"),
                Edge("ee", "cube", "contact-compatibility", "match"),
                Edge("ee", "cube", "grasp-compatibility", "match"),
            ],
        )

        relations = [edge.relation for edge in _paper_display_edges(graph)]

        self.assertEqual(
            relations, ["grasp", "grasp-compatibility"]
        )

    def test_place_sphere_paper_view_adds_static_table_bin_support(self):
        graph = Graph(
            frame=0,
            env_id="PlaceSphere-v1",
            camera="base",
            nodes=[
                Node("actor:bin", "object", "bin"),
                Node("actor:table-workspace", "object", "table-workspace"),
            ],
        )

        shown = _paper_display_edges(graph)

        self.assertEqual(graph.edges, [])
        self.assertEqual(
            {(edge.relation, edge.label) for edge in shown},
            {("support", "dst-holds")},
        )

    def test_peg_insertion_paper_view_adds_static_table_box_support(self):
        """The box with the hole is kinematic, like PlaceSphere's bin: nothing
        ever pushes it, so the runtime graph has no pair to report."""
        graph = Graph(
            frame=0,
            env_id="PegInsertionSide-v1",
            camera="base",
            nodes=[
                Node("actor:box_with_hole", "object", "box_with_hole"),
                Node("actor:table-workspace", "object", "table-workspace"),
                Node("actor:peg", "object", "peg"),
            ],
        )

        shown = _paper_display_edges(graph)

        self.assertEqual(graph.edges, [])
        self.assertEqual(
            [(e.src, e.dst, e.relation, e.label) for e in shown],
            [("actor:box_with_hole", "actor:table-workspace",
              "support", "dst-holds")],
        )

    def test_layout_support_is_not_drawn_twice_when_the_builder_emits_it(self):
        """The pair is added only where the runtime is silent about it."""
        graph = Graph(
            frame=0,
            env_id="PegInsertionSide-v1",
            camera="base",
            nodes=[
                Node("actor:box_with_hole", "object", "box_with_hole"),
                Node("actor:table-workspace", "object", "table-workspace"),
            ],
            edges=[Edge("actor:box_with_hole", "actor:table-workspace",
                        "support", "dst-holds")],
        )

        shown = _paper_display_edges(graph)

        self.assertEqual(len(shown), 1)

    def test_a_task_with_no_layout_pair_gains_no_edge(self):
        graph = Graph(
            frame=0,
            env_id="PickCube-v1",
            camera="base",
            nodes=[
                Node("actor:cube", "object", "cube"),
                Node("actor:table-workspace", "object", "table-workspace"),
            ],
        )

        self.assertEqual(_paper_display_edges(graph), [])

    def test_hidden_facts_leave_the_reference_background_visible(self):
        graph = Graph(
            frame=0,
            env_id="test",
            camera="base",
            nodes=[
                Node("ee", "ee", "ee"),
                Node("cube", "object", "cube"),
            ],
            edges=[Edge("ee", "cube", "contact", "not-holds")],
        )

        image = render_graph_array(graph, height=160)

        self.assertEqual(image.dtype, np.uint8)
        # The fixed canvas background is #fdf0e9, including in array/video
        # rendering (allow one value for backend rounding).
        np.testing.assert_allclose(image[0, 0], [253, 240, 233], atol=1)

        native = render_graph_array(graph)
        self.assertEqual(native.shape, (1200, 1200, 3))

    def test_two_objects_and_ee_form_a_triangle(self):
        graph = Graph(
            frame=0,
            env_id="test",
            camera="base",
            nodes=[
                Node("ee", "ee", "ee"),
                Node("cube", "object", "cube"),
                Node("table", "object", "table"),
            ],
        )

        pos = _radial_layout(graph, radius=5.5, node_r=0.32)

        object_edge = pos["table"] - pos["cube"]
        ee_offset = pos["ee"] - pos["cube"]
        cross = object_edge[0] * ee_offset[1] - object_edge[1] * ee_offset[0]
        self.assertGreater(abs(float(cross)), 1.0)

    def test_place_sphere_bin_relations_share_one_cluster_anchor(self):
        graph = Graph(frame=94, env_id="PlaceSphere-v1", camera="base")

        fraction = _paper_cluster_fraction(
            graph,
            ("actor:bin", "actor:sphere"),
        )

        self.assertAlmostEqual(fraction, 0.48)
        self.assertGreater(fraction, 0.0)
        self.assertLess(fraction, 1.0)

    def test_place_sphere_paper_view_preserves_context_relations(self):
        graph = Graph(
            frame=94,
            env_id="PlaceSphere-v1",
            camera="base",
            nodes=[
                Node("ee", "ee", "ee"),
                Node("actor:bin", "object", "bin"),
                Node("actor:sphere", "object", "sphere"),
                Node("actor:table-workspace", "object", "table-workspace"),
            ],
            edges=[
                Edge("ee", "actor:bin", "planar-distance", "medium"),
                Edge("ee", "actor:sphere", "planar-distance", "very-near"),
                Edge("actor:bin", "actor:sphere", "planar-distance", "near"),
                Edge("ee", "actor:table-workspace", "height-offset", "above"),
                Edge("actor:sphere", "actor:table-workspace", "support", "dst-holds"),
            ],
        )

        shown = _paper_display_edges(graph)
        facts = {(edge.src, edge.dst, edge.relation) for edge in shown}

        self.assertIn(("ee", "actor:bin", "planar-distance"), facts)
        self.assertIn(("ee", "actor:table-workspace", "height-offset"), facts)
        self.assertIn(("ee", "actor:sphere", "planar-distance"), facts)
        self.assertIn(("actor:bin", "actor:sphere", "planar-distance"), facts)
        self.assertIn(("actor:sphere", "actor:table-workspace", "support"), facts)

    def test_place_sphere_ee_bin_labels_stay_on_vertical_edge(self):
        graph = Graph(frame=94, env_id="PlaceSphere-v1", camera="base")

        offset = _paper_pair_offset(graph, ("actor:bin", "ee"))

        np.testing.assert_allclose(offset, [0.0, 0.80])

    def test_paper_graph_is_cropped_and_has_no_title_strip(self):
        from PIL import Image

        graph = Graph(
            frame=80,
            env_id="PlaceSphere-v1",
            camera="base",
            nodes=[
                Node("ee", "ee", "ee"),
                Node("actor:bin", "object", "bin"),
                Node("actor:sphere", "object", "sphere"),
                Node("actor:table-workspace", "object", "table-workspace"),
            ],
            edges=[
                Edge("ee", "actor:sphere", "grasp", "holds"),
                Edge("ee", "actor:sphere", "grasp-compatibility", "match"),
            ],
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "paper.png"
            render_graph(graph, str(path), paper_style=True)
            with Image.open(path) as image:
                width, height = image.size

        self.assertLess(height, 1000)
        self.assertLess(width, 1200)
        self.assertGreater(width, height)


if __name__ == "__main__":
    unittest.main()
