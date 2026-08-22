import unittest

import numpy as np

from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.viz.graph_draw import (
    _group_by_family,
    _radial_layout,
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


if __name__ == "__main__":
    unittest.main()
