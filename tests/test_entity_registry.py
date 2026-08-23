"""Vertex indexing under unconditional retention.

Retention removed every reason the registry had to choose: a vertex admitted
this episode keeps its row until reset, so there is no resident whose eviction
would be correct. Capacity became a configuration fact and overflow raises.
These tests pin that contract and the index stability it rests on.
"""

import unittest

from scenegraph.core.schema import Node
from scenegraph.core.selector import EntityRegistry


def node(node_id, roles=("interacted",), object_type=None):
    return Node(
        node_id=node_id,
        node_type="object",
        name=node_id,
        visible=True,
        attributes={
            "whitelist_roles": list(roles),
            "whitelist_key": object_type or node_id,
        },
    )


def ee():
    return Node(node_id="ee", node_type="ee", name="end_effector")


def frame(registry, ids):
    """One builder step: the end effector plus every retained object."""
    nodes = {"ee": ee(), **{i: node(i, object_type=i.split("-")[-1]) for i in ids}}
    return registry.assign(nodes)


class IndexStabilityTest(unittest.TestCase):
    def test_ee_always_holds_row_zero(self):
        registry = EntityRegistry(n_max=4)
        out = frame(registry, ["a-apple"])
        self.assertEqual(out["ee"].index, 0)

    def test_index_survives_the_object_leaving_the_frame(self):
        """Retention means the row is the object's for the episode. A node
        re-injected without pixels must come back to the same row or every
        slot-aligned prediction shifts under it."""
        registry = EntityRegistry(n_max=4)
        first = frame(registry, ["a-apple", "b-bowl"])
        idx = first["a-apple"].index
        frame(registry, ["b-bowl"])              # apple occluded
        again = frame(registry, ["a-apple", "b-bowl"])
        self.assertEqual(again["a-apple"].index, idx)

    def test_arrival_order_does_not_disturb_residents(self):
        registry = EntityRegistry(n_max=5)
        first = frame(registry, ["a-apple"])
        idx = first["a-apple"].index
        out = frame(registry, ["c-can", "a-apple", "b-bowl"])
        self.assertEqual(out["a-apple"].index, idx)
        self.assertEqual(len({n.index for n in out.values()}), 4)

    def test_reset_releases_every_row(self):
        registry = EntityRegistry(n_max=3)
        frame(registry, ["a-apple", "b-bowl"])
        registry.reset_episode()
        self.assertEqual(len(registry), 0)
        self.assertEqual(registry.episode_entities, 0)
        frame(registry, ["c-can", "d-dish"])     # would have overflowed
        self.assertEqual(len(registry), 2)


class CapacityIsFatalTest(unittest.TestCase):
    """Nothing is dropped to make room. Overflow is a config error."""

    def test_one_object_over_capacity_raises(self):
        registry = EntityRegistry(n_max=3)       # ee plus two objects
        frame(registry, ["a-apple", "b-bowl"])
        with self.assertRaises(RuntimeError) as cm:
            frame(registry, ["a-apple", "b-bowl", "c-can"])
        self.assertIn("c-can", str(cm.exception))

    def test_the_error_names_the_residents_holding_the_rows(self):
        registry = EntityRegistry(n_max=3)
        frame(registry, ["a-apple", "b-bowl"])
        with self.assertRaises(RuntimeError) as cm:
            frame(registry, ["a-apple", "b-bowl", "c-can"])
        message = str(cm.exception)
        self.assertIn("a-apple", message)
        self.assertIn("b-bowl", message)

    def test_a_duplicated_type_is_not_a_licence_to_evict(self):
        """The old registry evicted from the most represented type. Two apples
        and one bowl must now raise like any other overflow."""
        registry = EntityRegistry(n_max=3)
        frame(registry, ["a-apple", "b-apple"])
        with self.assertRaises(RuntimeError):
            frame(registry, ["a-apple", "b-apple", "c-bowl"])

    def test_exactly_at_capacity_is_fine(self):
        registry = EntityRegistry(n_max=4)
        out = frame(registry, ["a-apple", "b-bowl", "c-can"])
        self.assertEqual(len(out), 4)            # three objects plus the ee
        self.assertEqual(sorted(n.index for n in out.values()), [0, 1, 2, 3])

    def test_capacity_counts_objects_not_rows(self):
        """n_max includes the end effector, so n_max=2 admits one object."""
        registry = EntityRegistry(n_max=2)
        frame(registry, ["a-apple"])
        with self.assertRaises(RuntimeError):
            frame(registry, ["a-apple", "b-bowl"])


class EpisodeEntitiesTest(unittest.TestCase):
    def test_it_counts_distinct_instances_this_episode(self):
        registry = EntityRegistry(n_max=6)
        frame(registry, ["a-apple", "b-bowl"])
        frame(registry, ["a-apple"])
        frame(registry, ["a-apple", "b-bowl", "c-can"])
        self.assertEqual(registry.episode_entities, 3)

    def test_it_is_the_number_to_size_n_max_against(self):
        """Under retention live occupancy equals the episode's instance count,
        which is what makes it the right thing to measure before a run."""
        registry = EntityRegistry(n_max=8)
        frame(registry, ["a-apple", "b-bowl"])
        frame(registry, ["c-can"])
        self.assertEqual(registry.episode_entities, len(registry))


if __name__ == "__main__":
    unittest.main()
