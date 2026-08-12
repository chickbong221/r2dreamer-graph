import unittest

from scenegraph.core.schema import Node
from scenegraph.core.selector import EntityRegistry


def node(node_id, roles=("interacted",)):
    return Node(
        node_id=node_id,
        node_type="object",
        name=node_id,
        visible=True,
        attributes={"whitelist_roles": list(roles)},
    )


def ee():
    return Node(node_id="ee", node_type="ee", name="end_effector")


class EntityRegistryFifoTest(unittest.TestCase):

    def test_new_instance_replaces_oldest_and_reuses_its_slot(self):
        registry = EntityRegistry(n_max=3)
        first = registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})
        a_index = first["a"].index
        b_index = first["b"].index

        current = registry.assign({
            "ee": ee(), "a": node("a"), "b": node("b"), "c": node("c"),
        })

        self.assertNotIn("a", current)
        self.assertEqual(current["b"].index, b_index)
        self.assertEqual(current["c"].index, a_index)
        self.assertEqual(registry.evicted_ids, ["a"])

    def test_old_overflow_instance_does_not_rotate_back_next_frame(self):
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})
        registry.assign({
            "ee": ee(), "a": node("a"), "b": node("b"), "c": node("c"),
        })

        current = registry.assign({
            "ee": ee(), "a": node("a"), "b": node("b"), "c": node("c"),
        })

        self.assertEqual(set(current), {"ee", "b", "c"})
        self.assertEqual(registry.evicted_ids, [])

    def test_age_wins_over_whitelist_role_priority(self):
        registry = EntityRegistry(n_max=2)
        registry.assign({"ee": ee(), "target": node("target", ("interacted",))})

        current = registry.assign({
            "ee": ee(),
            "target": node("target", ("interacted",)),
            "support": node("support", ("support",)),
        })

        self.assertNotIn("target", current)
        self.assertIn("support", current)


if __name__ == "__main__":
    unittest.main()
