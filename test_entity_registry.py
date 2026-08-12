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


class EntityRegistryEvictionTest(unittest.TestCase):

    def test_overrepresented_type_is_evicted_before_oldest_singleton(self):
        registry = EntityRegistry(n_max=4)  # ee plus three objects
        first = registry.assign({
            "ee": ee(),
            "a-important": node("a-important", object_type="bowl"),
            "b-can-1": node("b-can-1", object_type="can"),
            "c-can-2": node("c-can-2", object_type="can"),
        })
        important_index = first["a-important"].index

        current = registry.assign({
            "ee": ee(),
            "a-important": node("a-important", object_type="bowl"),
            "b-can-1": node("b-can-1", object_type="can"),
            "c-can-2": node("c-can-2", object_type="can"),
            "d-box": node("d-box", object_type="box"),
        })

        self.assertIn("a-important", current)
        self.assertEqual(current["a-important"].index, important_index)
        self.assertNotIn("b-can-1", current)
        self.assertIn("d-box", current)
        self.assertEqual(registry.evicted_ids, ["b-can-1"])

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

    def test_resident_exact_target_is_never_evicted(self):
        registry = EntityRegistry(n_max=3)
        first = registry.assign({
            "ee": ee(),
            "target": node("target", object_type="bowl"),
            "can": node("can", object_type="can"),
        }, protected_id="target")
        target_index = first["target"].index

        current = registry.assign({
            "ee": ee(),
            "target": node("target", object_type="bowl"),
            "can": node("can", object_type="can"),
            "box": node("box", object_type="box"),
        }, protected_id="target")

        self.assertIn("target", current)
        self.assertEqual(current["target"].index, target_index)
        self.assertNotIn("can", current)

    def test_incoming_target_bypasses_age_rejection(self):
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})
        # Make the target older in first-seen history, then let newer residents
        # occupy the slots. A normal incoming object would be rejected here.
        registry._first_seen["target"] = -1
        current = registry.assign({
            "ee": ee(),
            "a": node("a"),
            "b": node("b"),
            "target": node("target", object_type="bowl"),
        }, protected_id="target")

        self.assertIn("target", current)
        self.assertEqual(len([key for key in current if key != "ee"]), 2)

    def test_only_exact_target_instance_is_protected(self):
        registry = EntityRegistry(n_max=3)
        registry.assign({
            "ee": ee(),
            "target-bowl": node("target-bowl", object_type="bowl"),
            "other-bowl": node("other-bowl", object_type="bowl"),
        }, protected_id="target-bowl")

        current = registry.assign({
            "ee": ee(),
            "target-bowl": node("target-bowl", object_type="bowl"),
            "other-bowl": node("other-bowl", object_type="bowl"),
            "can": node("can", object_type="can"),
        }, protected_id="target-bowl")

        self.assertIn("target-bowl", current)
        self.assertNotIn("other-bowl", current)
        self.assertIn("can", current)


if __name__ == "__main__":
    unittest.main()
