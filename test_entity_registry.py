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


class RetainedTargetTest(unittest.TestCase):
    """The subtask target keeps its registry position once admitted.

    History-off graphs emit only what a camera sees, so ``retain`` releases
    absent objects. The target is the exception: it is fixed for the episode, the
    world model has to keep predicting it while it is occluded, and releasing its
    index would let the next arriving object take the slot.
    """

    TARGET = "t-apple"

    def frame(self, registry, ids, keep_target=True):
        """One builder step: retain what is visible plus the retained target."""
        nodes = {"ee": ee(), **{i: node(i, object_type=i.split("-")[1]) for i in ids}}
        retain = set(nodes)
        if keep_target and registry.index_of(self.TARGET) is not None:
            retain.add(self.TARGET)
        registry.retain(retain)
        return registry.assign(nodes, protected_id=self.TARGET)

    def test_an_absent_target_keeps_its_index_and_its_protection(self):
        registry = EntityRegistry(n_max=4)  # ee plus three objects
        first = self.frame(registry, [self.TARGET, "a-can", "b-can"])
        index = first[self.TARGET].index

        # The target leaves the view while two other objects stay and a third
        # arrives. It is not emitted, but its slot is still its own.
        later = self.frame(registry, ["a-can", "b-can", "c-box"])
        self.assertNotIn(self.TARGET, later)
        self.assertEqual(registry.index_of(self.TARGET), index)
        for other in later.values():
            if other.node_type != "ee":
                self.assertNotEqual(other.index, index)

        # And it returns to exactly the same position.
        back = self.frame(registry, [self.TARGET, "a-can"])
        self.assertEqual(back[self.TARGET].index, index)

    def test_a_retained_target_costs_one_object_slot(self):
        registry = EntityRegistry(n_max=4)
        self.frame(registry, [self.TARGET, "a-can"])
        # Three non-targets compete for the two remaining object slots.
        current = self.frame(registry, ["a-can", "b-can", "c-can"])
        self.assertEqual(len(registry), 3)  # target plus two non-targets
        self.assertEqual(len([n for n in current.values() if n.node_type != "ee"]), 2)
        self.assertGreater(registry.overflow_drops, 0)

    def test_the_target_force_admits_after_the_table_filled_without_it(self):
        registry = EntityRegistry(n_max=4)
        # Three non-targets take every object slot before the target is seen.
        self.frame(registry, ["a-can", "b-can", "c-can"])
        self.assertEqual(len(registry), 3)
        current = self.frame(registry, [self.TARGET, "a-can", "b-can", "c-can"])
        self.assertIn(self.TARGET, current)

    def test_releasing_the_target_is_a_no_op_before_it_is_admitted(self):
        registry = EntityRegistry(n_max=4)
        # No slot sits reserved for a target that has never appeared.
        current = self.frame(registry, ["a-can", "b-can", "c-can"])
        self.assertEqual(len([n for n in current.values() if n.node_type != "ee"]), 3)
        self.assertIsNone(registry.index_of(self.TARGET))

    def test_reset_clears_the_retained_target(self):
        registry = EntityRegistry(n_max=4)
        self.frame(registry, [self.TARGET, "a-can"])
        registry.reset_episode()
        self.assertIsNone(registry.index_of(self.TARGET))
        self.assertEqual(len(registry), 0)
        self.assertEqual(registry.episode_entities, 0)


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


class EntityRegistryRetainTest(unittest.TestCase):
    """``retain`` is what keeps capacity describing the current frame.

    Without it, history-off graphs never release a slot: ``commit`` is skipped
    so ``evict_expired`` has nothing to expire, and the registry fills with
    objects that are no longer vertices.
    """

    def test_absent_resident_releases_its_slot(self):
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})

        registry.retain({"ee", "a"})

        self.assertEqual(len(registry), 1)
        self.assertIsNone(registry.index_of("b"))

    def test_absent_resident_is_forgotten_not_merely_unseated(self):
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})

        registry.retain({"ee", "a"})

        self.assertNotIn("b", registry._first_seen)

    def test_absent_rejected_instance_is_forgotten_too(self):
        # assign stamps _first_seen on every pending id, including ones that
        # overflow then rejects. Sweeping only residents would leave that age
        # behind and keep re-rejecting the instance every time it returns.
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})
        registry.assign({
            "ee": ee(), "a": node("a"), "b": node("b"), "c": node("c"),
        })
        registry.assign({
            "ee": ee(), "a": node("a"), "b": node("b"), "c": node("c"),
        })
        self.assertIn("a", registry._first_seen)  # rejected, not a resident
        self.assertIsNone(registry.index_of("a"))

        registry.retain({"ee", "b", "c"})

        self.assertNotIn("a", registry._first_seen)

    def test_returning_instance_is_admitted_after_retain(self):
        # The end-to-end failure: an instance displaced while absent used to
        # stay locked out for the rest of the episode by its stale age.
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})
        registry.assign({
            "ee": ee(), "a": node("a"), "b": node("b"), "c": node("c"),
        })
        self.assertIsNone(registry.index_of("a"))

        registry.retain({"ee", "a"})
        current = registry.assign({"ee": ee(), "a": node("a")})

        self.assertIn("a", current)

    def test_retain_is_not_an_eviction(self):
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})

        registry.retain({"ee"})

        self.assertEqual(registry.overflow_drops, 0)
        self.assertEqual(registry.evicted_ids, [])

    def test_same_frame_overflow_still_evicts_by_diversity(self):
        # retain must not disarm the real capacity gate: four objects visible
        # at once with three slots still costs one, chosen by type count.
        registry = EntityRegistry(n_max=4)
        visible = {
            "ee": ee(),
            "a-bowl": node("a-bowl", object_type="bowl"),
            "b-can-1": node("b-can-1", object_type="can"),
            "c-can-2": node("c-can-2", object_type="can"),
        }
        registry.assign(visible)
        registry.retain(set(visible))

        current = registry.assign(
            {**visible, "d-mug": node("d-mug", object_type="mug")}
        )

        self.assertIn("a-bowl", current)
        self.assertIn("d-mug", current)
        self.assertEqual(registry.evicted_ids, ["b-can-1"])
        self.assertEqual(registry.overflow_drops, 1)

    def test_retain_is_a_noop_when_nothing_is_absent(self):
        # Why history-on is unaffected beyond the builder's guard: with
        # k_persist=-1, merge_persistent re-injects every node ever seen, so
        # the keep set is a superset of everything tracked.
        registry = EntityRegistry(n_max=4)
        visible = {"ee": ee(), "a": node("a"), "b": node("b"), "c": node("c")}
        first = registry.assign(visible)
        before = {k: v.index for k, v in first.items()}
        ages = dict(registry._first_seen)

        registry.retain(set(visible))
        current = registry.assign(visible)

        self.assertEqual({k: v.index for k, v in current.items()}, before)
        self.assertEqual(registry._first_seen, ages)
        self.assertEqual(registry.overflow_drops, 0)

    def test_episode_entities_counts_distinct_instances(self):
        # Independent of retain, so it still reports whether the episode ever
        # held enough instances for the vertex budget to bind.
        registry = EntityRegistry(n_max=3)
        registry.assign({"ee": ee(), "a": node("a"), "b": node("b")})
        registry.retain({"ee"})
        registry.assign({"ee": ee(), "c": node("c")})

        self.assertEqual(registry.episode_entities, 3)

        registry.reset_episode()
        self.assertEqual(registry.episode_entities, 0)


if __name__ == "__main__":
    unittest.main()
