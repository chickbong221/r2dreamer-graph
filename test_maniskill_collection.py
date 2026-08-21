"""Scene extraction and success-gated evidence for normal ManiSkill."""

import unittest
from types import SimpleNamespace

from scenegraph.adapters.interaction_events import (
    EE_KEY, BucketStore, DiscoveryWindow, EpisodeEvidence, InteractionEvent,
    make_bucket,
)
from scenegraph.adapters.maniskill_scene import scene_entities, scene_entity_keys
from scenegraph.adapters.privileged_state import set_merged_view_aliasing
from test_merged_view_identity import Actor, _Body, _Scene, _merge


def _env(scene, actors, ground=None, table=None, robot_links=(), views=()):
    for a in actors:
        scene.actors[a.name] = a
    for v in views:
        scene.actor_views[v.name] = v
    builder = SimpleNamespace(ground=ground, table=table,
                              scene_objects=[x for x in (table, ground) if x])
    robot = SimpleNamespace(get_links=lambda: list(robot_links))
    return SimpleNamespace(scene=scene, table_scene=builder,
                           agent=SimpleNamespace(robot=robot))


def _actor(scene, name, sid, idxs=(0,)):
    return Actor(name, scene, [_Body(sid)], list(idxs))


def _tabletop():
    """PickCube-like: ground, table, cube, plus one robot link."""
    scene = _Scene()
    ground = _actor(scene, "ground", 1)
    table = _actor(scene, "table-workspace", 2)
    cube = _actor(scene, "cube", 3)
    link = _actor(scene, "panda_hand", 4)
    env = _env(scene, [ground, table, cube, link], ground=ground,
               table=table, robot_links=[link])
    return env


class SceneExtractionTest(unittest.TestCase):
    def test_ground_excluded_table_kept(self):
        keys = scene_entity_keys(_tabletop())
        self.assertIn("actor:table-workspace", keys)
        self.assertIn("actor:cube", keys)
        self.assertNotIn("actor:ground", keys)

    def test_robot_links_excluded(self):
        self.assertNotIn("actor:panda_hand", scene_entity_keys(_tabletop()))

    def test_ground_excluded_by_identity_when_renamed(self):
        scene = _Scene()
        ground = _actor(scene, "floor_plane", 1)
        cube = _actor(scene, "cube", 2)
        env = _env(scene, [ground, cube], ground=ground)
        self.assertEqual(scene_entity_keys(env), ["actor:cube"])

    def test_kinematic_receptacles_are_kept(self):
        scene = _Scene()
        ground = _actor(scene, "ground", 1)
        env = _env(scene, [ground, _actor(scene, "bin", 2),
                           _actor(scene, "sphere", 3)], ground=ground)
        self.assertEqual(scene_entity_keys(env), ["actor:bin", "actor:sphere"])

    def test_merged_views_resolve_when_aliasing_is_on(self):
        scene = _Scene()
        ground = _actor(scene, "ground", 1)
        pegs = [_actor(scene, f"peg_{i}", 2 + i, idxs=(i,)) for i in range(4)]
        env = _env(scene, [ground, *pegs], ground=ground,
                   views=[_merge("peg", scene, pegs)])
        set_merged_view_aliasing(scene, True)
        for env_idx in range(4):
            self.assertEqual(scene_entity_keys(env, env_idx), ["actor:peg"])

    def test_dedup_by_stable_key(self):
        env = _tabletop()
        self.assertEqual(len(scene_entities(env)),
                         len(set(scene_entity_keys(env))))


class BucketKeyTest(unittest.TestCase):
    def test_symmetric_contact_is_stored_once(self):
        a = make_bucket("contact", "actor:cubeB", "actor:cubeA")
        b = make_bucket("contact", "actor:cubeA", "actor:cubeB")
        self.assertEqual(a, b)
        self.assertEqual((a.src, a.dst), ("actor:cubeA", "actor:cubeB"))

    def test_directed_relations_keep_orientation(self):
        s = make_bucket("support", "actor:cubeB", "actor:cubeA")
        self.assertEqual((s.src, s.dst), ("actor:cubeB", "actor:cubeA"))
        self.assertNotEqual(s, make_bucket("support", "actor:cubeA",
                                           "actor:cubeB"))

    def test_ee_stays_the_source_of_symmetric_contact(self):
        b = make_bucket("contact", EE_KEY, "actor:cube")
        self.assertEqual((b.src, b.dst), (EE_KEY, "actor:cube"))


class SuccessGateTest(unittest.TestCase):
    def _episode(self, buckets, success):
        ep = EpisodeEvidence()
        for i, b in enumerate(buckets):
            ep.add(InteractionEvent(b, i))
        ep.observe_success(success)
        return ep

    def test_failed_episode_contributes_nothing(self):
        store = BucketStore(target=300)
        n = self._episode([make_bucket("grasp", EE_KEY, "actor:cube")],
                          False).commit(store)
        self.assertEqual(n, 0)
        self.assertEqual(store.buckets(), [])
        self.assertEqual(store.episodes, 0)

    def test_success_anywhere_in_the_episode_admits_earlier_frames(self):
        store = BucketStore()
        ep = EpisodeEvidence()
        ep.add(InteractionEvent(make_bucket("contact", EE_KEY, "actor:cube"), 0))
        ep.observe_success(False)
        ep.add(InteractionEvent(make_bucket("grasp", EE_KEY, "actor:cube"), 9))
        ep.observe_success(True)
        self.assertEqual(ep.commit(store), 2)

    def test_commit_resets_the_buffer(self):
        store = BucketStore()
        ep = self._episode([make_bucket("grasp", EE_KEY, "actor:cube")], True)
        ep.commit(store)
        self.assertEqual(len(ep), 0)
        self.assertFalse(ep.success_once)

    def test_bucket_caps_at_target(self):
        store = BucketStore(target=3)
        b = make_bucket("grasp", EE_KEY, "actor:cube")
        for _ in range(10):
            self._episode([b], True).commit(store)
        self.assertEqual(len(store.samples[b]), 3)
        self.assertEqual(store.seen_counts[b], 10)
        self.assertTrue(store.is_done())

    def test_incomplete_buckets_are_reported_not_dropped(self):
        store = BucketStore(target=5)
        rare = make_bucket("contain", "actor:box", "actor:peg")
        self._episode([rare], True).commit(store)
        self.assertIn(rare, store.incomplete())
        self.assertFalse(store.is_done())
        self.assertIn("SHORT", store.report())

    def test_freeze_rejects_late_buckets(self):
        store = BucketStore(target=5)
        known = make_bucket("grasp", EE_KEY, "actor:cube")
        self._episode([known], True).commit(store)
        store.freeze()
        late = make_bucket("contact", "actor:cube", "actor:table-workspace")
        self._episode([known, late], True).commit(store)
        self.assertNotIn(late, store.samples)
        self.assertEqual(store.late[late], 1)
        self.assertIn("LATE", store.report())


class DiscoveryWindowTest(unittest.TestCase):
    def test_settles_after_patience_without_new_buckets(self):
        w = DiscoveryWindow(patience=3)
        b = make_bucket("grasp", EE_KEY, "actor:cube")
        w.observe([b])
        self.assertFalse(w.settled)
        for _ in range(3):
            w.observe([b])
        self.assertTrue(w.settled)

    def test_a_new_bucket_restarts_the_window(self):
        w = DiscoveryWindow(patience=2)
        b1 = make_bucket("grasp", EE_KEY, "actor:cube")
        b2 = make_bucket("support", "actor:table-workspace", "actor:cube")
        for _ in range(3):
            w.observe([b1])
        self.assertTrue(w.settled)
        w.observe([b1, b2])
        self.assertFalse(w.settled)


if __name__ == "__main__":
    unittest.main()


from scenegraph.tools.collect_maniskill_interactions import InteractionRecorder


class _FakeState:
    """Minimal PrivilegedState stand-in: forces and grasp come from tables."""

    def __init__(self, ee_force=None, grasped=(), pair_forces=None):
        self.tcp_pose_world = [0.0, 0, 0, 1.0, 0, 0, 0]
        self.gripper_width = 0.04
        self.ee_force = ee_force or {}
        self.grasped = set(grasped)
        self.pair_forces = pair_forces or {}

    def ee_object_contact_force(self, ent):
        return self.ee_force.get(ent.name, 0.0)

    def is_grasping(self, ent, max_angle=30):
        return ent.name in self.grasped

    def pairwise_force_vector(self, a, b):
        import numpy as np
        return np.asarray(
            self.pair_forces.get((a.name, b.name))
            or self.pair_forces.get((b.name, a.name))
            or [0.0, 0.0, 0.0], dtype=float)


def _recorder(env=None):
    env = env or _tabletop()   # ground (filtered), table-workspace, cube
    rec = InteractionRecorder(env)
    rec.on_env_reset()
    rec.observe(None, _FakeState())   # warm-up capture, emits nothing
    return rec


class RecorderDetectionTest(unittest.TestCase):
    def _buckets(self, rec):
        rec.finalize_episode()      # groups only emit when they close
        return {str(e.bucket) for e in rec.episode.events}

    def test_scene_excludes_ground_and_robot(self):
        rec = _recorder()
        self.assertEqual(sorted(rec.keys.values()),
                         ["actor:cube", "actor:table-workspace"])

    def test_ee_contact_and_grasp(self):
        rec = _recorder()
        rec.observe({"success": [False]},
                    _FakeState(ee_force={"cube": 1.0}, grasped=["cube"]))
        self.assertEqual(self._buckets(rec), {
            "contact / ee / actor:cube", "grasp / ee / actor:cube"})

    def test_below_eps_force_emits_nothing(self):
        rec = _recorder()
        rec.observe({}, _FakeState(ee_force={"cube": 0.01}))
        self.assertEqual(self._buckets(rec), set())

    def test_table_supports_cube(self):
        rec = _recorder()
        # force on table due to cube points down -> table carries cube
        rec.observe({}, _FakeState(
            pair_forces={("table-workspace", "cube"): [0.0, 0.0, -1.0]}))
        self.assertIn("support / actor:table-workspace / actor:cube",
                      self._buckets(rec))

    def test_support_direction_follows_force_sign(self):
        rec = _recorder()
        rec.observe({}, _FakeState(
            pair_forces={("table-workspace", "cube"): [0.0, 0.0, 1.0]}))
        self.assertIn("support / actor:cube / actor:table-workspace",
                      self._buckets(rec))

    def test_sideways_contact_is_not_support(self):
        rec = _recorder()
        rec.observe({}, _FakeState(
            pair_forces={("table-workspace", "cube"): [1.0, 0.0, 0.0]}))
        buckets = self._buckets(rec)
        self.assertTrue(any(b.startswith("contact /") for b in buckets))
        self.assertFalse(any(b.startswith("support /") for b in buckets))

    def test_object_contact_stored_in_canonical_order(self):
        rec = _recorder()
        rec.observe({}, _FakeState(
            pair_forces={("table-workspace", "cube"): [1.0, 0.0, 0.0]}))
        self.assertIn("contact / actor:cube / actor:table-workspace",
                      self._buckets(rec))

    def test_failed_episode_commits_nothing(self):
        rec = _recorder()
        rec.observe({"success": [False]},
                    _FakeState(ee_force={"cube": 1.0}, grasped=["cube"]))
        rec.finalize_episode()
        store = BucketStore(target=5)
        self.assertEqual(rec.episode.commit(store), 0)
        self.assertEqual(store.buckets(), [])

    def test_successful_episode_commits_one_sample_per_group(self):
        rec = _recorder()
        rec.observe({}, _FakeState(ee_force={"cube": 1.0}))
        rec.observe({"success": [True]},
                    _FakeState(ee_force={"cube": 1.0}, grasped=["cube"]))
        rec.finalize_episode()
        store = BucketStore(target=5)
        # Two contact frames are one group, plus one grasp interval.
        self.assertEqual(rec.episode.commit(store), 2)
        self.assertEqual(store.episodes, 1)

    def test_reset_clears_the_buffer(self):
        rec = _recorder()
        rec.observe({"success": [True]}, _FakeState(ee_force={"cube": 1.0}))
        rec.reset_episode()
        self.assertEqual(len(rec.episode), 0)
        self.assertFalse(rec.episode.success_once)
        self.assertEqual(rec.frame, 0)


class ReconfigureTest(unittest.TestCase):
    """PegInsertionSide sets reconfiguration_freq=1 at num_envs=1, so every
    reset destroys the actors and drops the scene-level aliasing flag."""

    def _pegs(self, sid):
        scene = _Scene()
        ground = _actor(scene, "ground", sid)
        pegs = [_actor(scene, "peg_0", sid + 1)]
        env = _env(scene, [ground, *pegs], ground=ground,
                   views=[_merge("peg", scene, pegs)])
        return env

    def test_entities_are_recaptured_after_reset(self):
        env = self._pegs(1)
        rec = _recorder(env)
        first = list(rec.entities)

        env.scene = self._pegs(10).scene     # reconfigure: brand-new actors
        env.table_scene = SimpleNamespace(
            ground=[a for a in env.scene.actors.values()
                    if a.name == "ground"][0])
        rec.on_env_reset()
        rec.observe(None, _FakeState())

        self.assertEqual(sorted(rec.keys.values()), ["actor:peg"])
        self.assertFalse(set(map(id, first)) & set(map(id, rec.entities)))

    def test_aliasing_survives_a_rebuilt_scene(self):
        env = self._pegs(1)
        rec = _recorder(env)
        env.scene = self._pegs(10).scene     # fresh scene, no flag on it
        env.table_scene = SimpleNamespace(
            ground=[a for a in env.scene.actors.values()
                    if a.name == "ground"][0])
        rec.on_env_reset()
        rec.observe(None, _FakeState())
        self.assertEqual(list(rec.keys.values()), ["actor:peg"])

    def test_stale_capture_would_have_been_empty(self):
        """Documents the bug: without recapture the old actors are used."""
        env = self._pegs(1)
        rec = _recorder(env)
        stale = list(rec.entities)
        env.scene = self._pegs(10).scene
        self.assertFalse(
            set(map(id, stale)) & set(map(id, env.scene.actors.values())))


from scenegraph.adapters.interaction_events import GroupAccumulator

_B = make_bucket("contact", "actor:a", "actor:b")


def _run(acc, frames, forces=None, payloads=None):
    """Feed positive frames, ticking every step in between."""
    out = []
    for step in range(max(frames) + 1 + acc.gap):
        if step in frames:
            i = frames.index(step)
            pay = dict(payloads[i]) if payloads else {}
            pay.setdefault("force", forces[i] if forces else 1.0)
            acc.observe(_B, step, pay)
        out.extend(acc.tick(step))
    return out


class GroupAccumulatorTest(unittest.TestCase):
    def test_single_frame_touch_is_a_group(self):
        samples = _run(GroupAccumulator("peak"), [3])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].payload["duration"], 1)
        self.assertEqual(samples[0].payload["n_frames"], 1)

    def test_short_gap_stays_one_group(self):
        samples = _run(GroupAccumulator("peak", gap=5), [0, 1, 4, 5])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].payload["n_frames"], 4)

    def test_five_blank_steps_split_the_group(self):
        samples = _run(GroupAccumulator("peak", gap=5), [0, 1, 10, 11])
        self.assertEqual(len(samples), 2)
        self.assertEqual([s.payload["n_frames"] for s in samples], [2, 2])

    def test_groups_are_never_averaged_together(self):
        samples = _run(GroupAccumulator("peak", gap=5), [0, 20],
                       forces=[1.0, 9.0])
        self.assertEqual([s.payload["peak_force"] for s in samples],
                         [1.0, 9.0])

    def test_peak_frame_supplies_orientation(self):
        payloads = [{"obj_pose": [float(i)] * 7} for i in range(5)]
        samples = _run(GroupAccumulator("peak", gap=5, window=5),
                       [0, 1, 2, 3, 4], forces=[1, 1, 9, 1, 1],
                       payloads=payloads)
        self.assertEqual(samples[0].payload["obj_pose"], [2.0] * 7)
        self.assertEqual(samples[0].payload["peak_frame"], 2)

    def test_window_caps_the_averaged_frames(self):
        samples = _run(GroupAccumulator("peak", gap=5, window=3),
                       list(range(10)))
        self.assertEqual(samples[0].payload["n_averaged"], 3)
        self.assertEqual(samples[0].payload["n_frames"], 10)

    def test_vectors_are_averaged_and_normals_renormalized(self):
        payloads = [{"contact_position": [0.0, 0, 0],
                     "contact_normal": [3.0, 0, 0]},
                    {"contact_position": [2.0, 0, 0],
                     "contact_normal": [0.0, 4.0, 0]}]
        samples = _run(GroupAccumulator("peak", gap=5), [0, 1],
                       payloads=payloads)
        pay = samples[0].payload
        self.assertEqual(pay["contact_position"], [1.0, 0.0, 0.0])
        norm = sum(x * x for x in pay["contact_normal"]) ** 0.5
        self.assertAlmostEqual(norm, 1.0)

    def test_grasp_keeps_the_last_frame_before_release(self):
        payloads = [{"gripper_width": float(i)} for i in range(4)]
        samples = _run(GroupAccumulator("last", gap=5), [0, 1, 2, 3],
                       payloads=payloads)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].payload["gripper_width"], 3.0)

    def test_two_grasp_intervals_give_two_samples(self):
        samples = _run(GroupAccumulator("last", gap=5), [0, 1, 20, 21])
        self.assertEqual(len(samples), 2)

    def test_close_all_finalizes_a_grasp_held_at_success(self):
        acc = GroupAccumulator("last", gap=5)
        acc.observe(_B, 0, {"force": 1.0, "gripper_width": 0.02})
        acc.tick(0)
        self.assertEqual(acc.tick(1), [])       # still open
        samples = acc.close_all()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].payload["gripper_width"], 0.02)


class PresenceTest(unittest.TestCase):
    def _commit(self, store, buckets, success=True):
        ep = EpisodeEvidence()
        for i, b in enumerate(buckets):
            ep.add(InteractionEvent(b, i))
        ep.observe_success(success)
        ep.commit(store)

    def test_presence_separates_brush_from_interaction(self):
        store = BucketStore(target=1000)
        always = make_bucket("grasp", EE_KEY, "actor:cubeA")
        brush = make_bucket("contact", EE_KEY, "actor:cubeB")
        for i in range(10):
            self._commit(store, [always] + ([brush] if i == 0 else []))
        self.assertEqual(store.presence(always), 1.0)
        self.assertAlmostEqual(store.presence(brush), 0.1)
        self.assertEqual(store.incidental(0.2), [brush])

    def test_failed_episodes_do_not_count_toward_presence(self):
        store = BucketStore()
        b = make_bucket("grasp", EE_KEY, "actor:cube")
        self._commit(store, [b], success=False)
        self.assertEqual(store.episodes, 0)
        self.assertEqual(store.presence(b), 0.0)


class IncidentalExclusionTest(unittest.TestCase):
    """A bucket seen in 1 episode of 25 cannot reach 300, so it must not
    hold the run open."""

    def _store(self, episodes=25, brush_every=25, target=5):
        store = BucketStore(target=target)
        always = make_bucket("grasp", EE_KEY, "actor:cubeA")
        brush = make_bucket("contact", EE_KEY, "actor:cubeB")
        for i in range(episodes):
            ep = EpisodeEvidence()
            ep.add(InteractionEvent(always, i))
            if i % brush_every == 0:
                ep.add(InteractionEvent(brush, i))
            ep.observe_success(True)
            ep.commit(store)
        return store, always, brush

    def test_run_would_never_finish_without_exclusion(self):
        store, _, brush = self._store()
        store.freeze(min_presence=0.0)
        self.assertIn(brush, store.incomplete())
        self.assertFalse(store.is_done())

    def test_freeze_excludes_incidental_and_lets_the_run_finish(self):
        store, always, brush = self._store()
        store.freeze(min_presence=0.2)
        self.assertIn(brush, store.excluded)
        self.assertNotIn(brush, store.incomplete())
        self.assertIn(always, store.complete_buckets())
        self.assertTrue(store.is_done())

    def test_excluded_buckets_stop_collecting_but_stay_reported(self):
        store, _, brush = self._store()
        store.freeze(min_presence=0.2)
        before = len(store.samples[brush])
        ep = EpisodeEvidence()
        ep.add(InteractionEvent(brush, 99))
        ep.observe_success(True)
        ep.commit(store)
        self.assertEqual(len(store.samples[brush]), before)
        self.assertEqual(store.late[brush], 0)   # excluded, not "late"
        self.assertIn("SKIP", store.report())

    def test_excluded_never_reaches_the_whitelist(self):
        store, always, brush = self._store()
        store.freeze(min_presence=0.2)
        self.assertEqual(store.complete_buckets(), [always])

    def test_frequent_buckets_are_not_excluded(self):
        store, always, _ = self._store(brush_every=1)
        store.freeze(min_presence=0.2)
        self.assertEqual(store.excluded, {})
        self.assertEqual(len(store.complete_buckets()), 2)
