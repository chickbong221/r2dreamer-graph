"""Merged-view actor identity: one logical key at any num_envs.

PegInsertionSide builds one actor per sub-scene and exposes the logical object
only through Actor.merge, which registers in scene.actor_views rather than
scene.actors. Without aliasing the stable key carries the sub-scene index, so
assets mined at num_envs=1 name actor:peg_0 while a vectorised run emits keys
the entity vocabulary has never seen.
"""

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

from scenegraph.adapters.privileged_state import (
    merged_view_alias_map,
    merged_view_aliasing_enabled,
    per_env_segmentation_id_map,
    set_merged_view_aliasing,
)
from scenegraph.core.entity_identity import canonical_actor_key, stable_entity_key


class _Body:
    def __init__(self, per_scene_id):
        self.per_scene_id = per_scene_id


class Actor:  # noqa: N801 - entity_kind() dispatches on the class name
    def __init__(self, name, scene, objs, scene_idxs, merged=False):
        self.name = name
        self.scene = scene
        self._objs = list(objs)
        self._scene_idxs = list(scene_idxs)
        self.merged = merged


class _Scene:
    def __init__(self):
        self.actors = {}
        self.articulations = {}
        self.actor_views = {}


def _merge(name, scene, members):
    objs, idxs = [], []
    for m in members:
        objs += m._objs
        idxs += m._scene_idxs
    return Actor(name, scene, objs, idxs, merged=True)


def _peg_scene(num_envs, extra=()):
    """PegInsertionSide shape: peg_{i} / box_with_hole_{i} behind two views."""
    scene = _Scene()
    pegs, boxes = [], []
    sid = 1
    for i in range(num_envs):
        peg = Actor(f"peg_{i}", scene, [_Body(sid)], [i])
        box = Actor(f"box_with_hole_{i}", scene, [_Body(sid + 1)], [i])
        sid += 2
        scene.actors[peg.name] = peg
        scene.actors[box.name] = box
        pegs.append(peg)
        boxes.append(box)
    for name in extra:
        # Plain actors present in every sub-scene, in no merged view.
        a = Actor(name, scene, [_Body(sid + k) for k in range(num_envs)],
                  list(range(num_envs)))
        sid += num_envs
        scene.actors[name] = a
    scene.actor_views["peg"] = _merge("peg", scene, pegs)
    scene.actor_views["box_with_hole"] = _merge("box_with_hole", scene, boxes)
    return scene


def _keys(scene, env_idx, alias):
    seg = per_env_segmentation_id_map(
        SimpleNamespace(scene=scene), env_idx, alias_merged_views=alias
    )
    return {stable_entity_key(e) for e in seg.values()}


class MergedViewIdentityTest(unittest.TestCase):
    def test_without_aliasing_keys_carry_subscene_index(self):
        """The bug this fix exists for."""
        self.assertIn("actor:peg_0", _keys(_peg_scene(4), 0, False))
        self.assertIn("actor:peg_3", _keys(_peg_scene(4), 3, False))

    def test_aliasing_gives_one_key_across_env_counts(self):
        expected = {"actor:peg", "actor:box_with_hole"}
        self.assertEqual(_keys(_peg_scene(1), 0, True), expected)
        for env_idx in (0, 7, 125):
            self.assertEqual(_keys(_peg_scene(126), env_idx, True), expected)

    def test_collection_and_runtime_agree(self):
        """num_envs=1 mining vs num_envs=126 runtime: identical vocabulary."""
        self.assertEqual(_keys(_peg_scene(1), 0, True),
                         _keys(_peg_scene(126), 42, True))

    def test_mshab_style_instances_stay_distinct(self):
        scene = _peg_scene(2, extra=("frl_apartment_table_01",
                                     "frl_apartment_table_02"))
        keys = _keys(scene, 0, True)
        self.assertIn("actor:frl_apartment_table_01", keys)
        self.assertIn("actor:frl_apartment_table_02", keys)

    def test_canonical_actor_key_still_splits_numbered_assets(self):
        """Guards against 'fixing' this with a global _\d+$ strip instead."""
        self.assertNotEqual(canonical_actor_key("frl_apartment_table_01"),
                            canonical_actor_key("frl_apartment_table_02"))

    def test_links_and_unmerged_actors_pass_through(self):
        scene = _peg_scene(2, extra=("cubeA",))
        self.assertIn("actor:cubeA", _keys(scene, 1, True))

    def test_ambiguous_view_membership_raises(self):
        scene = _peg_scene(2)
        shared = scene.actors["peg_0"]
        scene.actor_views["other"] = _merge("other", scene, [shared])
        scene.__dict__.pop("_teemo_merged_view_alias", None)
        with self.assertRaises(ValueError):
            merged_view_alias_map(scene)

    def test_two_same_env_actors_behind_one_view_raises(self):
        scene = _Scene()
        a = Actor("part_0", scene, [_Body(1)], [0])
        b = Actor("part_1", scene, [_Body(2)], [0])
        scene.actors.update({"part_0": a, "part_1": b})
        scene.actor_views["part"] = _merge("part", scene, [a, b])
        with self.assertRaises(ValueError):
            per_env_segmentation_id_map(
                SimpleNamespace(scene=scene), 0, alias_merged_views=True
            )

    def test_flag_defaults_off_and_invalidates_cache(self):
        scene = _peg_scene(2)
        env = SimpleNamespace(scene=scene)
        self.assertFalse(merged_view_aliasing_enabled(scene))
        before = {stable_entity_key(e)
                  for e in per_env_segmentation_id_map(env, 0).values()}
        self.assertIn("actor:peg_0", before)

        set_merged_view_aliasing(scene, True)
        after = {stable_entity_key(e)
                 for e in per_env_segmentation_id_map(env, 0).values()}
        self.assertEqual(after, {"actor:peg", "actor:box_with_hole"})


def _worker_keys(num_envs, env_idx, backend):
    """Build one env and print its per-env stable keys. Subprocess entry point.

    SAPIEN enables GPU PhysX once per process, so the CPU collection shape and
    the GPU runtime shape cannot be built in one interpreter.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make("PegInsertionSide-v1", num_envs=num_envs,
                   sim_backend=backend)
    try:
        env.reset(seed=0)
        set_merged_view_aliasing(env, True)
        seg = per_env_segmentation_id_map(env, env_idx)
        keys = sorted(stable_entity_key(e) for e in seg.values())
        print("KEYS " + json.dumps(keys))
    finally:
        env.close()


@unittest.skipUnless(
    importlib.util.find_spec("mani_skill"), "needs ManiSkill"
)
class RealPegInsertionIdentityTest(unittest.TestCase):
    """Server-only: the same assertion against the real task."""

    def _keys(self, num_envs, env_idx, backend):
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--keys",
             str(num_envs), str(env_idx), backend],
            capture_output=True, text=True, timeout=900,
        )
        line = next((l for l in out.stdout.splitlines()
                     if l.startswith("KEYS ")), None)
        if line is None:
            self.fail(f"worker rc={out.returncode}\n{out.stdout}\n{out.stderr}")
        return set(json.loads(line[len("KEYS "):]))

    def test_peg_and_box_keys_match_across_env_counts(self):
        one = self._keys(1, 0, "cpu")      # collection shape
        self.assertIn("actor:peg", one)
        self.assertIn("actor:box_with_hole", one)
        many = self._keys(16, 9, "gpu")    # runtime shape
        self.assertEqual(one, many)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--keys":
        _worker_keys(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        unittest.main()
