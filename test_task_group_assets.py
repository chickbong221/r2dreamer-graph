"""Task-group isolation across the mining pipeline and the runtime gate.

The failure these guard against is silent: every file exists, parses and
validates, it just describes the wrong scene. set_table's bowl rests in a
counter drawer and prepare_groceries' on the counter, so a whitelist mined
under one task names furniture the other never puts the bowl on.

Everything here runs without ManiSkill, mshab or torch: the collector's
discovery and the miner's membership policy are pure functions over paths and
dicts, and the runtime checks are exercised through a GraphBuilder over
hand-written assets.
"""

import json
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from scenegraph.configs.loader import load_config
from scenegraph.core.graph_builder import GraphBuilder
from scenegraph.core.whitelist import (
    load_whitelist,
    resolve_whitelist_path,
    whitelist_group_dir,
)
from scenegraph.tools.build_subtask_whitelists import (
    MEMBERSHIP_FULL_EVIDENCE,
    MEMBERSHIP_TARGET_SUPPORTERS,
    _WhitelistBuilder,
)
from scenegraph.tools.build_union_whitelist import merge
from scenegraph.tools.collect_robot_success_states import (
    _already_done,
    _discover_work,
)
from scenegraph.tools.prune_whitelists import prune_payload
from scenegraph.tools import prune_whitelists


BOWL = "actor:024_bowl"
DRAWER = "link:kitchen_counter-0/drawer3"
COUNTER = "link:kitchen_counter-0/body"
BRUSHED = "actor:003_cracker_box"


def _member(roles, itypes, supports=None, kind="link"):
    entry = {"roles": sorted(roles), "interaction_types": sorted(itypes),
             "kind": kind}
    if supports:
        entry["supports"] = sorted(supports)
    return entry


def _whitelist(task_group, supporter, *, target=BOWL, extra_members=None,
               policy=MEMBERSHIP_TARGET_SUPPORTERS):
    members = {
        target: _member(["interacted"], ["contact", "grasp"], kind="actor"),
        supporter: _member(["interacted", "support"], ["contact", "support"],
                           supports=[target]),
    }
    members.update(extra_members or {})
    return {
        "_schema_version": 4,
        "subtask": "pick",
        "task_group": task_group,
        "membership_policy": policy,
        "target": target,
        "members": members,
        "bin_edges": {
            "planar-distance": [0.15, 0.30, 1.00, 1.50],
            "height-offset": [-0.15, -0.05, 0.05, 0.15],
            "grasp-compatibility": [1 / 3, 2 / 3],
            "contact-compatibility": [1 / 3, 2 / 3],
            "support-compatibility": [1 / 3, 2 / 3],
            "contain-compatibility": [1 / 3, 2 / 3],
        },
        "bin_stats_robust": {"planar_distance": 1.5, "height_offset": 0.3},
        "_n_successful_rollouts": 30,
    }


class TempTree(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, path: Path, payload) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return path


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
class TestCollectorDiscovery(TempTree):
    """One checkpoint per (task, object), never collapsed to the object."""

    def _ckpt_tree(self):
        root = self.tmp / "rl"
        for task in ("prepare_groceries", "set_table", "tidy_house"):
            for obj in ("024_bowl", "013_apple"):
                pt = root / task / "pick" / obj / "policy.pt"
                pt.parent.mkdir(parents=True, exist_ok=True)
                pt.write_text("")
        return root

    def test_same_object_in_two_groups_is_two_units_of_work(self):
        work = _discover_work(self._ckpt_tree(), "pick", [], [])
        bowls = [(task, obj) for task, obj, _ in work if obj == "024_bowl"]
        self.assertEqual(
            sorted(bowls),
            [("prepare_groceries", "024_bowl"), ("set_table", "024_bowl"),
             ("tidy_house", "024_bowl")],
        )

    def test_task_filter_selects_one_group(self):
        work = _discover_work(self._ckpt_tree(), "pick", ["set_table"], [])
        self.assertEqual({task for task, _, _ in work}, {"set_table"})

    def test_rollouts_land_in_per_group_files(self):
        # The two groups' bowl rollouts must not be able to name the same
        # pickle -- that is what let the last task collected overwrite the rest.
        asset = self.tmp / "data"
        paths = set()
        for task in ("set_table", "prepare_groceries"):
            pkl = (asset / "robot_success_states" / "fetch" / task / "pick"
                   / "024_bowl.pkl")
            pkl.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl, "wb") as stream:
                pickle.dump({
                    "_schema_version": 8,
                    "robot_qpos": [[0.0]],
                    "tcp_pose_wrt_base": [[0.0]],
                    "interaction_rollouts": [{"target_key": BOWL}],
                    "provenance": {"task_group": task},
                }, stream)
            paths.add(str(pkl))
            self.assertTrue(_already_done(asset, task, "pick", "024_bowl", 1))
        self.assertEqual(len(paths), 2)
        # A group with no collection of its own is not covered by another's.
        self.assertFalse(_already_done(asset, "tidy_house", "pick", "024_bowl", 1))


# --------------------------------------------------------------------------- #
# Membership policy
# --------------------------------------------------------------------------- #
class TestMembershipPolicy(unittest.TestCase):
    """full-evidence preserves what target-supporters would throw away."""

    def _builder(self, policy):
        builder = _WhitelistBuilder(
            "pick", BOWL, task_group="set_table", membership_policy=policy)
        # The arm rested the bowl on a drawer and brushed a cracker box on the
        # way out. Only the drawer is part of the task.
        builder.roles[BOWL].add("interacted")
        builder.roles[DRAWER].update({"interacted", "support"})
        builder.roles[BRUSHED].add("interacted")
        builder.supports[DRAWER].add(BOWL)
        return builder

    def test_target_supporters_drops_the_brushed_past_object(self):
        admitted = self._builder(MEMBERSHIP_TARGET_SUPPORTERS)._admitted()
        self.assertEqual(admitted, {BOWL, DRAWER})

    def test_full_evidence_keeps_every_interacted_entity(self):
        admitted = self._builder(MEMBERSHIP_FULL_EVIDENCE)._admitted()
        self.assertEqual(admitted, {BOWL, DRAWER, BRUSHED})

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            _WhitelistBuilder("pick", BOWL, membership_policy="keep-everything")


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #
class TestPruning(TempTree):
    def _raw(self):
        return _whitelist(
            "set_table", DRAWER,
            extra_members={BRUSHED: _member(["interacted"], ["contact"],
                                            kind="actor")},
            policy=MEMBERSHIP_FULL_EVIDENCE,
        )

    def test_pruning_reproduces_the_runtime_membership(self):
        pruned = prune_payload(self._raw(), MEMBERSHIP_TARGET_SUPPORTERS)
        self.assertEqual(set(pruned["members"]), {BOWL, DRAWER})
        self.assertEqual(pruned["membership_policy"],
                         MEMBERSHIP_TARGET_SUPPORTERS)

    def test_pruning_leaves_no_dangling_support_reference(self):
        raw = self._raw()
        raw["members"][DRAWER]["supports"] = [BOWL, BRUSHED]
        pruned = prune_payload(raw, MEMBERSHIP_TARGET_SUPPORTERS)
        self.assertEqual(pruned["members"][DRAWER]["supports"], [BOWL])

    def test_full_evidence_prune_is_membership_preserving(self):
        raw = self._raw()
        pruned = prune_payload(raw, MEMBERSHIP_FULL_EVIDENCE)
        self.assertEqual(set(pruned["members"]), set(raw["members"]))

    def test_pruning_does_not_move_the_relation_bins(self):
        raw = self._raw()
        pruned = prune_payload(raw, MEMBERSHIP_TARGET_SUPPORTERS)
        self.assertEqual(pruned["bin_edges"], raw["bin_edges"])
        self.assertEqual(pruned["bin_stats_robust"], raw["bin_stats_robust"])

    def test_raw_evidence_survives_the_prune_on_disk(self):
        raw_dir = self.tmp / "raw" / "set_table"
        out_dir = self.tmp / "runtime" / "set_table"
        source = self._write(raw_dir / "pick_024_bowl.json", self._raw())
        before = source.read_text()

        code = prune_whitelists.main([
            "--raw-dir", str(raw_dir), "--out-dir", str(out_dir),
            "--task-group", "set_table", "--subtask", "pick",
        ])
        self.assertEqual(code, 0)
        # The expensive artefact is untouched and still holds the evidence the
        # runtime file dropped, so a different policy costs a re-prune.
        self.assertEqual(source.read_text(), before)
        self.assertIn(BRUSHED, json.loads(source.read_text())["members"])
        runtime = json.loads((out_dir / "pick_024_bowl.json").read_text())
        self.assertNotIn(BRUSHED, runtime["members"])
        self.assertTrue((out_dir / "pick_all.json").is_file())

    def test_pruning_in_place_is_refused(self):
        raw_dir = self.tmp / "raw" / "set_table"
        self._write(raw_dir / "pick_024_bowl.json", self._raw())
        code = prune_whitelists.main([
            "--raw-dir", str(raw_dir), "--out-dir", str(raw_dir),
            "--subtask", "pick",
        ])
        self.assertEqual(code, 2)

    def test_pruning_refuses_a_file_from_another_group(self):
        raw_dir = self.tmp / "raw" / "set_table"
        self._write(raw_dir / "pick_024_bowl.json",
                    _whitelist("prepare_groceries", COUNTER))
        code = prune_whitelists.main([
            "--raw-dir", str(raw_dir), "--out-dir", str(self.tmp / "out"),
            "--task-group", "set_table", "--subtask", "pick",
        ])
        self.assertEqual(code, 1)


# --------------------------------------------------------------------------- #
# Union
# --------------------------------------------------------------------------- #
class TestUnionStaysWithinOneGroup(TempTree):
    def test_union_records_the_group_it_merged(self):
        group = self.tmp / "set_table"
        self._write(group / "pick_024_bowl.json", _whitelist("set_table", DRAWER))
        self._write(group / "pick_013_apple.json",
                    _whitelist("set_table", "link:fridge-0/body",
                               target="actor:013_apple"))
        data = merge(group, "pick")
        self.assertEqual(data["task_group"], "set_table")
        self.assertEqual(sorted(data["_merged_from"]),
                         ["pick_013_apple.json", "pick_024_bowl.json"])
        self.assertIn(DRAWER, data["members"])

    def test_union_refuses_to_merge_two_groups(self):
        mixed = self.tmp / "mixed"
        self._write(mixed / "pick_024_bowl.json", _whitelist("set_table", DRAWER))
        self._write(mixed / "pick_003_cracker_box.json",
                    _whitelist("prepare_groceries", COUNTER,
                               target=BRUSHED))
        # Bins widened by another task's scene would stretch this task's
        # relation tokens past anything its own demonstrations produced.
        with self.assertRaises(ValueError) as ctx:
            merge(mixed, "pick")
        self.assertIn("mixes task groups", str(ctx.exception))

    def test_union_refuses_unlabelled_files(self):
        legacy = self.tmp / "legacy"
        payload = _whitelist("set_table", DRAWER)
        payload.pop("task_group")
        self._write(legacy / "pick_024_bowl.json", payload)
        with self.assertRaises(ValueError):
            merge(legacy, "pick")


# --------------------------------------------------------------------------- #
# Runtime selection
# --------------------------------------------------------------------------- #
class TestRuntimeSelection(TempTree):
    def _two_groups(self) -> Path:
        root = self.tmp / "subtask_whitelists"
        self._write(root / "set_table" / "pick_024_bowl.json",
                    _whitelist("set_table", DRAWER))
        self._write(root / "prepare_groceries" / "pick_024_bowl.json",
                    _whitelist("prepare_groceries", COUNTER))
        return root

    def test_group_directory_picks_the_matching_scene(self):
        root = self._two_groups()
        for group, supporter in (("set_table", DRAWER),
                                 ("prepare_groceries", COUNTER)):
            path = resolve_whitelist_path(
                whitelist_group_dir(str(root), group), "pick", BOWL)
            self.assertIsNotNone(path)
            whitelist = load_whitelist(path)
            self.assertEqual(whitelist.task_group, group)
            self.assertTrue(whitelist.contains(supporter))

        # The distinguishing supporter of each group is absent from the other,
        # which is the whole reason the directories are separate.
        set_table = load_whitelist(resolve_whitelist_path(
            whitelist_group_dir(str(root), "set_table"), "pick", BOWL))
        self.assertFalse(set_table.contains(COUNTER))

    def test_group_directory_needs_a_group(self):
        self.assertIsNone(whitelist_group_dir(str(self.tmp), ""))
        self.assertIsNone(whitelist_group_dir("", "set_table"))

    def _builder(self, group: str, root: Path) -> GraphBuilder:
        cfg = {
            "temporal": {"K": 2},
            "selection": {"n_max": 8, "k_persist": -1},
            "whitelist_dir": whitelist_group_dir(str(root), group),
            "task_group": group,
        }
        return GraphBuilder(object(), cfg, uid_vocab=16)

    def test_builder_refuses_a_whitelist_from_another_group(self):
        root = self._two_groups()
        # The file is physically present in the set_table tree but records
        # another group -- a hand-copy that every other check would accept.
        self._write(root / "set_table" / "pick_003_cracker_box.json",
                    _whitelist("prepare_groceries", COUNTER, target=BRUSHED))
        builder = self._builder("set_table", root)
        whitelist = load_whitelist(
            str(root / "set_table" / "pick_003_cracker_box.json"))
        with self.assertRaises(ValueError) as ctx:
            builder._check_task_group(whitelist, "pick_003_cracker_box.json")
        self.assertIn("prepare_groceries", str(ctx.exception))

    def test_builder_accepts_its_own_group(self):
        root = self._two_groups()
        builder = self._builder("set_table", root)
        whitelist = load_whitelist(
            str(root / "set_table" / "pick_024_bowl.json"))
        builder._check_task_group(whitelist, "pick_024_bowl.json")


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
class TestLoaderResolvesPerGroup(unittest.TestCase):
    def test_both_assets_resolve_under_the_group(self):
        cfg = load_config(task_group="set_table", require_assets=False)
        self.assertEqual(cfg["task_group"], "set_table")
        self.assertEqual(os.path.basename(cfg["whitelist_dir"]), "set_table")
        self.assertEqual(
            os.path.basename(cfg["whitelists"]["root_abs"]),
            "subtask_whitelists",
        )
        self.assertTrue(
            cfg["affordances"]["asset_path_abs"].endswith("set_table.json"))

    def test_two_groups_never_resolve_to_the_same_asset(self):
        a = load_config(task_group="set_table", require_assets=False)
        b = load_config(task_group="tidy_house", require_assets=False)
        self.assertNotEqual(a["whitelist_dir"], b["whitelist_dir"])
        self.assertNotEqual(a["affordances"]["asset_path_abs"],
                            b["affordances"]["asset_path_abs"])

    def test_required_assets_need_a_group(self):
        with self.assertRaises(ValueError):
            load_config(require_assets=True)


if __name__ == "__main__":
    unittest.main()
