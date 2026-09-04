"""The live potential probe scores what training would score.

The probe is the only thing that answers "does a real success end at 1.0", so
a probe that scores a *different* graph than training packs answers nothing.
Two ways it did:

* it matched the schedule's ``$active_target`` sentinel against real entity
  ids, which never match, so every fact about the target was dropped before
  scoring -- and the Pick schedule is almost entirely facts about the target;
* it never packed, so the protected rows the sentinel resolves through were
  never checked.

Everything here runs against the shipped tidy_house assets and the frozen
Pick schedule, with the real vocabularies -- a fixture with invented ids can
agree with itself while disagreeing with the run. The row contract needs no
torch; the scoring does, so it runs on the server with the rest of the stack.
"""

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scenegraph.adapters.graph_pack import pack_graph
from scenegraph.adapters.graph_vocab import build_graph_vocab
from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.core.schedule import (
    ACTIVE_TARGET_ENTITY_ID,
    compile_from_source,
    mshab_schedule_source,
)
from scenegraph.core.sites import SITE_EE_REST

CONFIGS = "scenegraph/configs"
WHITELIST_DIR = f"{CONFIGS}/subtask_whitelists/tidy_house"
TARGET = "actor:004_sugar_box"
COUNTER = "link:kitchen_counter-0/body"


def _vocab():
    """The run's own vocabulary, built the way the runtime builds it."""
    return build_graph_vocab(WHITELIST_DIR)


def _pose(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def _graph(drop=()):
    """The successful frame: holding the target, back at the rest site.

    Carries every fact the terminal phase names, because a phase whose slots
    are not all readable is not scored -- and the terminal one has no later
    phase to inherit credit from.
    """
    ee = Node(node_id="ee", node_type="ee", name="ee",
              pose_world=_pose(z=1.0), attributes={})
    target = Node(node_id=TARGET, node_type="object", name="obj_0",
                  pose_world=_pose(z=0.98),
                  attributes={"whitelist_key": TARGET})
    site = Node(node_id=SITE_EE_REST, node_type="object", name=SITE_EE_REST,
                pose_world=_pose(z=1.0),
                attributes={"whitelist_key": SITE_EE_REST, "is_site": True})
    counter = Node(node_id=COUNTER, node_type="object", name="env-0_body",
                   pose_world=_pose(), attributes={"whitelist_key": COUNTER})
    edges = [
        Edge("ee", TARGET, "planar-distance", "very-near"),
        Edge("ee", TARGET, "height-offset", "level"),
        Edge("ee", TARGET, "grasp-compatibility", "match"),
        Edge("ee", TARGET, "grasp", "holds"),
        Edge("ee", SITE_EE_REST, "planar-distance", "very-near"),
        Edge("ee", SITE_EE_REST, "height-offset", "level"),
        Edge("ee", SITE_EE_REST, "reached", "holds"),
    ]
    return Graph(
        frame=0, env_id="PickSubtaskTrain-v0", camera="fetch_head",
        nodes=[counter, site, target, ee],       # deliberately out of order
        edges=[e for e in edges if e.relation not in drop],
        meta={"active_target_node_id": TARGET,
              "protected_site_node_id": SITE_EE_REST})


class PackedRowsAreWhatTheScorerReadsTest(unittest.TestCase):
    """No torch: the packing contract alone."""

    def _packed(self, graph=None, n_max=6, e_max=12):
        return pack_graph(graph if graph is not None else _graph(), _vocab(),
                          n_max=n_max, e_max=e_max, n_cams=2)

    def test_the_protected_rows_hold(self):
        packed, vocab = self._packed(), _vocab()
        self.assertEqual(int(packed["graph_node_ent"][0]), vocab.entity.ee_id)
        self.assertEqual(int(packed["graph_node_ent"][1]),
                         vocab.entity.encode(TARGET))
        self.assertEqual(int(packed["graph_node_ent"][2]),
                         vocab.entity.encode(SITE_EE_REST))
        self.assertEqual(int(packed["graph_node_target"][1]), 1)
        self.assertEqual(int(packed["graph_node_target"].sum()), 1)

    def test_edge_endpoints_are_rows_not_entity_ids(self):
        """The distinction the probe used to get wrong: the counter's entity
        id and its row are different numbers, and reading one as the other
        addresses a different node."""
        packed, vocab = self._packed(), _vocab()
        real = packed["graph_edge_rel"] != vocab.relation.pad_id
        self.assertEqual({int(x) for x in packed["graph_edge_src"][real]}, {0})
        counter_row = int(np.where(
            packed["graph_node_ent"] == vocab.entity.encode(COUNTER))[0][0])
        self.assertNotEqual(counter_row, vocab.entity.encode(COUNTER))
        self.assertEqual({int(x) for x in packed["graph_edge_dst"][real]},
                         {1, 2})

    def test_padded_edge_slots_carry_the_pad_relation(self):
        """Which is how the probe tells a real fact from an empty slot."""
        packed = self._packed(e_max=12)
        self.assertEqual(int((packed["graph_edge_rel"] != 0).sum()), 7)

    def test_a_frame_with_no_target_refuses_to_pack(self):
        """Row 1 is where the sentinel resolves, so an unnamed target makes
        every phase unreadable -- the probe must not score such a frame."""
        graph = _graph()
        graph.meta["active_target_node_id"] = None
        with self.assertRaises(RuntimeError) as ctx:
            self._packed(graph)
        self.assertIn("row 1", str(ctx.exception))

    def test_the_sentinel_matches_no_real_entity(self):
        """Why the old id-against-slot comparison discarded every target
        fact: the schedule names the target by a sentinel that is not, and
        cannot be, any vocabulary id."""
        vocab = _vocab()
        self.assertLess(ACTIVE_TARGET_ENTITY_ID, 0)
        self.assertNotIn(ACTIVE_TARGET_ENTITY_ID,
                         set(vocab.entity.token_to_id.values()))


class ProbeCommandTest(unittest.TestCase):
    def test_raw_cameras_reach_the_real_node_builder(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from scenegraph.core.node_builder import build_nodes
        from tests.probes.probe_policy_potential import build_probe_frame

        cameras = ["fetch_head", "fetch_hand"]
        raw = {"sensor_data": {
            cam: {"rgb": np.zeros((2, 8, 8, 3), dtype=np.uint8),
                  "segmentation": np.stack([
                      np.full((8, 8, 1), index + 1),
                      np.full((8, 8, 1), index + 11)])}
            for index, cam in enumerate(cameras)}}
        state = SimpleNamespace(
            env_idx=1, tcp_pose_world=_pose(), seg_id_map={},
            robot_links=set(), robot_link_names=set(),
            ee_links=[], ee_link_names=set())

        def step(obs, frame, **kwargs):
            self.assertEqual(frame, 7)
            self.assertTrue(kwargs.pop("episode_boundary"))
            return build_nodes(obs, state, camera_order=cameras,
                               appearance=False, **kwargs)

        builder = SimpleNamespace(env_idx=1, step=Mock(side_effect=step))
        nodes, _, camera, _ = build_probe_frame(
            builder, raw, 7, cameras, episode_boundary=True)
        self.assertIn("ee", nodes)
        self.assertEqual(camera, "fetch_head")
        passed = builder.step.call_args.kwargs["seg_overrides"]
        self.assertEqual(list(passed), cameras)
        np.testing.assert_array_equal(passed["fetch_head"], np.full((8, 8), 11))
        np.testing.assert_array_equal(passed["fetch_hand"], np.full((8, 8), 12))

        # A genuinely missing camera must not silently become a one-camera run.
        del raw["sensor_data"]["fetch_hand"]
        builder.step.reset_mock()
        with self.assertRaises(KeyError):
            build_probe_frame(builder, raw, 8, cameras, episode_boundary=False)
        builder.step.assert_not_called()

    def test_requested_capacity_reaches_the_builder(self):
        from types import SimpleNamespace
        from tests.probes.probe_policy_potential import Report, build_config

        args = SimpleNamespace(
            thresholds=f"{CONFIGS}/thresholds.yaml", task="tidy_house",
            affordance=f"{CONFIGS}/affordances/tidy_house.json",
            whitelist_dir=WHITELIST_DIR, object_object_spatial=False,
            disable_object_object_relations=True,
            cameras=["fetch_head", "fetch_hand"],
            visibility_policy="projected_camera", n_max=32)
        cfg = build_config(args, Report())
        self.assertEqual(cfg["selection"]["n_max"], 32)
        self.assertFalse(cfg["object_object_spatial"])
        self.assertTrue(cfg["disable_object_object_relations"])
        args.disable_object_object_relations = False
        self.assertFalse(build_config(args, Report())["disable_object_object_relations"])

    def test_command_entry_point_runs_argument_parsing(self):
        # Importing main() is not enough: the server runs the file directly.
        script = pathlib.Path(__file__).parent / "probes" / "probe_policy_potential.py"
        result = subprocess.run([sys.executable, str(script), "--help"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--occupancy-out", result.stdout)
        missing = subprocess.run([sys.executable, str(script)],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(missing.returncode, 2)

    def test_full_episode_occupancy_round_trips_to_existing_auditor(self):
        from tests.probes.probe_policy_potential import write_occupancy
        from scenegraph.tools.audit_graph_capacity import main as audit, occupancy_report

        before, after = _graph(), _graph()
        after.nodes.append(Node(
            node_id=f"{TARGET}-second", node_type="object", name="second",
            pose_world=_pose(), attributes={"whitelist_key": TARGET}))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "episode.occupancy.json"
            write_occupancy(path, [before, after])
            frames = json.loads(path.read_text(encoding="utf-8"))
            report = occupancy_report(frames)
            self.assertEqual(report["n_frames"], 2)
            self.assertEqual(report["peak_nodes"], 5)
            self.assertEqual(report["peak_instances_per_key"], 2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit(["--whitelist-dir", WHITELIST_DIR,
                                        "--occupancy-json", str(path),
                                        "--n-max", "5", "--e-max", "7"]), 0)
                self.assertEqual(audit(["--whitelist-dir", WHITELIST_DIR,
                                        "--occupancy-json", str(path),
                                        "--n-max", "4"]), 1)


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch is not installed")


class ScoringTest(unittest.TestCase):
    """End to end through the real scorer and the frozen schedule."""

    @classmethod
    def setUpClass(cls):
        _torch()

    def _score(self, graph):
        from tests.probes.probe_policy_potential import score_frames

        vocab = _vocab()
        source = mshab_schedule_source(
            "tidy_house", "pick", CONFIGS, f"{CONFIGS}/schedules",
            WHITELIST_DIR)
        schedule = compile_from_source(source, vocab.entity)
        return score_frames([graph], schedule, vocab, n_max=6, e_max=24)

    def test_the_successful_frame_scores_one(self):
        potentials, valids = self._score(_graph())
        self.assertTrue(valids[0])
        self.assertAlmostEqual(potentials[0], 1.0, places=5)

    def test_the_target_fact_is_read_rather_than_discarded(self):
        """Drop the grasp and the score has to fall. Under the old
        id-against-sentinel comparison it was never read, so removing it
        changed nothing at all."""
        potentials, _ = self._score(_graph(drop=("grasp",)))
        self.assertLess(potentials[0], 1.0)

    def test_the_site_fact_is_read_too(self):
        potentials, _ = self._score(_graph(drop=("reached",)))
        self.assertLess(potentials[0], 1.0)


if __name__ == "__main__":
    unittest.main()
