"""Progress for the SO-101 tasks: schedules compiled against the recorded facts.

Frames are packed by the repository's own ``pack_graph`` from the real-robot
graph config, so the rows, edge orientation and label ids are the ones the
converted datasets carry.
"""

from __future__ import annotations

import unittest

import numpy as np

from sim_vla.tests.common import require_torch

DEFAULTS = {"contact": "not-holds", "grasp": "not-holds", "support": "not-holds",
            "contain": "not-holds", "planar-distance": "far", "height-offset": "level",
            "grasp-compatibility": "poor-match", "contact-compatibility": "poor-match",
            "support-compatibility": "poor-match", "contain-compatibility": "poor-match"}


def setting(task: str):
    """``(cfg, graph metadata, spec, vocab)`` for one ``--data real`` task."""
    from real_robot.graphs.schema import load_graph_config
    from real_robot.graphs.vocabulary import build_vocab, vocab_tables
    from sim_vla.config import load_config
    from sim_vla.data.convert_real import graph_metadata

    cfg = load_config(task, "graph_progress", data="real")
    graph_task = cfg["task"]["env_id"].split("/", 1)[1]
    config = load_graph_config()
    vocab = build_vocab(config)
    manifest = {"graph": config.identity(), "vocab": vocab_tables(vocab)}
    return cfg, graph_metadata(manifest, graph_task), config.spec(graph_task), vocab


def frame(spec, vocab, labels, target, drop=()):
    """One packed frame; ``labels`` maps stored ``(src, dst, relation)`` to a label."""
    from scenegraph.adapters.graph_pack import pack_graph
    from scenegraph.core.schema import Edge, Graph, Node

    graph = Graph(frame=0, env_id="test", camera="top+wrist")
    for entity in spec.entities:
        graph.nodes.append(Node(
            node_id=entity.node_id, node_type=entity.type, name=entity.name,
            bbox=np.zeros((len(spec.cameras), 4), np.float32),
            attributes={} if entity.type == "ee" else {"whitelist_key": entity.key}))
    for fact in spec.facts:
        if fact.key in drop:
            continue
        graph.edges.append(Edge(
            src=spec.entity(fact.src).node_id, dst=spec.entity(fact.dst).node_id,
            relation=fact.relation, label=labels.get(fact.key, DEFAULTS[fact.relation])))
    graph.meta["active_target_node_id"] = spec.entity(target).node_id
    return pack_graph(graph, vocab, n_max=spec.n_max, e_max=spec.e_max,
                      n_cams=len(spec.cameras), use_target_flag=True)


def score(potential, frames):
    """``(phi, valid)`` for a list of packed frames, as one ``(1, T)`` batch."""
    import torch

    from scenegraph.adapters.graph_pack import GRAPH_KEYS

    batch = {key: torch.as_tensor(np.stack([f[key] for f in frames])[None]) for key in GRAPH_KEYS}
    phi, valid = potential.targets(batch)
    return phi[0].tolist(), valid[0].tolist()


def build(task):
    from sim_vla.training.progress import build_potential

    cfg, graph_meta, spec, vocab = setting(task)
    metadata = {"env_id": cfg["task"]["env_id"], "graph": graph_meta}
    return build_potential(cfg, metadata), spec, vocab


class TestStackCube(unittest.TestCase):
    def setUp(self):
        require_torch()
        self.potential, self.spec, self.vocab = build("stackcube")

    def phi(self, labels, drop=()):
        return score(self.potential, [frame(self.spec, self.vocab, labels, "blue_cube", drop)])

    def test_progress_rises_through_the_phases_and_ends_at_one(self):
        grasped = {("ee", "blue_cube", "planar-distance"): "very-near",
                   ("ee", "blue_cube", "grasp-compatibility"): "match",
                   ("ee", "blue_cube", "contact"): "holds",
                   ("ee", "blue_cube", "grasp"): "holds",
                   ("blue_cube", "red_cube", "planar-distance"): "near"}
        stacked = {("ee", "blue_cube", "planar-distance"): "near",
                   ("blue_cube", "red_cube", "contact"): "holds",
                   ("blue_cube", "red_cube", "support"): "dst-holds",
                   ("blue_cube", "red_cube", "planar-distance"): "very-near",
                   ("blue_cube", "red_cube", "support-compatibility"): "match"}
        (start,), _ = self.phi({})
        (holding,), _ = self.phi(grasped)
        (done,), valid = self.phi(stacked)
        self.assertEqual(valid, [True])
        self.assertLess(start, 0.2)
        self.assertGreater(holding, 0.5)
        self.assertLess(holding, 1.0)
        self.assertAlmostEqual(done, 1.0, places=5)

    def test_the_red_cube_resting_on_the_blue_is_not_success(self):
        (flipped,), _ = self.phi({("blue_cube", "red_cube", "support"): "src-holds",
                                  ("blue_cube", "red_cube", "contact"): "holds"})
        self.assertLess(flipped, 0.5)

    def test_a_frame_missing_a_scored_fact_is_unscorable(self):
        _, valid = self.phi({}, drop={("ee", "blue_cube", "grasp")})
        self.assertEqual(valid, [False])


class TestCubesInCup(unittest.TestCase):
    def setUp(self):
        require_torch()
        self.potential, self.spec, self.vocab = build("cubes_in_cup")

    def placed(self, cube):
        pair = ("cup", "red_cube") if cube == "red_cube" else ("blue_cube", "cup")
        holder = "src-holds" if pair[0] == "cup" else "dst-holds"
        return {(*pair, "contain"): holder, (*pair, "contact"): "holds",
                (*pair, "planar-distance"): "very-near", (*pair, "contain-compatibility"): "match"}

    def phi(self, labels, target):
        (value,), (valid,) = score(self.potential, [frame(self.spec, self.vocab, labels, target)])
        self.assertTrue(valid)
        return value

    def test_each_cube_in_the_cup_is_half_the_task_in_either_order(self):
        nothing = self.phi({}, "red_cube")
        red_first = self.phi(self.placed("red_cube"), "blue_cube")
        blue_first = self.phi(self.placed("blue_cube"), "red_cube")
        both = self.phi(self.placed("red_cube") | self.placed("blue_cube"), "blue_cube")
        self.assertLess(nothing, 0.5)
        self.assertGreaterEqual(red_first, 0.5)
        self.assertAlmostEqual(red_first, blue_first, places=5)
        self.assertLess(red_first, 1.0)
        self.assertAlmostEqual(both, 1.0, places=5)


class TestCompile(unittest.TestCase):
    def setUp(self):
        require_torch()

    def test_both_tasks_compile_against_what_they_record(self):
        for task, phases in (("stackcube", 4), ("cubes_in_cup", 8)):
            potential, _, _ = build(task)
            self.assertEqual(potential.phases, phases, task)

    def test_a_schedule_naming_an_unrecorded_fact_is_refused(self):
        from scenegraph.core.schedule import ScheduleError
        from sim_vla.training.progress import RecordedSchedulePotential, recorded_schedules

        stack_cfg = setting("stackcube")[0]
        cup_meta = setting("cubes_in_cup")[1]
        with self.assertRaises(ScheduleError):
            RecordedSchedulePotential(recorded_schedules(stack_cfg), cup_meta, 19)

    def test_preflight_names_what_the_dataset_lacks(self):
        from sim_vla.training.progress import preflight

        cfg, graph_meta, _, _ = setting("stackcube")
        preflight(cfg, {"graph": graph_meta})
        with self.assertRaises(SystemExit) as caught:
            preflight(cfg, {"graph": {key: value for key, value in graph_meta.items() if key != "facts"}})
        self.assertIn("prepare_real", str(caught.exception))
        half = cfg | {"task": dict(cfg["task"], progress_schedules=[
            {"schedule": "stackcube.json", "weight": 0.5}])}
        with self.assertRaises(SystemExit) as caught:
            preflight(half, {"graph": graph_meta})
        self.assertIn("sum to 0.5", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
