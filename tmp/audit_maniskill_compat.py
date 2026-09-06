"""Read-only compatibility probe against the recorded post-ManiSkill revision."""
import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scenegraph.configs.loader import load_config
from scenegraph.core.graph_builder import GraphBuilder
from scenegraph.core import schedule, relation_rules
from scenegraph.adapters import graph_pack, site_providers
from scenegraph.adapters.graph_vocab import build_graph_vocab
from scenegraph.core.schema import Graph, Node, Edge
import progress

BASE = "8fd3819"
def old_module(path, name):
    source = subprocess.check_output(
        ["git", "-c", "safe.directory=E:/Code/r2dreamer-graph", "show", f"{BASE}:{path}"],
        cwd=ROOT, text=True, encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module

old_schedule = old_module("scenegraph/core/schedule.py", "scenegraph.core._audit_schedule")
old_pack = old_module("scenegraph/adapters/graph_pack.py", "scenegraph.adapters._audit_pack")
old_rules = old_module("scenegraph/core/relation_rules.py", "scenegraph.core._audit_rules")
old_progress = old_module("progress.py", "_audit_progress")
old_provider = old_module("scenegraph/adapters/site_providers.py", "scenegraph.adapters._audit_provider")
torch.manual_seed(71)
rng = np.random.default_rng(71)

for task in ("PickCube-v1", "PegInsertionSide-v1"):
    cfg = load_config(task_group=task)
    cfg.update(object_object_spatial=True, disable_object_object_relations=False)
    cfg["selection"]["n_max"] = 8
    builder = GraphBuilder(None, cfg, use_target_flag=False)
    builder._bind_task_whitelist()
    vocab = build_graph_vocab(cfg["whitelist_dir"])
    source = schedule.maniskill_schedule_source(task, str(ROOT / "scenegraph/configs"),
                str(ROOT / "scenegraph/configs/schedules"), cfg["whitelist_dir"])
    new = schedule.compile_from_source(source, vocab.entity)
    old = old_schedule.compile_from_files(task, str(ROOT / "scenegraph/configs/schedules"),
                                         str(ROOT / "scenegraph/configs"), vocab.entity)
    assert dataclasses.asdict(new) == dataclasses.asdict(old)
    assert relation_rules.required_bin_keys(cfg) == old_rules.required_bin_keys(cfg)
    scorers = [mod.TaskScheduleReplayPotential(s, len(vocab.absolute))
               for mod, s in ((progress, new), (old_progress, old))]
    members = json.loads(Path(source.union_whitelist_path).read_text())["members"]
    for trial in range(100):
        keys = list(rng.permutation(list(members)))
        nodes = [Node("ee", "ee", "ee", pose_world=[0,0,0,1,0,0,0], index=0)]
        for i, key in enumerate(keys, 1):
            nodes.append(Node(key, "object", key, index=i,
                              pose_world=[*rng.normal(size=3), 1, 0, 0, 0],
                              bbox=rng.random((1,4)),
                              attributes={"whitelist_key": key}))
        graph = Graph(trial, task, "camera", nodes=nodes)
        ids = {vocab.entity.encode(n.attributes["whitelist_key"]): n.node_id
               for n in nodes[1:]}
        ids[vocab.entity.ee_id] = "ee"
        relation_names = {v:k for k,v in vocab.relation.token_to_id.items()}
        absolute_names = {v:k for k,v in vocab.absolute.token_to_id.items()}
        for rel, src, dst in new.slots:
            labels = relation_rules.ABS_LABELS[relation_names[rel]]
            graph.edges.append(Edge(ids[src], ids[dst], relation_names[rel], rng.choice(labels)))
        # Compare packing both fully populated and partially observed frames.
        if trial % 3 == 0:
            graph.edges = graph.edges[:-1]
        packed = [mod.pack_graph(copy.deepcopy(graph), vocab, n_max=8, e_max=168,
                                 n_cams=1, use_target_flag=False)
                  for mod in (graph_pack, old_pack)]
        for key in packed[0]:
            np.testing.assert_array_equal(packed[0][key], packed[1][key])
        ent = torch.zeros((1,8), dtype=torch.long)
        rows = {}
        for row, node in enumerate(nodes):
            identity = vocab.entity.ee_id if row == 0 else vocab.entity.encode(node.node_id)
            ent[0,row] = identity
            rows[node.node_id] = row
        args = (ent,
                torch.tensor([vocab.relation.encode(e.relation) for e in graph.edges]),
                torch.tensor([vocab.absolute.encode(e.label) for e in graph.edges]),
                torch.tensor([rows[e.src] for e in graph.edges]),
                torch.tensor([rows[e.dst] for e in graph.edges]),
                torch.zeros(len(graph.edges), dtype=torch.long), 1)
        results = [scorer(*args) for scorer in scorers]
        for a,b in zip(*results):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    print(f"{task}: runtime asset binding PASS; {len(new.phases)} phases, {len(new.slots)} slots; "
          "old/new compiled schedule, required bins, packing and potential identical in 100 synthetic frames")

# Exercise the changed tensor-row readers with a heterogeneous vector batch.
for count in (1, 10, 200):
    poses = torch.randn(count,7)
    values = torch.rand(count)
    for idx in range(count):
        np.testing.assert_array_equal(site_providers._row(poses,idx), old_provider._row(poses,idx))
        assert site_providers._scalar(values,idx) == old_provider._scalar(values,idx)
print("Tensor site readers: old/new identical for every row of batches 1, 10 and 200")

sys.path.insert(0, str(ROOT / "tests"))
import test_edge_emission_bounds as emission_fixture
import test_sites as site_fixture
for n_objects in range(1, 8):
    results = []
    for module in (relation_rules, old_rules):
        emission_fixture.rr = module
        results.append(emission_fixture._emit(n_objects, emission_fixture._ROLES))
    assert results[0] == results[1]
emission_fixture.rr = relation_rules
print("Physical/compatibility/spatial emission structure: old/new identical for 1-7 objects (physics and compatibility math stubbed)")
for distance in np.linspace(-0.5, 0.5, 101):
    fixture = site_fixture.GoalEdgeEmissionTest()
    graph = fixture._scene(cube_x=distance)
    cfg = fixture._cfg(site_fixture._spec(site_fixture._decl(
        key="actor:goal_site", subject="actor:cube")))
    assert relation_rules.goal_edges(graph, None, cfg) == old_rules.goal_edges(graph, None, cfg)
    spec, decl = site_fixture._hole_spec(head=(distance,0.3,0.1))
    spec = dataclasses.replace(spec, axis_world=np.array([1.0, 0.0, 0.0]))
    graph = site_fixture._graph(site_fixture._obj(site_fixture.PEG, (distance,0.3,0.1)),
                               site_fixture._obj(site_fixture.HOLE, (0,0.3,0.1), dynamic=False))
    cfg = site_fixture._ladder_cfg(spec, decl)
    assert relation_rules.object_object_spatial_edges(graph, None, cfg) == old_rules.object_object_spatial_edges(graph, None, cfg)
    assert relation_rules.goal_edges(graph, None, cfg) == old_rules.goal_edges(graph, None, cfg)
print("PickCube goal and peg-head/hole-mouth ladder: old/new edges identical at 101 positions each")
