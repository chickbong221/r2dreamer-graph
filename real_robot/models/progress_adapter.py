"""Progress potentials for the kitchen, from the repository's schedule machinery.

The progress-aware method needs a bounded potential ``Phi`` over graph states.
The repository already turns a phase schedule into one:
:func:`scenegraph.core.schedule.compile_schedule` resolves roles to entity ids,
stored pair orientation and mirrored labels, and refuses clauses the graph
cannot score; :class:`progress.TaskScheduleReplayPotential` reads observed
labels out of packed graphs with cumulative credit and current-frame gates.
Both are used unchanged.

The compiler checks scorability against mined simulator assets -- interaction
tokens, affordance components, calibrated bins. The kitchen has none of those:
its labels come from Gemini under the frozen specification. So the adapter
builds the equivalent view from the kitchen graph configuration itself: an
entity carries an interaction token, a component flag or a bin exactly when a
configured fact would need it. Every value in that view is a presence flag;
there are no numeric edges, because no code decides a bin.

The potential is kept apart from the environment reward everywhere: it is
computed from packed graphs at encoding time, stored in its own latent-cache
columns, and read only by the progress branch.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from scenegraph.core.schedule import CompiledSchedule, compile_schedule
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    FAMILY_MANIPULAND,
    FAMILY_RECEPTACLE,
    FAMILY_STRUCTURAL,
    OBJECT_OBJECT_SCOPE,
    ee_family_bin_key,
    spatial_bin_key,
)

from ..common import repo_path, stable_hash
from ..graphs.schema import GraphSpec

PRESENT = [1.0]


def schedule_asset_view(spec: GraphSpec) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """``(objects, members, bin_edges)`` in the shape the compiler reads, from the facts."""
    members: Dict[str, Dict[str, Any]] = {}
    objects: Dict[str, Dict[str, Any]] = {}
    families = {}
    for entity in spec.entities:
        if entity.type != "object":
            continue
        if entity.id in spec.targets:
            family = FAMILY_MANIPULAND
        elif entity.id == "table":
            family = FAMILY_STRUCTURAL
        else:
            family = FAMILY_RECEPTACLE
        families[entity.id] = family
        members[entity.key] = {"interaction_types": [], "family": family}
        objects[entity.key] = {}

    def token(entity_id: str, name: str) -> None:
        types = members[spec.entity(entity_id).key]["interaction_types"]
        if name not in types:
            types.append(name)

    def component(entity_id: str, name: str) -> None:
        objects[spec.entity(entity_id).key][name] = PRESENT

    bin_edges: Dict[str, Any] = {}
    for fact in spec.facts:
        ends = [e for e in (fact.src, fact.dst) if e != "ee"]
        if fact.relation in ("contact", "grasp", "support", "contain"):
            for entity_id in ends:
                token(entity_id, fact.relation)
        elif fact.relation == "grasp-compatibility":
            component(fact.dst, "grasp_components")
        elif fact.relation == "contact-compatibility":
            for entity_id in ends:
                component(entity_id, "contact_components")
        elif fact.relation in ("support-compatibility", "contain-compatibility"):
            # The kitchen's roles are fixed: the pot holds, the other object is held.
            holder = "pot" if "pot" in (fact.src, fact.dst) else fact.dst
            held = fact.src if holder == fact.dst else fact.dst
            host, guest = (("support_components", "bottom_components") if fact.relation == "support-compatibility"
                           else ("contain_components", "key_components"))
            component(holder, host)
            component(held, guest)
        elif fact.relation in ("planar-distance", "height-offset"):
            if fact.src == "ee":
                if fact.relation == "planar-distance":
                    bin_edges[spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance")] = PRESENT
                else:
                    bin_edges[ee_family_bin_key(families[fact.dst])] = PRESENT
            else:
                bin_edges[spatial_bin_key(OBJECT_OBJECT_SCOPE, fact.relation)] = PRESENT
    return objects, members, bin_edges


def compile_kitchen_schedule(spec: GraphSpec, vocab, schedule_path: str) -> CompiledSchedule:
    with open(repo_path(schedule_path), "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    objects, members, bin_edges = schedule_asset_view(spec)
    return compile_schedule(raw, objects, members, bin_edges, vocab.entity, sites={}, structural=set())


def schedule_identity(spec: GraphSpec, schedule_path: str) -> Dict[str, str]:
    with open(repo_path(schedule_path), "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {"schedule": stable_hash(raw), "graph": stable_hash(spec.identity())}


class KitchenProgress:
    """``Phi`` and its validity for packed frames, via ``TaskScheduleReplayPotential``."""

    def __init__(self, spec: GraphSpec, vocab, schedule_path: str, device="cpu"):
        import torch
        from progress import TaskScheduleReplayPotential

        self.torch = torch
        self.schedule = compile_kitchen_schedule(spec, vocab, schedule_path)
        self.scorer = TaskScheduleReplayPotential(self.schedule, len(vocab.absolute)).to(device)
        self.device = torch.device(device)

    def potential(self, packed: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """``packed`` holds the nine graph arrays stacked over frames."""
        from graph import compact_graph

        torch = self.torch
        tensors = {key: torch.as_tensor(np.asarray(value)).to(self.device) for key, value in packed.items()}
        compact = compact_graph(tensors)
        with torch.no_grad():
            phi, valid = self.scorer(compact.node_ent, compact.edge_rel, compact.edge_abs, compact.edge_src_local,
                                     compact.edge_dst_local, compact.edge_graph, compact.graph_count)
        return phi.float().cpu().numpy(), valid.bool().cpu().numpy()


class ProgressHead:
    """A bounded ``[0, 1]`` regressor on frozen latents, separate from every reward head.

    Fitted on every recorded transition whose observed-graph potential is
    valid, and measured on the diagnostic rows -- which are among those it was
    fitted on, so the measurement shows fit, not generalisation. Its weights
    and its batches come from their own seed, so fitting it cannot change the
    initialisation or sampling of the actor and critics.
    """

    def __init__(self, z_dim: int, hidden, device, seed: int = 0):
        import torch

        from .iql import MLP

        self.torch = torch
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = MLP(z_dim, 1, hidden)
        self.net = net.to(device)
        self.device = torch.device(device)

    def __call__(self, z):
        return self.torch.sigmoid(self.net(z).squeeze(-1))

    def mean_absolute_error(self, transitions) -> Dict[str, float]:
        torch = self.torch
        errors, counts = 0.0, 0.0
        with torch.no_grad():
            for batch in transitions.iterate(4096):
                mask = batch["phi_valid"]
                errors += float(((self(batch["z"]) - batch["phi"]).abs() * mask).sum())
                counts += float(mask.sum())
        return {"mae": errors / max(counts, 1.0), "valid_rows": counts}

    def fit(self, recorded, diagnostic, steps: int, batch_size: int, lr: float, delta: float, log=None
            ) -> Dict[str, Any]:
        """Huber regression onto observed-graph potentials, masked where they are invalid."""
        from ..data.selection import DIAGNOSTIC_NOTE

        torch = self.torch
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        for step in range(1, int(steps) + 1):
            batch = recorded.sample(batch_size, generator=generator)
            mask = batch["phi_valid"]
            error = torch.nn.functional.huber_loss(self(batch["z"]), batch["phi"], reduction="none", delta=delta)
            loss = (error * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if log is not None and step % 1000 == 0:
                log(step, {"progress_head/loss": float(loss)})
        self.net.requires_grad_(False)
        return {"steps": int(steps), "recorded": self.mean_absolute_error(recorded),
                "diagnostic": self.mean_absolute_error(diagnostic) if diagnostic.size else None,
                "note": DIAGNOSTIC_NOTE}

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state):
        self.net.load_state_dict(state)
        self.net.requires_grad_(False)
