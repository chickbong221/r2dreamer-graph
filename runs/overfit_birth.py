"""Stage B gate: can the slot world model learn one deterministic birth?

A single synthetic episode, repeated:

    step 0..k-1   end effector only
    step k        object B appears, far from the end effector
    step k+1..    B persists and its distance label improves every step

Nothing about this is stochastic and the action is constant, so the birth step is
a deterministic function of the recurrent state. If the presence head, the birth
content head, and the proposal matching cannot fit this, they will not fit
anything, and the real-data numbers will be unreadable because a base-rate
predictor scores well on them.

Deliberately self-contained: no hydra, no tensordict, no simulator, so it runs in
any environment that has torch. The observation embedding is held constant, so
neither the posterior nor the prior can read the timestep out of the pixels --
everything has to come through h.

    python runs/overfit_birth.py --steps 600
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from graph import (  # noqa: E402
    SCHEMA_SIMPLE_SLOT,
    GraphEncoder,
    SlotGraphDecoder,
    compact_graph,
)
from progress import PICK_STAGES, ProgressScorer  # noqa: E402
from rssm import RSSM, SLOT_META_TARGET  # noqa: E402


UID_EE, UID_OBJECT = 1, 2
ENT_EE, ENT_OBJECT = 1, 2
REL_PLANAR = 5
# very-far, far, medium, near, very-near
DISTANCE = [7, 6, 5, 4, 3]
TEMP_DECREASE = 2


def graph_config(n_max, slot_dim, embed=16):
    return SimpleNamespace(
        simple=True, state_mode="slots", slot_births=True,
        n_max=n_max, slot_dim=slot_dim, slot_heads=4, slot_mixer_layers=1,
        units=slot_dim, simple_units=slot_dim, semantic_dim=slot_dim,
        layers=1, n_cams=2, app_dim=8, entity_vocab=14, n_rel=11, n_abs=19,
        n_temp=6, embed=embed, app=4, bbox=4, bbox_beta=0.1,
        uid_vocab=32, uid_embed=8, reverse_edges=True, act="SiLU",
    )


def rssm_config(deter, hidden, device):
    return SimpleNamespace(
        stoch=8, hybrid_stoch=8, deter=deter, hidden=hidden, discrete=8,
        img_layers=2, obs_layers=1, dyn_layers=1, blocks=8, act="SiLU",
        norm=True, unimix_ratio=0.01, initial="learned", device=device,
        sem_stoch=2, sem_discrete=8, sem_layers=1,
    )


def sequence(batch, time, birth, n_max, e_max=8, device="cpu"):
    """The synthetic episode, as the relation-only observation contract."""
    shape = (batch, time)
    graph = {
        key: torch.zeros(*shape, n_max, dtype=torch.uint8, device=device)
        for key in ("graph_node_ent", "graph_node_uid", "graph_node_target")
    }
    for key in ("src", "dst", "rel", "abs", "temp"):
        graph[f"graph_edge_{key}"] = torch.zeros(
            *shape, e_max, dtype=torch.uint8, device=device
        )
    # The end effector is present from the first frame.
    graph["graph_node_ent"][..., 0] = ENT_EE
    graph["graph_node_uid"][..., 0] = UID_EE
    for t in range(birth, time):
        graph["graph_node_ent"][:, t, 1] = ENT_OBJECT
        graph["graph_node_uid"][:, t, 1] = UID_OBJECT
        graph["graph_node_target"][:, t, 1] = 1
        graph["graph_edge_src"][:, t, 0] = 0
        graph["graph_edge_dst"][:, t, 0] = 1
        graph["graph_edge_rel"][:, t, 0] = REL_PLANAR
        step = min(t - birth, len(DISTANCE) - 1)
        graph["graph_edge_abs"][:, t, 0] = DISTANCE[step]
        graph["graph_edge_temp"][:, t, 0] = TEMP_DECREASE if step else 0
    is_first = torch.zeros(*shape, dtype=torch.bool, device=device)
    is_first[:, 0] = True
    return graph, is_first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--time", type=int, default=16)
    parser.add_argument("--birth", type=int, default=6)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=64)
    parser.add_argument("--deter", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    gcfg = graph_config(args.n_max, args.slot_dim)
    encoder = GraphEncoder(gcfg).to(device)
    decoder = SlotGraphDecoder(gcfg).to(device)
    model = RSSM(
        rssm_config(args.deter, args.hidden, args.device),
        embed_size=8,
        act_dim=2,
        semantic=True,
        graph_simple=True,
        graph_slots=True,
        graph_config=gcfg,
    ).to(device)
    scorer = ProgressScorer(PICK_STAGES, gcfg.n_abs).to(device)

    parameters = (
        list(encoder.parameters()) + list(decoder.parameters()) + list(model.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    graph, is_first = sequence(
        args.batch, args.time, args.birth, args.n_max, device=args.device
    )
    # Constant, uninformative: the birth has to be predicted from h, not read
    # out of the observation embedding.
    embed = torch.zeros(args.batch, args.time, 8, device=device)
    action = torch.zeros(args.batch, args.time, 2, device=device)
    step_valid = torch.ones(args.batch, args.time, dtype=torch.bool, device=device)

    print(
        f"birth at step {args.birth} of {args.time}; "
        f"{args.n_max} slots of {args.slot_dim}"
    )
    for iteration in range(1, args.steps + 1):
        slots = encoder(graph).slots
        observed = model.observe(
            embed, action, model.initial(args.batch), is_first, slot_obs=slots
        )
        alive = observed["slot_alive"]
        usable = ~observed["reset"][..., None] & ~observed["replaced"]
        persistent = observed["matched"] & usable
        born = observed["born"] & usable
        inactive = ~model.slot_mask(alive) & usable

        losses = {
            "slotdyn": model.slot_dynamics_loss(
                observed["prior_slot"], observed["sem"], persistent | born
            ),
            "slotalive": model.slot_alive_loss(
                observed["prior_alive_logit"], alive, persistent, born, inactive
            ),
        }
        graph_losses, metrics = decoder(
            observed["sem"],
            observed["prior_slot"],
            compact_graph(graph, SCHEMA_SIMPLE_SLOT),
            observed["dest"],
            alive,
            observed["slot_meta"][..., SLOT_META_TARGET],
            step_valid,
            scorer.relations,
        )
        losses.update(graph_losses)
        dyn, rep = model.kl_loss(observed["logit"], observed["prior_logit"], 1.0)
        losses["dyn"], losses["rep"] = dyn.mean(), rep.mean()

        total = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()

        if iteration % args.every and iteration != args.steps:
            continue
        with torch.no_grad():
            probability = torch.sigmoid(observed["prior_alive_logit"].float())
            birth_prob = probability[born].mean() if born.any() else probability.sum() * 0
            dead_prob = probability[inactive].mean()
            cosine = F.cosine_similarity(
                observed["prior_slot"].float(), observed["sem"].float(), dim=-1, eps=1e-6
            )
            birth_cos = cosine[born].mean() if born.any() else cosine.sum() * 0
            print(
                f"[{iteration:>5}] loss {float(total):7.3f} | "
                f"birth p {float(birth_prob):.3f} dead p {float(dead_prob):.3f} | "
                f"birth cos {float(birth_cos):+.3f} | "
                f"progress acc {float(metrics['prior_progress_acc']):.3f} | "
                f"target acc {float(metrics['prior_target_acc']):.3f}"
            )

    # Does an imagined rollout starting before the birth actually create a slot?
    with torch.no_grad():
        slots = encoder(graph).slots
        observed = model.observe(
            embed, action, model.initial(args.batch), is_first, slot_obs=slots
        )
        start = args.birth - 2
        rollout = model.imagine_with_action(
            observed["stoch"][:, start],
            observed["deter"][:, start],
            action[:, start:],
            observed["sem"][:, start],
            observed["slot_meta"][:, start],
            observed["slot_alive"][:, start],
        )
        occupancy = rollout["slot_alive"].gt(0.5).float().sum(-1).mean(0)
    print("\nimagined occupancy from step", start)
    print("  " + "  ".join(f"{float(v):.2f}" for v in occupancy))
    print(
        f"  started at {float(observed['slot_alive'][:, start].gt(0.5).sum(-1).float().mean()):.2f}"
        f", birth is {args.birth - start} imagined steps in"
    )

    print(
        "\nGate: birth p > 0.9, dead p < 0.1, birth cos > 0.9, progress acc ~ 1.0,\n"
        "and imagined occupancy rising at the right step. Anything less and the\n"
        "real-data presence numbers are not interpretable."
    )


if __name__ == "__main__":
    main()
