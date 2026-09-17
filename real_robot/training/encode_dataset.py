"""Freeze the selected world model and cache the latent transitions policies learn from.

    python -m real_robot.training.encode_dataset --world-model wm_base --name wm_base_final --progress
    python -m real_robot.training.encode_dataset --world-model wm_base --checkpoint step_00020000 --name wm_base_20k

One cache holds every training episode. Each episode is run chronologically
from its first frame -- no burn-in approximation -- and each frame's posterior
feature is stored with the action taken there, the reward for arriving at the
next frame, that frame's feature and the termination flag. A recorded action
with no next observation is stored but marked as no transition. Diagnostic
episodes are marked rows of the same cache, not a second one.

The latent convention is fixed: the posterior mode of ``z``, which makes the
cache a deterministic function of the observations and the weights. The robot
wrapper reads the same convention from this cache's identity.

``identity.json`` records the checkpoint file's SHA-256 and a digest of its
weights, the dataset and annotation identity, the reward definition and fitted
scales, the action mapping and normaliser, the latent convention and the
selection. A cache directory is never rewritten under a different identity:
encoding other weights -- another checkpoint, or the same run trained further
-- needs a new ``--name``, and imagined transitions and policies built on the
old cache keep referring to it and refuse the new one.

With ``--progress`` the observed-graph potential of the progress schedule is
cached alongside, in its own columns. It is never mixed into the reward.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..common import add_config_arguments, file_sha256, load_configs, read_json, repo_path, utc_now, write_json
from ..data.episode_dataset import BuiltEpisodeStore
from ..data.latent_dataset import LATENT_FORMAT, TRANSITIONS_FILE, compatibility
from ..data.selection import DIAGNOSTIC_NOTE
from ..data.sequence_dataset import full_episode
from ..models.world_model import load_world_model, to_torch, weights_digest
from .checkpoints import existing_checkpoints

GRAPH_COLUMNS = ("graph_node_ent", "graph_node_bbox", "graph_node_centroid", "graph_node_target",
                 "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs", "graph_edge_temp")
ANNOTATION_FIELDS = ("annotation_mode", "gemini_model", "gemini_backend", "prompt_version", "video_fps", "bins",
                     "graph", "temporal_window")
LATENT_CONVENTION = "posterior mode of z, each episode run from its first frame"


def resolve_checkpoint(dataset_cfg, reference: str, kind: str = "final") -> str:
    """A checkpoint file, or ``<kind>.pt`` of a run under ``runs/world_model``. Nothing else is substituted."""
    path = repo_path(reference)
    if os.path.isfile(path):
        return path
    run = os.path.join(repo_path(dataset_cfg["paths"]["runs"]), "world_model", reference)
    candidate = os.path.join(run, f"{kind}.pt")
    if os.path.isfile(candidate):
        return candidate
    raise SystemExit(f"no {kind}.pt for world model {reference!r} in {run}; "
                     f"checkpoints there: {existing_checkpoints(run) or 'none'}")


def latent_identity(checkpoint: str, payload, manifest, feat_dim: int, sha256: str, weights: str,
                    progress: Optional[Dict[str, Any]], coverage: Dict[str, Any]) -> Dict[str, Any]:
    dataset = manifest.identity
    return {
        "format": LATENT_FORMAT,
        "created": utc_now(),
        "world_model": {"checkpoint": os.path.relpath(checkpoint, repo_path("")).replace(os.sep, "/"),
                        "kind": payload.get("kind"), "step": int(payload["step"]),
                        "sha256": sha256, "weights": weights, "identity": payload["identity"]},
        "dataset": manifest.dataset_key(),
        "dataset_root": manifest.root,
        "annotation": {key: dataset.get(key) for key in ANNOTATION_FIELDS},
        "reward": dataset["reward"],
        "action": {"mapping": dataset["action_mapping"], "transform": dataset["action_transform"]},
        "latent": {"inference": "mode", "convention": LATENT_CONVENTION, "feat_dim": int(feat_dim),
                   "action_dim": int(manifest.action_dim)},
        "selection": {**dataset["selection"], "diagnostic_packed": manifest.diagnostic_episodes()},
        "coverage": {"episodes": coverage["built"], "training": coverage["training"],
                     "complete": coverage["complete"]},
        "progress": progress,
        "note": DIAGNOSTIC_NOTE,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Cache latent transitions from a frozen world model.")
    parser.add_argument("--world-model", required=True, help="run name under runs/world_model, or a checkpoint path")
    parser.add_argument("--checkpoint", default="final", help="final | best_diagnostic | step_XXXXXXXX | latest")
    parser.add_argument("--name", required=True, help="latent cache name under paths.latents")
    parser.add_argument("--progress", action="store_true", help="also cache the progress-schedule potential")
    parser.add_argument("--schedule", default="real_robot/configs/kitchen_schedule.json")
    parser.add_argument("--allow-partial-dataset", action="store_true",
                        help="encode although some training episodes are not packed (smoke tests only)")
    parser.add_argument("--force", action="store_true", help="re-encode a cache with the same identity")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "graph"], args.overrides)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = resolve_checkpoint(configs["dataset"], args.world_model, args.checkpoint)
    sha256 = file_sha256(checkpoint)
    model, payload, manifest = load_world_model(checkpoint, device)
    weights = weights_digest(payload["state"]["model"])
    coverage = manifest.require_complete(args.allow_partial_dataset, "encode_dataset")
    store = BuiltEpisodeStore(manifest.root)
    episodes = manifest.training_episodes()
    diagnostic = set(manifest.diagnostic_episodes())

    progress = None
    progress_identity: Optional[Dict[str, Any]] = None
    if args.progress:
        from ..graphs.schema import GraphSpec
        from ..graphs.vocabulary import build_vocab
        from ..models.progress_adapter import KitchenProgress, schedule_identity

        spec = GraphSpec.from_config(configs["graph"])
        progress = KitchenProgress(spec, build_vocab(spec), args.schedule, device)
        progress_identity = schedule_identity(spec, args.schedule)

    identity = latent_identity(checkpoint, payload, manifest, model.feat_size, sha256, weights, progress_identity,
                               coverage)
    root = os.path.join(repo_path(configs["dataset"]["paths"]["latents"]), args.name)
    identity_path = os.path.join(root, "identity.json")
    if os.path.isfile(identity_path):
        existing = read_json(identity_path)
        try:
            same = (compatibility(existing) == compatibility(identity)
                    and existing.get("progress") == identity["progress"])
        except KeyError:
            same = False
        if not same:
            raise SystemExit(f"{root} holds a latent cache built under a different contract (another checkpoint, "
                             "dataset or setting). Choose a new --name; artifacts built on the old cache keep "
                             "referring to it.")
        if not args.force:
            print(f"[encode] {root} already holds this cache; pass --force to re-encode it")
            return
        os.remove(identity_path)
    os.makedirs(root, exist_ok=True)

    columns: Dict[str, List[np.ndarray]] = {}
    offset = 0
    for episode in episodes:
        arrays = store.load(episode, images=False)
        out = model.encode_sequence(to_torch(full_episode(store, episode), device), sample=False)
        length = int(out["feat"].shape[1])
        transition = arrays["transition_valid"][:length].astype(bool)
        rows = np.arange(length)
        nxt = np.full(length, -1, dtype=np.int64)
        nxt[transition] = offset + rows[transition] + 1

        def append(key: str, value: np.ndarray) -> None:
            columns.setdefault(key, []).append(value)

        append("feat", out["feat"][0].to(torch.float16).cpu().numpy())
        append("action", arrays["action"][:length].astype(np.float32))
        append("reward", np.nan_to_num(arrays["reward"][:length], nan=0.0).astype(np.float32))
        append("done", arrays["done"][:length].astype(bool))
        append("transition_valid", transition)
        append("obs_valid", arrays["obs_valid"][:length].astype(bool))
        append("next_index", nxt)
        append("episode", np.full(length, int(episode), dtype=np.int64))
        append("frame", rows.astype(np.int64))
        append("diagnostic", np.full(length, int(episode) in diagnostic, dtype=bool))
        if progress is not None:
            phi, valid = progress.potential({key: arrays[key][:length] for key in GRAPH_COLUMNS})
            valid = valid & arrays["graph_valid"][:length].astype(bool)
            append("progress_phi", np.where(valid, phi, np.nan).astype(np.float32))
            append("progress_valid", valid)
        offset += length
        print(f"[encode] episode {episode}{' (diagnostic)' if episode in diagnostic else ''}: {length} frames",
              flush=True)

    if weights_digest(model.state_dict()) != weights:
        raise RuntimeError("the world model's weights changed while encoding; the cache is not written")
    data = {key: np.concatenate(values) for key, values in columns.items()}
    partial = os.path.join(root, "transitions.partial.npz")
    np.savez(partial, **data)
    os.replace(partial, os.path.join(root, TRANSITIONS_FILE))
    identity["counts"] = {"frames": int(data["feat"].shape[0]),
                          "transitions": int(data["transition_valid"].sum()),
                          "diagnostic_transitions": int((data["transition_valid"] & data["diagnostic"]).sum())}
    write_json(identity_path, identity)
    print(f"[encode] {identity['counts']['transitions']} transitions over {len(episodes)} episodes "
          f"({identity['counts']['diagnostic_transitions']} in diagnostic episodes, also trained on) -> {root}")


if __name__ == "__main__":
    main()
