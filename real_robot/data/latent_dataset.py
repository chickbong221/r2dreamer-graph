"""Recorded and imagined latent transitions for the policy stages.

A latent cache holds every training episode, encoded once by one frozen
world-model checkpoint, and says in ``identity.json`` exactly what produced it:
the checkpoint file's SHA-256 and a digest of its weights, the dataset and
annotation identity, the reward definition and fitted scales, the action
mapping and normaliser, the latent-inference convention and the episode
selection. The diagnostic episodes are rows of the same cache, marked in its
``diagnostic`` column; they are not a second cache.

Imagined transitions carry the cache's compatibility record. Every reader
compares it field by field before loading arrays, so features from other
weights -- even with the same architecture and configuration -- are never
mixed with these, and a changed world model requires a new cache and new
imagined transitions.

Transition ``i`` is ``(z_i, a_i, r_i, z_{next_i}, d_i)``: ``r_i`` is the reward
for executing ``a_i`` and arriving at the next observation, ``d_i`` true task
termination on that arrival. A recorded action with no recorded next
observation is not a transition and is never sampled.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..common import read_json, repo_path, require_identity, stable_hash

LATENT_FORMAT = "real_robot/latent-cache-v2"
ROLLOUT_FORMAT = "real_robot/imagined-transitions-v2"
EARLIER_FORMATS = ("real_robot/latent-cache-v1", "real_robot/synthetic-rollouts-v1")
TRANSITIONS_FILE = "transitions.npz"
# What a consumer of recorded latents has to agree on with the cache.
COMPATIBILITY_FIELDS = ("world_model", "dataset", "annotation", "reward", "action", "latent", "selection")


def load_latent_identity(root: str) -> Dict[str, Any]:
    path = os.path.join(repo_path(root), "identity.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no latent cache identity at {path}; run encode_dataset first")
    data = read_json(path)
    if data.get("format") in EARLIER_FORMATS:
        raise ValueError(f"{path} was written by an earlier version of this package; re-encode under a new --name")
    if data.get("format") != LATENT_FORMAT:
        raise ValueError(f"{path}: not a latent cache")
    return data


def compatibility(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a cache identity that recorded and imagined transitions must share."""
    world_model = identity["world_model"]
    return {
        "world_model": {"sha256": world_model["sha256"], "weights": world_model["weights"],
                        "step": int(world_model["step"])},
        **{field: identity[field] for field in COMPATIBILITY_FIELDS if field != "world_model"},
    }


def compatibility_hash(identity: Mapping[str, Any]) -> str:
    return stable_hash(compatibility(identity))


def resolve_latents(dataset_cfg: Mapping[str, Any], reference: str) -> str:
    path = repo_path(reference)
    if os.path.isdir(path):
        return path
    return os.path.join(repo_path(dataset_cfg["paths"]["latents"]), reference)


class LatentTransitions:
    """A latent cache in host memory: every valid transition, with the diagnostic rows marked."""

    def __init__(self, root: str):
        self.root = repo_path(root)
        self.identity = load_latent_identity(self.root)
        path = os.path.join(self.root, TRANSITIONS_FILE)
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}
        valid = self.arrays["transition_valid"].astype(bool) & (self.arrays["next_index"] >= 0)
        self.index = np.flatnonzero(valid)
        if self.index.size == 0:
            raise ValueError(f"{path}: no valid transitions")
        self.diagnostic = self.arrays["diagnostic"].astype(bool)[self.index]

    @property
    def compatibility(self) -> Dict[str, Any]:
        return compatibility(self.identity)

    @property
    def feat_dim(self) -> int:
        return int(self.arrays["feat"].shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.arrays["action"].shape[-1])

    def __len__(self) -> int:
        return int(self.index.size)

    def progress(self) -> Optional[Dict[str, np.ndarray]]:
        if "progress_phi" not in self.arrays:
            return None
        return {"phi": self.arrays["progress_phi"], "valid": self.arrays["progress_valid"].astype(bool)}

    def to_torch(self, device, include_progress: bool = False) -> "TorchTransitions":
        return TorchTransitions.from_source(self, device, include_progress)


class TorchTransitions:
    """Transitions on the training device, sampled by row. ``subset`` shares the feature tensor."""

    def __init__(self, torch_module, device, feat, rows: Dict[str, Any], progress: Optional[Dict[str, Any]]):
        self.torch = torch_module
        self.device = device
        self.feat = feat
        self.cur, self.nxt = rows["cur"], rows["nxt"]
        self.action, self.reward, self.done = rows["action"], rows["reward"], rows["done"]
        self.diagnostic_mask = rows["diagnostic"]
        self.progress = progress
        self.size = int(self.cur.shape[0])

    @classmethod
    def from_source(cls, source: LatentTransitions, device, include_progress: bool) -> "TorchTransitions":
        import torch

        device = torch.device(device)
        arrays, index = source.arrays, source.index
        nxt = arrays["next_index"][index]
        rows = {
            "cur": torch.as_tensor(index, device=device, dtype=torch.long),
            "nxt": torch.as_tensor(nxt, device=device, dtype=torch.long),
            "action": torch.as_tensor(arrays["action"][index], device=device, dtype=torch.float32),
            "reward": torch.as_tensor(arrays["reward"][index], device=device, dtype=torch.float32),
            "done": torch.as_tensor(arrays["done"][index].astype(np.float32), device=device),
            "diagnostic": torch.as_tensor(source.diagnostic, device=device),
        }
        progress = None
        if include_progress:
            if "progress_phi" not in arrays:
                raise ValueError("this latent cache was encoded without progress potentials")
            phi = np.nan_to_num(arrays["progress_phi"].astype(np.float32), nan=0.0)
            valid = arrays["progress_valid"].astype(bool)
            progress = {
                "phi": torch.as_tensor(phi[index], device=device),
                "phi_next": torch.as_tensor(phi[nxt], device=device),
                "valid": torch.as_tensor((valid[index] & valid[nxt]).astype(np.float32), device=device),
            }
        # Features stay float16 on the device and are cast per batch.
        feat = torch.as_tensor(arrays["feat"], device=device)
        return cls(torch, device, feat, rows, progress)

    def subset(self, mask) -> "TorchTransitions":
        """The rows where ``mask`` (a boolean over this object's rows) holds, sharing the features."""
        torch = self.torch
        keep = torch.as_tensor(mask, device=self.device).bool()
        rows = {"cur": self.cur[keep], "nxt": self.nxt[keep], "action": self.action[keep],
                "reward": self.reward[keep], "done": self.done[keep], "diagnostic": self.diagnostic_mask[keep]}
        progress = None if self.progress is None else {key: value[keep] for key, value in self.progress.items()}
        return TorchTransitions(torch, self.device, self.feat, rows, progress)

    def diagnostic_rows(self) -> "TorchTransitions":
        return self.subset(self.diagnostic_mask)

    def batch(self, rows) -> Dict[str, Any]:
        torch = self.torch
        out = {
            "z": self.feat[self.cur[rows]].float(),
            "z_next": self.feat[self.nxt[rows]].float(),
            "action": self.action[rows],
            "reward": self.reward[rows],
            "cont": 1.0 - self.done[rows],
            "synthetic": torch.zeros(rows.shape[0], device=self.device),
        }
        if self.progress is not None:
            out["phi"] = self.progress["phi"][rows]
            out["phi_next"] = self.progress["phi_next"][rows]
            out["phi_valid"] = self.progress["valid"][rows]
        return out

    def sample(self, batch_size: int, generator=None) -> Dict[str, Any]:
        rows = self.torch.randint(0, self.size, (int(batch_size),), device=self.device, generator=generator)
        return self.batch(rows)

    def iterate(self, batch_size: int):
        for start in range(0, self.size, int(batch_size)):
            rows = self.torch.arange(start, min(start + int(batch_size), self.size), device=self.device)
            yield self.batch(rows)


class SyntheticTransitions:
    """Imagined transitions written by ``generate_rollouts``, kept apart from recorded ones."""

    def __init__(self, root: str, expected_compatibility: Mapping[str, Any]):
        self.root = repo_path(root)
        meta_path = os.path.join(self.root, "rollouts.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"no imagined transitions at {meta_path}; run generate_rollouts first")
        self.meta = read_json(meta_path)
        if self.meta.get("format") in EARLIER_FORMATS:
            raise ValueError(f"{meta_path} was written by an earlier version of this package; regenerate it")
        if self.meta.get("format") != ROLLOUT_FORMAT:
            raise ValueError(f"{meta_path}: not a set of imagined transitions")
        require_identity(expected_compatibility, self.meta["latent_compatibility"],
                         f"imagined transitions in {self.root}", fields=COMPATIBILITY_FIELDS)
        shards = sorted(glob.glob(os.path.join(self.root, "shard_*.npz")))
        if len(shards) != int(self.meta["shards"]):
            raise ValueError(f"{self.root}: {len(shards)} shards on disk, rollouts.json lists {self.meta['shards']}")
        parts: Dict[str, List[np.ndarray]] = {}
        for shard in shards:
            with np.load(shard, allow_pickle=False) as data:
                for key in data.files:
                    parts.setdefault(key, []).append(data[key])
        self.arrays = {key: np.concatenate(values) for key, values in parts.items()}
        if int(self.arrays["z"].shape[0]) != int(self.meta["transitions"]):
            raise ValueError(f"{self.root}: shards hold {self.arrays['z'].shape[0]} transitions, "
                             f"rollouts.json lists {self.meta['transitions']}")

    def identity(self) -> Dict[str, Any]:
        """What a policy run built on these transitions records about them."""
        return {"format": self.meta["format"], "settings": self.meta["settings_hash"],
                "behavior_policy": self.meta["behavior_policy"]["sha256"],
                "transitions": int(self.meta["transitions"]), "created": self.meta["created"]}

    def __len__(self) -> int:
        return int(self.arrays["z"].shape[0])

    def to_torch(self, device) -> "TorchSynthetic":
        return TorchSynthetic(self, device)


class TorchSynthetic:
    def __init__(self, source: SyntheticTransitions, device):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        a = source.arrays
        self.z = torch.as_tensor(a["z"], device=self.device)
        self.z_next = torch.as_tensor(a["z_next"], device=self.device)
        self.action = torch.as_tensor(a["action"], device=self.device, dtype=torch.float32)
        self.reward = torch.as_tensor(a["reward"], device=self.device, dtype=torch.float32)
        self.cont = torch.as_tensor(a["cont"], device=self.device, dtype=torch.float32)
        self.size = int(self.z.shape[0])

    def batch(self, rows) -> Dict[str, Any]:
        torch = self.torch
        return {
            "z": self.z[rows].float(),
            "z_next": self.z_next[rows].float(),
            "action": self.action[rows],
            "reward": self.reward[rows],
            "cont": self.cont[rows],
            "synthetic": torch.ones(rows.shape[0], device=self.device),
        }

    def sample(self, batch_size: int, generator=None) -> Dict[str, Any]:
        rows = self.torch.randint(0, self.size, (int(batch_size),), device=self.device, generator=generator)
        return self.batch(rows)

    def head(self, count: int) -> Dict[str, Any]:
        """The first ``count`` rows: a fixed set for diagnostics."""
        return self.batch(self.torch.arange(0, min(int(count), self.size), device=self.device))
