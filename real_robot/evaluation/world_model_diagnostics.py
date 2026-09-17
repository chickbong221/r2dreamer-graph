"""World-model diagnostics on the fixed diagnostic episodes.

The diagnostic episodes are training episodes. What is measured here is how
well the model fits them and whether the pipeline behaves:

* **losses** over fixed burn-in windows: every reconstruction and graph term,
  the reward head's error and the continuation head's accuracy;
* **one-step and short open-loop prediction**: from the posterior state at a
  recorded frame, the recorded actions are replayed through the latent
  dynamics alone, and the predicted reward, continuation, decoded state and
  decoded images are compared with the recording at every step, together with
  how far the predicted latent drifts from the posterior. Step one is the
  one-step error. The graph decoder is not used: imagined states have no
  observed geometry, and handing it recorded boxes would read the future;
* **burn-in sensitivity**: the recurrent state rebuilt from a burn-in window,
  as training rebuilds it, against the state from running the whole episode
  from its first frame, at several steps after the burn-in.

None of it measures generalisation; every report carries that note.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np
import torch
from torch.amp import autocast

from ..data.selection import DIAGNOSTIC_NOTE
from ..data.sequence_dataset import SequenceSampler, episode_window, full_episode, last_observation
from ..models.world_model import to_torch


class DiagnosticWindows:
    """Burn-in windows from the diagnostic episodes, drawn once with a fixed seed."""

    def __init__(self, store, episodes: Sequence[int], burn_in: int, length: int, batch_size: int, count: int,
                 seed: int):
        self.sampler = SequenceSampler(store, [int(e) for e in episodes], burn_in, length, batch_size, seed=seed)
        self.draws = [self.sampler.draw() for _ in range(int(count))]
        self.burn_in = int(burn_in)
        self.length = int(length)
        self.batch_size = int(batch_size)

    @property
    def episodes(self) -> List[int]:
        return list(self.sampler.episodes)

    def batches(self) -> Iterator[Dict[str, np.ndarray]]:
        for start in range(0, len(self.draws), self.batch_size):
            yield self.sampler.batch(self.draws[start:start + self.batch_size])


class _EvalMode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        self.model.eval()
        return self.model

    def __exit__(self, *exc):
        if self.was_training:
            self.model.train()
        return False


def loss_metrics(model, windows: DiagnosticWindows, device) -> Dict[str, float]:
    """Every world-model loss and head metric, averaged over the fixed windows."""
    device = torch.device(device)
    totals: Dict[str, float] = {}
    weight = 0.0
    with _EvalMode(model), torch.no_grad(), autocast(device_type=device.type, dtype=model.amp_dtype,
                                                     enabled=device.type == "cuda"):
        for batch in windows.batches():
            rows = float(batch["is_first"].shape[0])
            _, metrics, _ = model.compute_losses(to_torch(batch, device), windows.burn_in)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * rows
            weight += rows
    return {key: value / max(weight, 1.0) for key, value in totals.items()}


@torch.no_grad()
def open_loop_metrics(model, store, episodes: Sequence[int], horizon: int, starts_per_episode: int,
                      device) -> Dict[str, Any]:
    """Recorded actions through the latent dynamics alone, compared with the recording per step."""
    device = torch.device(device)
    horizon = int(horizon)
    image_names = [f"{key}_mse" for key in model.image_keys]
    names = ["reward_abs", "cont_abs", "cont_correct", "state_mse", "deter_relative_l2", "stoch_agreement",
             "sem_cosine"] + image_names
    sums = {name: np.zeros(horizon) for name in names}
    counts = np.zeros(horizon)
    with _EvalMode(model):
        for episode in episodes:
            arrays = store.load(episode)
            last = last_observation(arrays)
            valid = np.asarray(arrays["transition_valid"], dtype=bool)
            candidates = [t for t in range(0, max(last, 0)) if valid[t]]
            if not candidates:
                continue
            picks = np.linspace(0, len(candidates) - 1, min(int(starts_per_episode), len(candidates)))
            starts = np.asarray(sorted({candidates[int(round(float(p)))] for p in picks}), dtype=np.int64)
            batch = to_torch(full_episode(store, episode), device)
            posterior = model.encode_sequence(batch)["feat"][0].float()          # (T, F), T = last + 1
            states = batch["state"][0].float()
            feat = posterior[torch.as_tensor(starts, device=device)]
            alive = np.ones(len(starts), dtype=bool)
            for step in range(horizon):
                t = starts + step                                              # transition t -> t + 1
                alive &= (t < last)
                t = np.minimum(t, max(last - 1, 0))
                alive &= valid[t]
                if not alive.any():
                    break
                action = torch.as_tensor(arrays["action"][t], device=device, dtype=torch.float32)
                out = model.imagine(feat, action)
                feat = out["feat"].float()
                nxt = torch.as_tensor(t + 1, device=device, dtype=torch.long)
                target = posterior[nxt]
                mask = alive.astype(np.float64)

                reward_true = np.nan_to_num(np.asarray(arrays["reward"][t], dtype=np.float64), nan=0.0)
                cont_true = 1.0 - np.asarray(arrays["done"][t], dtype=np.float64)
                cont_prob = out["cont"].float().cpu().numpy().astype(np.float64)
                values = {
                    "reward_abs": np.abs(out["reward"].float().cpu().numpy() - reward_true),
                    "cont_abs": np.abs(cont_prob - cont_true),
                    "cont_correct": ((cont_prob > 0.5) == (cont_true > 0.5)).astype(np.float64),
                }
                stoch_p, sem_p, deter_p = model.split_feat(feat)
                stoch_q, sem_q, deter_q = model.split_feat(target)
                values["deter_relative_l2"] = (torch.linalg.vector_norm(deter_p - deter_q, dim=-1)
                                               / torch.linalg.vector_norm(deter_q, dim=-1).clamp_min(1e-6)
                                               ).cpu().numpy()
                values["stoch_agreement"] = (stoch_p.argmax(-1) == stoch_q.argmax(-1)).float().mean(-1).cpu().numpy()
                values["sem_cosine"] = torch.nn.functional.cosine_similarity(sem_p, sem_q, dim=-1).cpu().numpy()
                decoded = model.decode(feat)
                values["state_mse"] = (decoded["state"].mode()[:, 0] - states[nxt]).square().mean(-1).cpu().numpy()
                for key, name in zip(model.image_keys, image_names):
                    predicted = decoded[key].mode()[:, 0]
                    recorded = batch[key][0][nxt].float() / 255.0
                    values[name] = (predicted - recorded).square().mean(dim=(-3, -2, -1)).cpu().numpy()
                for name, value in values.items():
                    sums[name][step] += float((np.asarray(value, dtype=np.float64) * mask).sum())
                counts[step] += float(mask.sum())
    by_step = {name: [float(sums[name][h] / counts[h]) if counts[h] else None for h in range(horizon)]
               for name in names}
    return {"horizon": horizon, "episodes": [int(e) for e in episodes],
            "samples_by_step": [int(c) for c in counts], "by_step": by_step}


@torch.no_grad()
def burn_in_metrics(model, store, episodes: Sequence[int], burn_in: int, length: int, probes: Sequence[int],
                    device) -> Dict[str, Any]:
    """Whole-episode inference against burn-in windows, at the same frames."""
    device = torch.device(device)
    probes = [int(p) for p in probes]
    rows: List[Dict[str, float]] = []
    with _EvalMode(model):
        for episode in episodes:
            arrays = store.load(episode)
            last = last_observation(arrays)
            start = min(max(burn_in + 5, last // 2), max(last - length, burn_in + 1))
            if start <= 0 or start + max(probes) > last:
                continue
            whole = model.encode_sequence(to_torch(full_episode(store, episode), device))
            window = episode_window(arrays, store.manifest.image_keys, start, burn_in, length)
            piece = model.encode_sequence(to_torch({k: v[None] for k, v in window.items()}, device))
            for offset in probes:
                frame = start + offset
                if frame > last or burn_in + offset >= piece["feat"].shape[1]:
                    continue
                reference = whole["feat"][0, frame].float()
                approximate = piece["feat"][0, burn_in + offset].float()
                stoch_r, sem_r, deter_r = model.split_feat(reference)
                stoch_a, sem_a, deter_a = model.split_feat(approximate)
                rows.append({
                    "episode": int(episode), "frame": int(frame), "steps_after_burn_in": int(offset),
                    "deter_relative_l2": float(torch.norm(deter_a - deter_r) / torch.norm(deter_r).clamp_min(1e-6)),
                    "sem_cosine": float(torch.nn.functional.cosine_similarity(sem_a[None], sem_r[None])[0]),
                    "stoch_agreement": float((stoch_a.argmax(-1) == stoch_r.argmax(-1)).float().mean()),
                    "reward_difference": float(abs(piece["reward"][0, burn_in + offset] - whole["reward"][0, frame])),
                })
    summary = {}
    for offset in probes:
        subset = [r for r in rows if r["steps_after_burn_in"] == offset]
        if subset:
            summary[str(offset)] = {key: float(np.mean([r[key] for r in subset]))
                                    for key in ("deter_relative_l2", "sem_cosine", "stoch_agreement",
                                                "reward_difference")}
    return {"per_probe": summary, "rows": rows}


def summarize(report: Mapping[str, Any]) -> Dict[str, float]:
    """Flat ``diagnostic/...`` scalars for the run log."""
    out: Dict[str, float] = {}
    for key, value in report["losses"].items():
        out[f"diagnostic/{key}"] = float(value)
    open_loop = report["open_loop"]
    for name, values in open_loop["by_step"].items():
        known = [(h, v) for h, v in enumerate(values) if v is not None]
        if not known:
            continue
        if known[0][0] == 0:
            out[f"diagnostic/one_step/{name}"] = float(known[0][1])
        step, value = known[-1]
        if step > 0:
            out[f"diagnostic/open_loop_h{step + 1}/{name}"] = float(value)
    for probe, values in report["burn_in"]["per_probe"].items():
        for name, value in values.items():
            out[f"diagnostic/burn_in_plus{probe}/{name}"] = float(value)
    return out


def run_diagnostics(model, store, windows: DiagnosticWindows, cfg: Mapping[str, Any], device) -> Dict[str, Any]:
    """Every diagnostic, as one report. ``cfg`` is the ``diagnostics`` section of ``world_model.yaml``."""
    episodes = windows.episodes
    report: Dict[str, Any] = {
        "note": DIAGNOSTIC_NOTE,
        "diagnostic_episodes": episodes,
        "diagnostic_episodes_in_training": True,
        "windows": [[int(e), int(s)] for e, s in windows.draws],
        "losses": loss_metrics(model, windows, device),
        "open_loop": open_loop_metrics(model, store, episodes, int(cfg["open_loop"]["horizon"]),
                                       int(cfg["open_loop"]["starts_per_episode"]), device),
        "burn_in": burn_in_metrics(model, store, episodes, windows.burn_in, windows.length,
                                   list(cfg["burn_in_probes"]), device),
    }
    report["scalars"] = summarize(report)
    return report
