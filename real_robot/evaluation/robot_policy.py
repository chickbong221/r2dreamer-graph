"""Run the trained policy on the matching robot.

    RGB + robot state + current graph -> recurrent world-model state -> IQL actor
    -> action conversion -> robot command

    python -m real_robot.evaluation.robot_policy --iql maiql_base --replay 12   # offline dry run

Everything that has to match training is read from the artifacts, not
re-specified here: the image resize, the proprioceptive features, the action
transform and its inverse, the latent convention, and the graph contract. The
world model is the exact checkpoint the policy's latent cache was encoded
with, checked by file hash and weight digest, and the policy is the run's
``final.pt`` unless another checkpoint is named.

Three things the robot side has to respect:

* **Graph timing.** The policy runs every control step; the graph arrives at
  its own update rate. The last graph is held between updates, which is what
  the past-only annotation mode reproduces in training. A graph older than
  ``max_graph_age`` is treated as absent: the graph token is masked rather
  than the state being conditioned on a stale scene.
* **Episode boundaries.** ``reset()`` clears the recurrent state and the graph
  history; the first step is flagged, exactly as ``is_first`` does in training.
* **Graph history.** The last ``K + 1`` graphs are kept, so a graph source that
  needs the temporal window can read it here rather than keeping its own.

The dense-reward annotator plays no part in action selection. Gemini is not in
this loop; if the graph source is too slow for the chosen update rate, a
smaller causal graph predictor is the component to add, and nothing above
changes.
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from ..common import add_config_arguments, load_configs, repo_path, require_identity, utc_now, write_json
from ..data.episode_dataset import ActionTransform, resize_image, state_features
from ..graphs.pack import empty_frame
from ..models.iql import IQL
from ..models.world_model import load_world_model
from ..training.checkpoints import load_checkpoint

GRAPH_KEYS_ORDER = ("graph_node_ent", "graph_node_bbox", "graph_node_centroid", "graph_node_target",
                    "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs", "graph_edge_temp")


class RobotPolicy:
    def __init__(self, world_model, manifest, actor, spec, device, latent: str = "mode",
                 max_graph_age: float = 0.5):
        self.model = world_model.eval()
        self.actor = actor.eval()
        self.manifest = manifest
        self.spec = spec
        self.device = torch.device(device)
        self.sample = latent == "sample"
        self.max_graph_age = float(max_graph_age)
        inputs = manifest.data["model_inputs"]
        self.image_size = list(inputs["image_size"])
        self.resize_mode = str(inputs["resize_mode"])
        self.state_feature_names = list(inputs["state_features"])
        self.effort_scale = float(inputs["effort_scale"])
        self.cameras = list(inputs["cameras"])
        action = manifest.data["action"]
        self.transform = ActionTransform.from_spec({
            "command_indices": action["indices"], "command_names": action["names"],
            "representation": action["representation"], "gripper": action.get("gripper", {}),
            "normalization": {"low": action["low"], "high": action["high"], "margin": action["margin"]},
        })
        self.temporal_window = int(manifest.data["graph"]["temporal_window"])
        self.history: collections.deque = collections.deque(maxlen=self.temporal_window + 1)
        self.empty_graph = empty_frame(spec)
        self.reset()

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        stoch, deter, sem = self.model.rssm.initial(1)
        self.state = (stoch, deter, sem)
        self.prev_action = torch.zeros(1, self.manifest.action_dim, device=self.device)
        self.first = True
        self.history.clear()
        self.steps = 0
        self.stale_graphs = 0
        self.latencies: list = []

    def graph_history(self) -> list:
        """The last ``K + 1`` graphs and their timestamps, oldest first."""
        return list(self.history)

    # -------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, images: Mapping[str, np.ndarray], robot_state: np.ndarray,
             graph: Optional[Mapping[str, np.ndarray]] = None, graph_timestamp: Optional[float] = None,
             now: Optional[float] = None, velocity: Optional[np.ndarray] = None,
             effort: Optional[np.ndarray] = None) -> np.ndarray:
        """One control step. Returns command values in the recorded units."""
        started = time.perf_counter()
        now = time.time() if now is None else now
        observation = {}
        for camera in self.cameras:
            key = f"image_{camera}"
            frame = images.get(camera, images.get(key))
            if frame is None:
                raise KeyError(f"no image for camera {camera!r}")
            resized = resize_image(np.asarray(frame), self.image_size, self.resize_mode)
            observation[key] = torch.as_tensor(resized, device=self.device)[None].float() / 255.0
        state = state_features(np.asarray(robot_state, dtype=np.float64)[None], velocity, effort,
                               self.state_feature_names, self.effort_scale)
        observation["state"] = torch.as_tensor(state, device=self.device).float()

        if graph is not None:
            self.history.append({"graph": {k: np.asarray(v) for k, v in graph.items()},
                                 "timestamp": float(graph_timestamp if graph_timestamp is not None else now)})
        recent = self.history[-1] if self.history else None
        fresh = recent is not None and (now - recent["timestamp"]) <= self.max_graph_age
        if recent is not None and not fresh:
            self.stale_graphs += 1
        packed = recent["graph"] if fresh else self.empty_graph
        graph_batch = {key: torch.as_tensor(np.asarray(packed[key]))[None].to(self.device)
                       for key in GRAPH_KEYS_ORDER}

        # The encoders take any leading batch shape; here it is a single step.
        embed = self.model.encoder(observation)
        token = self.model.graph_encoder(graph_batch).token
        if not fresh:
            token = torch.zeros_like(token)
        reset = torch.tensor([self.first], device=self.device)
        stoch, deter, sem = self.model.posterior_step(*self.state, self.prev_action, embed, token, reset,
                                                      sample=self.sample)
        self.state = (stoch, deter, sem)
        feat = self.model.rssm.get_feat(stoch, deter, sem)
        normalized = self.actor.act(feat, deterministic=True)
        self.prev_action = normalized
        self.first = False
        self.steps += 1
        self.latencies.append(time.perf_counter() - started)
        return self.transform.denormalize(normalized[0].cpu().numpy())

    def diagnostics(self) -> Dict[str, float]:
        latencies = np.asarray(self.latencies) if self.latencies else np.zeros(1)
        return {"steps": float(self.steps), "stale_graph_steps": float(self.stale_graphs),
                "policy_latency_ms_mean": float(latencies.mean() * 1000),
                "policy_latency_ms_p95": float(np.percentile(latencies, 95) * 1000)}


def load_policy(configs, iql_run: str, checkpoint: str = "final", device=None) -> RobotPolicy:
    from ..data.latent_dataset import compatibility, load_latent_identity
    from ..graphs.schema import GraphSpec

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(repo_path(configs["dataset"]["paths"]["runs"]), "iql", iql_run)
    path = os.path.join(run_dir, f"{checkpoint}.pt")
    payload = load_checkpoint(path)
    meta = payload["meta"]
    identity = load_latent_identity(meta["latents"])
    require_identity(payload["identity"]["latents"], compatibility(identity), f"policy {path} and its latent cache")
    world_model = identity["world_model"]
    model, _, manifest = load_world_model(repo_path(world_model["checkpoint"]), device,
                                          expected_dataset=identity["dataset"],
                                          expected_weights={"sha256": world_model["sha256"],
                                                            "weights": world_model["weights"]})
    agent = IQL(int(identity["latent"]["feat_dim"]), int(identity["latent"]["action_dim"]), meta["iql"],
                total_steps=1, progress_beta=float(meta.get("progress_beta", 0.0)))
    agent.load_state_dict(payload["state"]["model"])
    agent = agent.to(device)
    spec = GraphSpec.from_config(configs["graph"])
    return RobotPolicy(model, manifest, agent.actor, spec, device,
                       latent=str(identity["latent"]["inference"]))


def replay_episode(policy: RobotPolicy, configs, episode: int, graph_stride: int = 1) -> Dict[str, Any]:
    """Drive the wrapper from a recorded episode: the timing and action check before hardware."""
    from ..data.episode_dataset import BuiltEpisodeStore
    from ..preprocessing.prepare_videos import read_frames
    from ..data.episode_dataset import RawEpisodeSource

    store = BuiltEpisodeStore(policy.manifest.root)
    arrays = store.load(episode)
    source = RawEpisodeSource(configs)
    frames = {camera: read_frames(source.video_path(episode, camera)) for camera in policy.cameras}
    policy.reset()
    commands, recorded = [], []
    for t in range(int(arrays["obs_valid"].sum())):
        graph = None
        if t % max(1, graph_stride) == 0:
            graph = {key: arrays[key][t] for key in GRAPH_KEYS_ORDER}
        action = policy.step({camera: frames[camera][t] for camera in policy.cameras},
                             arrays["state_raw"][t], graph=graph, graph_timestamp=t / 15.0, now=t / 15.0)
        commands.append(action)
        recorded.append(policy.transform.denormalize(arrays["action"][t]))
    commands, recorded = np.asarray(commands), np.asarray(recorded)
    return {"episode": int(episode), "steps": len(commands),
            "command_mae_vs_recorded": [float(v) for v in np.abs(commands - recorded).mean(axis=0)],
            "diagnostics": policy.diagnostics()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Robot inference wrapper (offline dry run).")
    parser.add_argument("--iql", required=True, help="model-assisted IQL run name under runs/iql")
    parser.add_argument("--checkpoint", default="final", help="final (the run's policy) | latest | step_XXXXXXXX")
    parser.add_argument("--replay", type=int, default=None, help="replay this packed episode through the wrapper")
    parser.add_argument("--graph-stride", type=int, default=5,
                        help="deliver a graph every N control steps, as on the robot")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "graph"], args.overrides)
    policy = load_policy(configs, args.iql, args.checkpoint)
    if args.replay is None:
        print("[robot] policy loaded; import RobotPolicy and call step() from the robot's control loop")
        return
    report = replay_episode(policy, configs, args.replay, args.graph_stride)
    out = os.path.join(repo_path(configs["dataset"]["paths"]["runs"]), "iql", args.iql,
                       f"replay_episode_{args.replay:06d}.json")
    write_json(out, {**report, "created": utc_now(), "graph_stride": args.graph_stride})
    print(f"[robot] replayed episode {report['episode']}: {report['steps']} steps, "
          f"policy latency {report['diagnostics']['policy_latency_ms_mean']:.1f} ms mean / "
          f"{report['diagnostics']['policy_latency_ms_p95']:.1f} ms p95, "
          f"stale-graph steps {int(report['diagnostics']['stale_graph_steps'])}")
    print("  per-command MAE vs the recorded command: "
          + ", ".join(f"{name}={value:.4f}" for name, value in
                      zip(policy.transform.names, report["command_mae_vs_recorded"])))
    print(f"[robot] -> {out}")


if __name__ == "__main__":
    main()
