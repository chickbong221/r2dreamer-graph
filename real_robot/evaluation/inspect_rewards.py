"""Distance curves, milestone labels, stages and dense reward, with the reward checks.

    python -m real_robot.evaluation.inspect_rewards --episodes pilot
    python -m real_robot.evaluation.inspect_rewards --episodes pilot --scales defaults

One figure per episode, four aligned panels over frames:

1. the estimated distances the reward reads, with gaps where geometry is unknown;
2. Gemini's milestone labels as bands, and its events as vertical lines;
3. the stage, the staged score ``S_t`` and the verified gripper closure;
4. the dense reward per transition and the completion frame.

``reward_checks.json`` collects every check on every episode -- smoothness
during approach and transport, no transport credit without a grasp, an empty
pot never completing, no dependence on episode length, completion not before
the observed event, truncation not treated as termination, failure not a
shortcut -- so a failing one names its episode and frames.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..common import add_config_arguments, episode_name, load_configs, repo_path, write_json

DISTANCES = (
    ("d_gripper_banana", "gripper to banana"),
    ("d_banana_pot_entry", "banana to pot entry"),
    ("banana_placement_error", "banana placement (planar)"),
    ("d_gripper_lid_handle", "gripper to lid handle"),
    ("lid_lateral_error", "lid lateral error"),
    ("lid_height_above_rim", "lid height above rim"),
)
BANDS = (
    ("grasp", "ee", "banana", "grasp(ee, banana)"),
    ("grasp", "ee", "lid", "grasp(ee, lid)"),
    ("contain", "pot", "banana", "pot contains banana"),
    ("support", "pot", "lid", "pot supports lid"),
)


def plot_episode(path: str, episode: int, inputs, result, annotation, spec, geometry) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result.stage)
    frames = np.arange(n)
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.6, 2, 2]})

    ax = axes[0]
    for key, label in DISTANCES:
        ax.plot(frames, geometry[key], label=label, linewidth=1.4)
    ax.axhline(0.0, color="#999", linewidth=0.6)
    ax.set_ylabel("metres")
    ax.set_title(f"{episode_name(episode)} ({annotation.mode})  scene frame: {geometry['meta']['scene_frame']}")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1]
    for row, (relation, a, b, label) in enumerate(BANDS):
        mask = (annotation.holds(spec, relation, a, b) if relation == "grasp"
                else annotation.held_by(spec, relation, a, b))
        for start, end in _runs(mask):
            ax.broken_barh([(start, end - start + 1)], (row - 0.4, 0.8), color=f"C{row}")
    ax.set_yticks(range(len(BANDS)))
    ax.set_yticklabels([b[3] for b in BANDS], fontsize=8)
    for event in annotation.events:
        for axis in axes:
            axis.axvline(event["frame"], color="#555", linewidth=0.7, linestyle=":")
        ax.text(event["frame"], len(BANDS) - 0.4, f"{event['type']}\n{event['object']}", fontsize=7,
                rotation=90, va="bottom", ha="right")
    ax.set_ylim(-0.6, len(BANDS) + 1.2)

    ax = axes[2]
    ax.step(frames, result.stage, where="post", label="stage k", color="#333")
    ax.plot(frames, result.S * 6.0, label="6 * S_t = k + q", color="C1")
    ax.plot(frames, result.terms["gripper_closure"] * 5.0, label="gripper closure (x5)", color="C2", alpha=0.6)
    imputed = np.flatnonzero(result.terms["q_imputed"])
    if imputed.size:
        ax.scatter(imputed, result.S[imputed] * 6.0, s=6, color="red", label="q imputed")
    ax.set_ylabel("stage")
    ax.set_yticks(range(7))
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    ax.plot(frames, result.reward, color="C3", label="r_t (transition t -> t+1)")
    if result.completion_frame >= 0:
        for axis in axes:
            axis.axvline(result.completion_frame, color="green", linewidth=1.4)
        ax.text(result.completion_frame, 0.9, " verified completion", color="green", fontsize=8)
    observed = int(inputs.observed_completion_frame)
    if observed >= 0:
        ax.axvline(observed, color="purple", linewidth=1.0, linestyle="--")
        ax.text(observed, 0.6, " Gemini completion", color="purple", fontsize=8)
    ax.set_ylim(-1.05, 1.1)
    ax.set_xlabel("frame")
    ax.set_ylabel("reward")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _runs(mask: np.ndarray):
    start = None
    for t, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = t
        elif not value and start is not None:
            yield start, t - 1
            start = None


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource
    from ..preprocessing.freshness import ArtifactChain, warn_stale
    from ..rewards.kitchen import (RewardScales, compute_rewards, critical_failures, fallback_report, load_scales,
                                   reward_checks)

    parser = argparse.ArgumentParser(description="Plot distances and dense rewards; run the reward checks.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    parser.add_argument("--scales", choices=("fitted", "defaults"), default="fitted",
                        help="defaults: before scales are fitted (the plots say so)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph", "reward"], args.overrides)
    source = RawEpisodeSource(configs, mode=args.mode)
    action_spec = source.action_spec()
    if not action_spec.get("verified"):
        print("[reward] the action mapping is not verified yet; gripper closure uses "
              f"{action_spec['gripper']['source']} open/closed values", flush=True)
    reward_cfg = configs["reward"]
    scales = (RewardScales.from_defaults(reward_cfg) if args.scales == "defaults"
              else load_scales(reward_cfg, required=True))
    out_dir = os.path.join(repo_path(configs["dataset"]["paths"]["renders"]), "rewards")

    report: Dict[str, Any] = {"scales": scales.to_json(), "episodes": {}}
    failures: List[str] = []
    episodes = source.select(args.episodes)
    report["stale"] = {str(e): p for e, p in warn_stale(ArtifactChain(configs, source), episodes, "geometry",
                                                        "[reward]").items()}
    for episode in episodes:
        annotation = source.annotation(episode)
        geometry = source.geometry(episode)
        inputs = source.reward_inputs(episode)
        result = compute_rewards(inputs, scales, reward_cfg)
        checks = reward_checks(inputs, result, scales, reward_cfg)
        plot_path = os.path.join(out_dir, f"{episode_name(episode)}_{annotation.mode}_reward.png")
        plot_episode(plot_path, episode, inputs, result, annotation, source.spec, geometry)
        fallbacks = fallback_report(result)
        critical = [c["name"] for c in critical_failures(checks)]
        report["episodes"][str(episode)] = {
            "plot": plot_path,
            "fallbacks": fallbacks,
            "critical_failures": critical,
            "completion_frame": result.completion_frame,
            "gemini_completion_frame": int(inputs.observed_completion_frame),
            "stage_frames": {int(k): int((result.stage == k).sum()) for k in range(6)},
            "return": float(np.nansum(result.reward[result.reward_valid])),
            "notes": result.notes,
            "checks": checks,
        }
        failed = [c["name"] for c in checks if c["passed"] is False]
        failures += [f"episode {episode}: {name}" for name in failed]
        print(f"[reward] episode {episode}: completion {result.completion_frame} "
              f"(Gemini {inputs.observed_completion_frame}), return {report['episodes'][str(episode)]['return']:.1f}, "
              f"distance fallbacks {fallbacks['imputed_fraction']:.1%} (longest run {fallbacks['longest_imputed_run']}), "
              f"failed checks: {', '.join(failed) or 'none'}"
              f"{' -- CRITICAL: ' + ', '.join(critical) + ' (build_dataset refuses this episode)' if critical else ''}"
              f" -> {plot_path}", flush=True)
        for note in result.notes:
            print(f"    note: {note}")
    path = os.path.join(out_dir, f"reward_checks_{source.mode}.json")
    write_json(path, report)
    print(f"[reward] checks -> {path}")
    if failures:
        print("[reward] failing checks:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
