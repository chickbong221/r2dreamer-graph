"""Pack annotated episodes into the arrays training reads.

    python -m real_robot.preprocessing.build_dataset --episodes all --name full_episode_v1

For every episode this joins the recorded rows, the validated annotation, the
tracks and the geometry, and writes ``episodes/episode_XXXXXX/``:

* ``image_<camera>.npy`` -- ``(T, H, W, 3)`` uint8 at the model resolution;
* ``arrays.npz`` -- proprioceptive state, normalised commands, the raw fields,
  the nine packed graph arrays, the dense reward and its stages, and the
  validity flags (episode begin, recording end, task termination, valid
  observation, valid graph, valid transition);
* ``meta.json`` -- outcome, completion frame, reward checks, warnings.

Frames after a verified completion are recorded but are not part of the
episode for learning: they follow a terminal state. A final recorded frame is
otherwise an ordinary valid observation. A recorded action with no next
observation is not a valid transition.

``manifest.json`` carries the dataset identity and the episode selection:
every episode trains, and the diagnostic episodes are a subset of them, stored
by id. The build refuses an action specification that is not verified, a
selection that does not cover every episode of the pinned dataset, and reward
scales that were not fitted on every training episode.

Each episode is packed only from a current chain of artifacts, checked all the
way to its roots (:mod:`real_robot.preprocessing.freshness`): a valid annotation
made from the current graph, frozen bins, prompts, Gemini settings, prepared
videos and validation rules; tracks made from that annotation; depth from the
current weights and focal length; an alignment fitted on current tracks and
depth; geometry measured from all of those; and reward scales fitted on the
current annotation and geometry of every training episode -- if any of those is
stale, no episode is packed. A packed episode records the digests it was built
from and is rebuilt when any of them changes; an episode whose chain is stale,
or whose reward fails a critical check, is removed from the dataset and
reported. The manifest's content hash covers every packed episode's inputs, so
a world model trained before a rebuild refuses the rebuilt dataset.

Training reads these files only -- no Gemini, tracking or depth inference
happens after this stage.
"""

from __future__ import annotations

import argparse
import collections
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

from ..common import (
    add_config_arguments,
    load_configs,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)
from ..data.episode_dataset import ActionTransform, RawEpisodeSource, resize_image, state_features, transform_boxes
from ..data.manifest import MANIFEST_NAME, DatasetManifest
from ..data.selection import load_selection, selection_identity
from ..graphs.pack import pack_episode
from ..graphs.vocabulary import build_vocab, vocab_identity, vocab_sizes
from ..rewards.kitchen import (
    compute_rewards,
    critical_failures,
    fallback_report,
    inputs_from_artifacts,
    load_scales,
    reward_checks,
    reward_identity,
)
from .artifacts import file_digest
from .define_bins import load_frozen_bins
from .estimate_geometry import GeometryStage
from .freshness import ArtifactChain, problems_text
from .prepare_videos import read_frames


def dataset_identity(configs: Mapping[str, Mapping[str, Any]], source: RawEpisodeSource, transform: ActionTransform,
                     scales, bins: Mapping[str, Any], annotation_provenance: Mapping[str, Any],
                     scene_frame: str, selection: Mapping[str, Any],
                     action_spec: Mapping[str, Any]) -> Dict[str, Any]:
    spec = source.spec
    model_inputs = configs["dataset"]["model_inputs"]
    return {
        "source_revision": source.source_record()["resolved_revision"],
        "selection": selection_identity(selection),
        "annotation_mode": source.mode,
        "gemini_model": annotation_provenance.get("model"),
        "gemini_backend": annotation_provenance.get("backend"),
        "prompt_version": annotation_provenance.get("prompt_version"),
        "video_fps": annotation_provenance.get("video_fps"),
        "bins": bins["bins_hash"],
        "graph": stable_hash(spec.identity()),
        "temporal_window": spec.temporal_window,
        "tracking": stable_hash(configs["annotation"]["tracking"]),
        "geometry": GeometryStage(configs, mode=source.mode).settings_identity(),
        "scene_frame": scene_frame,
        "reward": reward_identity(configs["reward"], scales),
        "vocab": vocab_identity(build_vocab(spec)),
        "action_mapping": {"declaration": action_spec["declaration"]["hash"],
                           "confirmed_by": action_spec["declaration"]["confirmed_by"],
                           "gripper": {key: action_spec["gripper"][key]
                                       for key in ("index", "state_index", "units", "open_value", "closed_value")}},
        "action_transform": transform.identity(),
        "model_inputs": {
            "cameras": list(spec.cameras),
            "image_size": list(model_inputs["image_size"]),
            "resize_mode": model_inputs["resize_mode"],
            "state_features": list(model_inputs["state_features"]),
            "effort_scale": float(model_inputs["effort_scale"]),
        },
    }


def build_episode(configs, source: RawEpisodeSource, episode: int, transform: ActionTransform, scales,
                  out_dir: str, diagnostic: bool) -> Dict[str, Any]:
    spec = source.spec
    vocab = build_vocab(spec)
    model_inputs = configs["dataset"]["model_inputs"]
    reward_cfg = configs["reward"]
    action_spec = source.action_spec()

    table = source.table(episode)
    n = len(table["state"])
    annotation = source.annotation(episode)
    if annotation.n_frames != n:
        raise ValueError(f"episode {episode}: annotation covers {annotation.n_frames} frames, table {n}")
    tracks = source.tracks(episode)
    geometry = source.geometry(episode)

    os.makedirs(out_dir, exist_ok=True)
    size = list(model_inputs["image_size"])
    mode = model_inputs["resize_mode"]
    source_hw = None
    for camera in spec.cameras:
        frames = read_frames(source.video_path(episode, camera))
        if len(frames) != n:
            raise ValueError(f"episode {episode} {camera}: {len(frames)} frames for {n} rows")
        if camera == spec.cameras[0]:
            source_hw = frames.shape[1:3]
        resized = np.stack([resize_image(frame, size, mode) for frame in frames])
        np.save(os.path.join(out_dir, f"image_{camera}.npy"), resized)

    boxes = transform_boxes(tracks["boxes"], source_hw, size, mode)
    graph, graph_valid = pack_episode(spec, vocab, annotation, boxes, tracks["visible"],
                                      geometry["centroids"], geometry["centroid_known"])

    gripper = action_spec["gripper"]
    inputs = inputs_from_artifacts(spec, annotation, geometry, table["state"][:, int(gripper["state_index"])], gripper)
    result = compute_rewards(inputs, scales, reward_cfg)
    checks = reward_checks(inputs, result, scales, reward_cfg)

    fallbacks = fallback_report(result)
    completion = result.completion_frame
    terminal_frame = completion if completion >= 0 else (result.failure_frame if result.failure_frame >= 0 else -1)
    obs_valid = np.ones(n, dtype=bool)
    if terminal_frame >= 0:
        obs_valid[terminal_frame + 1:] = False
    task_terminal = np.zeros(n, dtype=bool)
    if terminal_frame >= 0:
        task_terminal[terminal_frame] = True
    next_valid = np.zeros(n, dtype=bool)
    next_valid[:-1] = obs_valid[1:]
    transition_valid = result.reward_valid & obs_valid & next_valid
    episode_begin = np.zeros(n, dtype=bool)
    episode_begin[0] = True
    recording_end = np.zeros(n, dtype=bool)
    recording_end[-1] = True

    arrays = {
        "state": state_features(table["state"], table.get("velocity"), table.get("effort"),
                                model_inputs["state_features"], model_inputs["effort_scale"]),
        "state_raw": table["state"].astype(np.float32),
        "action": transform.normalize(table["action"]),
        "action_raw": table["action"].astype(np.float32),
        "velocity": table.get("velocity", np.zeros((n, 7))).astype(np.float32),
        "effort": table.get("effort", np.zeros((n, 7))).astype(np.float32),
        "timestamp": table["timestamp"].astype(np.float32),
        "frame_index": table["frame_index"].astype(np.int32),
        "episode_begin": episode_begin,
        "recording_end": recording_end,
        "task_terminal": task_terminal,
        "obs_valid": obs_valid,
        "graph_valid": graph_valid & obs_valid,
        "reward": np.where(transition_valid, result.reward, np.nan).astype(np.float32),
        "done": result.done & transition_valid,
        "transition_valid": transition_valid,
        "stage": result.stage.astype(np.int8),
        "stage_q": result.q.astype(np.float32),
        "staged_score": result.staged_score.astype(np.float32),
        "stage_S": result.S.astype(np.float32),
        "centroid_known": geometry["centroid_known"],
        **{key: graph[key] for key in GRAPH_KEYS},
    }
    np.savez(os.path.join(out_dir, "arrays.npz"), **arrays)
    np.savez_compressed(os.path.join(out_dir, "reward_terms.npz"),
                        **{k: np.asarray(v) for k, v in result.terms.items()})
    failed = [c for c in checks if c["passed"] is False]
    meta = {
        "episode_index": int(episode),
        "n_frames": int(n),
        "training": True,
        "diagnostic": bool(diagnostic),
        "outcome": annotation.outcome,
        "events": annotation.events,
        "completion_frame": int(completion),
        "failure_frame": int(result.failure_frame),
        "reward_notes": result.notes,
        "reward_checks": checks,
        "reward_fallbacks": fallbacks,
        "annotation_warnings": annotation.warnings,
        "stage_frames": {int(k): int(v) for k, v in collections.Counter(result.stage.tolist()).items()},
        "return_undiscounted": float(np.nansum(arrays["reward"])),
        "graph_valid_fraction": float(np.mean(arrays["graph_valid"][obs_valid])),
        "scene_frame": geometry["meta"]["scene_frame"],
        "created": utc_now(),
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)
    return {"meta": meta, "failed_checks": failed, "checks": checks, "shapes": {
        "state": list(arrays["state"].shape[1:]), "action": list(arrays["action"].shape[1:]),
        "image": size + [3], **{key: list(graph[key].shape[1:]) for key in GRAPH_KEYS}}}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Pack annotated episodes into a training dataset.")
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--name", required=True, help="dataset directory name under paths.datasets")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    parser.add_argument("--force", action="store_true", help="rebuild episodes that are already packed")
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    configs = load_configs(["dataset", "annotation", "graph", "reward"], args.overrides)
    source = RawEpisodeSource(configs, mode=args.mode)
    spec = source.spec
    action_spec = source.action_spec()
    if not action_spec.get("verified"):
        raise SystemExit("[build] audit/action_spec.json is not verified; the action mapping has to be established "
                         "from evidence before anything trains on it:\n  "
                         + "\n  ".join(action_spec.get("problems") or ["(re-run audit_dataset)"]))
    transform = ActionTransform.from_spec(action_spec)
    try:
        selection = load_selection(configs["dataset"], source.available())
    except FileNotFoundError as exc:
        raise SystemExit(f"[build] {exc}")
    scales = load_scales(configs["reward"], required=True)
    fitted_on = sorted(int(e) for e in scales.provenance.get("episodes", []))
    if fitted_on != sorted(int(e) for e in selection["training"]):
        raise SystemExit(f"[build] reward scales were fitted on {len(fitted_on)} episodes, not on all "
                         f"{len(selection['training'])} training episodes. Run "
                         "`python -m real_robot.rewards.kitchen fit-scales --episodes all --force`.")
    bins = load_frozen_bins(configs["dataset"], spec)
    episodes = source.select(args.episodes)
    diagnostic = set(int(e) for e in selection["diagnostic"])

    # The scales enter every episode's reward, so every episode they were fitted on has to be current.
    chain = ArtifactChain(configs, source)
    stale_scales = chain.scales(scales)
    if stale_scales:
        raise SystemExit("[build] the reward scales were fitted on artifacts that are not current, so no reward "
                         "built with them can be trusted:\n  " + problems_text(stale_scales))
    chain_problems = {int(e): chain.geometry_chain(e) for e in episodes}
    current = [e for e in episodes if not chain_problems[int(e)]]
    if not current:
        raise SystemExit("[build] no episode has a current chain of artifacts:\n  " + problems_text(chain_problems))

    first = source.annotation(current[0])
    first_geometry = source.geometry(current[0])["meta"]
    identity = dataset_identity(configs, source, transform, scales, bins, first.provenance,
                                first_geometry["scene_frame"], selection, action_spec)
    root = os.path.join(repo_path(configs["dataset"]["paths"]["datasets"]), args.name)
    if os.path.isfile(os.path.join(root, MANIFEST_NAME)):
        manifest = DatasetManifest.load(root)
        manifest.require(identity, f"dataset {args.name}")
    else:
        from ..data.episode_dataset import state_dim

        model_inputs = configs["dataset"]["model_inputs"]
        manifest = DatasetManifest.create(
            root, identity=identity, selection=selection,
            shapes={"state": [state_dim(model_inputs["state_features"])], "action": [transform.dim]},
            model_inputs={**identity["model_inputs"],
                          "image_keys": [f"image_{c}" for c in spec.cameras]},
            graph={"vocab_sizes": vocab_sizes(build_vocab(spec)), "n_max": spec.n_max, "e_max": spec.e_max,
                   "n_cams": len(spec.cameras), "centroid_origin": spec.centroid_origin,
                   "centroid_scale": spec.centroid_scale, "scene_frame": identity["scene_frame"],
                   "temporal_window": spec.temporal_window},
            action={"indices": transform.indices, "names": transform.names,
                    "representation": transform.representation, "units": transform.units,
                    "low": transform.low.tolist(), "high": transform.high.tolist(), "margin": transform.margin,
                    "gripper": action_spec["gripper"]},
        )

    scale_inputs = scales.provenance.get("inputs") or {}
    shared_inputs = {"scales": stable_hash(reward_identity(configs["reward"], scales)),
                     "action_spec": file_digest(os.path.join(source.paths["audit"], "action_spec.json")),
                     "selection": manifest.selection_version}

    def episode_inputs(episode: int):
        """The digests an episode is packed from, and every reason its chain is not current."""
        problems = list(chain_problems[int(episode)])
        if str(int(episode)) not in scale_inputs:
            problems.append("the reward scales were not fitted on this episode; run rewards.kitchen fit-scales "
                            "--episodes all --force")
        inputs = {"annotation": file_digest(source.annotation_path(episode)),
                  "tracks": file_digest(source.tracks_path(episode)),
                  "geometry": file_digest(source.geometry_path(episode)), **shared_inputs}
        return inputs, problems

    summary = []
    refused: Dict[int, List[str]] = {}
    for episode in episodes:
        inputs, problems = episode_inputs(episode)
        previous = manifest.data.get("built", {}).get(str(episode))
        if problems:
            refused[episode] = problems
            if previous is not None:
                manifest.data["built"].pop(str(episode))
                manifest.save()
            continue
        if previous is not None and previous.get("inputs") == inputs and not args.force:
            continue
        out_dir = manifest.episode_dir(episode)
        annotation = source.annotation(episode)
        for key in ("model", "backend", "prompt_version"):
            if annotation.provenance.get(key) != first.provenance.get(key):
                raise SystemExit(f"[build] episode {episode} was annotated with a different {key} "
                                 f"({annotation.provenance.get(key)!r} vs {first.provenance.get(key)!r})")
        if annotation.mode != source.mode:
            raise SystemExit(f"[build] episode {episode} annotation mode {annotation.mode!r} != {source.mode!r}")
        geometry_meta = source.geometry(episode)["meta"]
        if geometry_meta["scene_frame"] != identity["scene_frame"]:
            raise SystemExit(f"[build] episode {episode} geometry is in {geometry_meta['scene_frame']}, "
                             f"the dataset in {identity['scene_frame']}")
        built = build_episode(configs, source, episode, transform, scales, out_dir, episode in diagnostic)
        meta = built["meta"]
        critical = critical_failures(built["checks"])
        if critical:
            refused[episode] = [f"reward check {c['name']} failed: {c['detail']}" for c in critical]
            if previous is not None:
                manifest.data["built"].pop(str(episode))
                manifest.save()
            continue
        manifest.data["shapes"].update(built["shapes"])
        manifest.record_episode(episode, {"n_frames": meta["n_frames"], "diagnostic": meta["diagnostic"],
                                          "completion_frame": meta["completion_frame"], "inputs": inputs,
                                          "reward_fallbacks": meta["reward_fallbacks"],
                                          "failed_checks": [c["name"] for c in built["failed_checks"]]})
        manifest.save()
        summary.append(meta)
        failed = ", ".join(c["name"] for c in built["failed_checks"]) or "none"
        print(f"[build] episode {episode}: {meta['n_frames']} frames, completion {meta['completion_frame']}, "
              f"return {meta['return_undiscounted']:.1f}, fallbacks "
              f"{meta['reward_fallbacks']['imputed_fraction']:.1%}, non-critical failed checks: {failed}", flush=True)
    manifest.save()
    coverage = manifest.coverage()
    print(f"[build] {len(summary)} episode(s) packed -> {manifest.root}")
    print(f"[build] {coverage['built']} of {coverage['training']} training episodes packed; "
          f"diagnostic episodes (also trained on) not yet packed: {coverage['diagnostic_missing'] or 'none'}")
    if refused:
        lines = [f"episode {episode}: " + "; ".join(reasons) for episode, reasons in sorted(refused.items())]
        raise SystemExit(f"[build] {len(refused)} episode(s) not packed:\n  " + "\n  ".join(lines))


if __name__ == "__main__":
    main()
