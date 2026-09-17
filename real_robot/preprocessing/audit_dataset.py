"""Check what the recorded fields are before anything trains on them.

    python -m real_robot.preprocessing.audit_dataset

The metadata names 13 state and 13 action dimensions under a key called
"displacement", records no units, and labels the right arm's velocity and
effort as left-arm fields. None of it is taken on trust -- and numbers are not
taken as proof either. What the action field commands is *declared* in
``configs/action_mapping.yaml``: the command dimensions, their representation
and units, and the gripper's open and closed values, each claim with the
evidence behind it. This stage measures the recordings and checks them against
that declaration:

1. which action dimensions equal the *next* recorded state to numerical
   precision -- labels written after the fact, not commands;
2. whether each dimension tracks an absolute target or a displacement;
3. the gripper's open and closed values and which way it closes;
4. the state/action timing: how many frames a command leads the state it
   produces;
5. whether every video has exactly one frame per recorded row, and whether
   timestamps are regular;
6. what velocity and effort measure, by comparing velocity with the time
   derivative of the state's joint angles;
7. how episodes end -- at rest or still moving. Whether that is completion is
   left to annotation.

A measurement can contradict a declaration; it can never certify one. Numerical
similarity between an action and a later state is what both a position target
and a hindsight label look like, and a gripper squeezing an object breaks the
similarity of a perfectly good command. So ``audit/action_spec.json`` is
``verified`` only when the declaration is confirmed by a named person, every
command dimension has a unit, the command and gripper claims each cite evidence
other than measurement, the data is intact, and no measurement contradicts the
declaration without an accepted, stated reason. ``build_dataset`` refuses
anything else.

Writes ``audit/audit_report.json`` with every number and
``audit/action_spec.json`` with the declared mapping, the checks, and the
normaliser fitted over every training episode.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common import (
    add_config_arguments,
    file_sha256,
    load_configs,
    load_yaml,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)

ANGLE_DIMS = (10, 11, 12)          # eef roll, pitch, yaw
JOINT_DIMS = tuple(range(0, 7))    # six joints and the gripper
LAGS = tuple(range(-2, 6))

ACTION_SPEC_FORMAT = "real_robot/action-spec-v2"
EVIDENCE_KINDS = ("metadata", "documentation", "recording_code", "hardware", "author", "measurement")
# Declared representation -> the measured classification it predicts.
REPRESENTATIONS = {"absolute_joint_position": "absolute_command", "joint_displacement": "displacement_command"}
UNKNOWN_UNITS = ("", "unknown", "none", "?")
GRIPPER_VALUE_TOLERANCE = 0.15      # of the declared open-closed span


def _wrap(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _dim_error(a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
    diff = a - b
    return np.abs(_wrap(diff) if dim in ANGLE_DIMS else diff)


def _pooled_lag_stats(tables: Dict[int, Dict[str, np.ndarray]], dim: int):
    """Per lag k: mean |action[t] - state[t + k]| and their correlation, pooled."""
    errors_by_lag, corr_by_lag = {}, {}
    for k in LAGS:
        errors, actions, states = [], [], []
        for table in tables.values():
            action, state = table["action"][:, dim], table["state"][:, dim]
            n = len(action)
            if k >= 0:
                a, s = action[: n - k], state[k:]
            else:
                a, s = action[-k:], state[: n + k]
            if len(a):
                errors.append(_dim_error(a, s, dim))
                actions.append(a)
                states.append(s)
        if not errors:
            errors_by_lag[k], corr_by_lag[k] = float("nan"), float("nan")
            continue
        a, s = np.concatenate(actions), np.concatenate(states)
        errors_by_lag[k] = float(np.mean(np.concatenate(errors)))
        corr_by_lag[k] = float(np.corrcoef(a, s)[0, 1]) if a.std() > 0 and s.std() > 0 else float("nan")
    return errors_by_lag, corr_by_lag


def audit_actions(tables: Dict[int, Dict[str, np.ndarray]], dims: Sequence[str]) -> List[Dict[str, Any]]:
    results = []
    for dim, name in enumerate(dims):
        next_errors, delta_errors, state_values, actions, deltas = [], [], [], [], []
        for table in tables.values():
            action, state = table["action"][:, dim], table["state"][:, dim]
            next_errors.append(_dim_error(action[:-1], state[1:], dim))
            delta = state[1:] - state[:-1]
            delta_errors.append(np.abs(action[:-1] - delta))
            state_values.append(state)
            actions.append(action[:-1])
            deltas.append(delta)
        next_err = np.concatenate(next_errors)
        delta_err = np.concatenate(delta_errors)
        state_std = float(np.std(np.concatenate(state_values)))
        lag_errors, lag_corr = _pooled_lag_stats(tables, dim)
        best_lag = min(lag_errors, key=lambda k: lag_errors[k])
        a, d = np.concatenate(actions), np.concatenate(deltas)
        delta_corr = float(np.corrcoef(a, d)[0, 1]) if a.std() > 0 and d.std() > 0 else float("nan")
        exact_next = bool(np.max(next_err) <= 1e-5)
        absolute_err = float(np.mean(next_err))
        delta_mean = float(np.mean(delta_err))
        # Correlation, not an error threshold: a gripper squeezing an object
        # holds its command well past the fingers for seconds, which is a large
        # error and still unmistakably a position target.
        if exact_next:
            kind = "derived_next_state"
        elif absolute_err < delta_mean and np.nan_to_num(lag_corr[best_lag]) >= 0.8:
            kind = "absolute_command"
        elif delta_mean < absolute_err and np.nan_to_num(delta_corr) >= 0.8:
            kind = "displacement_command"
        else:
            kind = "unresolved"
        results.append({
            "dim": dim, "name": name, "classification": kind,
            "max_abs_action_minus_next_state": float(np.max(next_err)),
            "mean_abs_action_minus_next_state": absolute_err,
            "mean_abs_action_minus_state_delta": delta_mean,
            "corr_action_with_state_delta": delta_corr,
            "state_std": state_std,
            "lag_errors": {str(k): v for k, v in lag_errors.items()},
            "lag_correlation": {str(k): v for k, v in lag_corr.items()},
            "best_lag_frames": int(best_lag),
        })
    return results


def audit_gripper(tables: Dict[int, Dict[str, np.ndarray]], dim: int = 6) -> Dict[str, Any]:
    state = np.concatenate([t["state"][:, dim] for t in tables.values()])
    action = np.concatenate([t["action"][:, dim] for t in tables.values()])
    first = np.array([t["state"][0, dim] for t in tables.values()])
    last = np.array([t["state"][-1, dim] for t in tables.values()])
    high = float(np.percentile(np.concatenate([state, action]), 99))
    low = float(np.percentile(action, 1))
    mid = (high + low) / 2.0
    # Episodes begin before anything is held, so the side the first frames sit
    # on is open.
    start_high = float(np.mean(first > mid))
    open_value, closed_value = (high, low) if start_high >= 0.5 else (low, high)
    # When the gripper squeezes an object, the command closes further than the
    # fingers can, so the command sits on the closed side of the measurement.
    gap = np.concatenate([t["action"][:, dim] - t["state"][:, dim] for t in tables.values()])
    squeeze = gap[np.abs(gap) > 0.05]
    command_side = float(np.mean(np.sign(squeeze))) if squeeze.size else 0.0
    consistent = (command_side < 0) == (closed_value < open_value) if squeeze.size else None
    return {
        "index": dim,
        "open_value": open_value,
        "closed_value": closed_value,
        "closing_direction": "decreasing" if closed_value < open_value else "increasing",
        "episodes_starting_open": float(np.mean(np.abs(first - open_value) < np.abs(first - closed_value))),
        "episodes_ending_open": float(np.mean(np.abs(last - open_value) < np.abs(last - closed_value))),
        "state_percentiles": {p: float(np.percentile(state, p)) for p in (1, 5, 50, 95, 99)},
        "action_percentiles": {p: float(np.percentile(action, p)) for p in (1, 5, 50, 95, 99)},
        "squeeze_frames": int(squeeze.size),
        "command_beyond_measurement_on_closed_side": consistent,
    }


def audit_velocity_effort(tables: Dict[int, Dict[str, np.ndarray]], fps: float) -> Dict[str, Any]:
    if not all("velocity" in t for t in tables.values()):
        return {"resolved": False, "reason": "no velocity field"}
    backward, forward, rms_v, rms_d = [], [], [], []
    for j in JOINT_DIMS:
        v_b, d_b, v_f, d_f = [], [], [], []
        for table in tables.values():
            state, velocity = table["state"][:, j], table["velocity"][:, j]
            diff = np.diff(state) * fps
            v_b.append(velocity[1:]); d_b.append(diff)
            v_f.append(velocity[:-1]); d_f.append(diff)
        vb, db = np.concatenate(v_b), np.concatenate(d_b)
        vf, df = np.concatenate(v_f), np.concatenate(d_f)
        backward.append(float(np.corrcoef(vb, db)[0, 1]) if vb.std() > 0 and db.std() > 0 else float("nan"))
        forward.append(float(np.corrcoef(vf, df)[0, 1]) if vf.std() > 0 and df.std() > 0 else float("nan"))
        rms_v.append(float(np.sqrt(np.mean(vb ** 2))))
        rms_d.append(float(np.sqrt(np.mean(db ** 2))))
    median_corr = float(np.nanmedian(backward))
    ratio = float(np.median(np.asarray(rms_v) / np.maximum(np.asarray(rms_d), 1e-9)))
    resolved = median_corr > 0.9 and 0.7 <= ratio <= 1.4
    effort = {}
    if all("effort" in t for t in tables.values()):
        values = np.concatenate([t["effort"] for t in tables.values()])
        effort = {"abs_max": [float(v) for v in np.max(np.abs(values), axis=0)],
                  "std": [float(v) for v in np.std(values, axis=0)]}
    return {
        "resolved": bool(resolved),
        "interpretation": ("joint velocity (rad/s) of the recorded (right) arm, whatever the metadata "
                           "names say" if resolved else "unresolved"),
        "corr_with_state_derivative_backward": backward,
        "corr_with_state_derivative_forward": forward,
        "rms_velocity": rms_v,
        "rms_state_derivative": rms_d,
        "median_rms_ratio": ratio,
        "effort": effort,
        "effort_note": ("recorded alongside velocity; its units are not verifiable from the data, so "
                        "it is kept out of the first model"),
    }


def audit_timing(tables: Dict[int, Dict[str, np.ndarray]], lengths: Dict[int, int], fps: float) -> Dict[str, Any]:
    problems = []
    dts = []
    for episode, table in tables.items():
        frames = table["frame_index"]
        if not np.array_equal(frames, np.arange(len(frames))):
            problems.append(f"episode {episode}: frame_index is not 0..N-1")
        if len(frames) != lengths.get(episode, -1):
            problems.append(f"episode {episode}: {len(frames)} rows, episodes.jsonl says {lengths.get(episode)}")
        dt = np.diff(table["timestamp"])
        dts.append(dt)
        if np.any(np.abs(dt - 1.0 / fps) > 0.25 / fps):
            problems.append(f"episode {episode}: irregular timestamps (dt range {dt.min():.4f}-{dt.max():.4f})")
    dt = np.concatenate(dts) if dts else np.zeros(0)
    return {"ok": not problems, "problems": problems[:50],
            "dt_mean": float(dt.mean()) if dt.size else None,
            "dt_min": float(dt.min()) if dt.size else None,
            "dt_max": float(dt.max()) if dt.size else None}


def audit_videos(source, episodes: Sequence[int], mode: str) -> Dict[str, Any]:
    from .prepare_videos import frame_times, video_metadata

    if mode == "none":
        return {"checked": "none"}
    lengths = source.lengths()
    problems, summary = [], {}
    for episode in episodes:
        rows = lengths[episode]
        table = source.table(episode) if mode == "decode" else None
        for camera in source.dataset_cfg["source"]["cameras"]:
            path = source.video_path(episode, camera)
            if not os.path.isfile(path):
                problems.append(f"episode {episode} {camera}: missing {path}")
                continue
            meta = video_metadata(path)
            summary.setdefault(camera, {"codec": meta["codec"], "size": [meta["width"], meta["height"]]})
            if mode == "decode":
                times = frame_times(path)
                if len(times) != rows:
                    problems.append(f"episode {episode} {camera}: {len(times)} frames for {rows} rows")
                    continue
                known = np.array([t for t in times if t is not None])
                if known.size == rows:
                    offset = known - known[0] - (table["timestamp"] - table["timestamp"][0])
                    if np.max(np.abs(offset)) > 0.5 / source.fps():
                        problems.append(f"episode {episode} {camera}: frame times drift from row timestamps "
                                        f"by up to {np.max(np.abs(offset)):.3f}s")
            elif meta["frames_declared"] and meta["frames_declared"] != rows:
                problems.append(f"episode {episode} {camera}: container declares "
                                f"{meta['frames_declared']} frames for {rows} rows")
    return {"checked": mode, "ok": not problems, "problems": problems[:100], "cameras": summary}


def audit_endings(tables: Dict[int, Dict[str, np.ndarray]], fps: float, tail: int = 10) -> Dict[str, Any]:
    speeds = {}
    for episode, table in tables.items():
        joints = table["state"][-tail:, 0:6]
        speeds[episode] = float(np.mean(np.abs(np.diff(joints, axis=0))) * fps) if len(joints) > 1 else 0.0
    values = np.array(list(speeds.values()))
    threshold = 0.05
    moving = sorted(e for e, s in speeds.items() if s > threshold)
    return {
        "tail_frames": tail,
        "rest_speed_threshold_rad_s": threshold,
        "median_tail_joint_speed": float(np.median(values)),
        "episodes_ending_in_motion": moving,
        "note": "motion at the end hints at truncation; completion is decided by annotation, not here",
    }


# --------------------------------------------------------------------------- #
# The declared mapping
# --------------------------------------------------------------------------- #
def _evidence_problems(claim: str, evidence: Any) -> List[str]:
    entries = [entry or {} for entry in (evidence or [])]
    problems = []
    for position, entry in enumerate(entries):
        if entry.get("kind") not in EVIDENCE_KINDS:
            problems.append(f"{claim}: evidence {position} has kind {entry.get('kind')!r}; use one of {EVIDENCE_KINDS}")
        if not str(entry.get("source") or "").strip() or not str(entry.get("statement") or "").strip():
            problems.append(f"{claim}: evidence {position} needs both a source and a statement")
    if not any(entry.get("kind") in EVIDENCE_KINDS and entry.get("kind") != "measurement" for entry in entries):
        problems.append(f"{claim}: no evidence other than measurement; numbers alone do not establish it")
    return problems


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(float(value)))


def _is_index(value: Any, size: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < size


def declaration_problems(declaration: Mapping[str, Any], n_dims: int) -> List[str]:
    """Everything the declaration itself lacks, independent of the recordings."""
    problems = []
    if declaration.get("confirmed") is not True:
        problems.append("the action mapping is not confirmed (action_mapping.yaml: confirmed)")
    if not str(declaration.get("confirmed_by") or "").strip():
        problems.append("action_mapping.yaml: confirmed_by is empty")

    command = declaration.get("command") or {}
    indices = list(command.get("indices") or [])
    names = list(command.get("names") or [])
    valid = [i for i in indices if _is_index(i, n_dims)]
    if not indices:
        problems.append("no command dimensions are declared")
    if len(valid) != len(indices):
        problems.append(f"command indices {indices} are not all dimensions 0..{n_dims - 1}")
    if len(set(valid)) != len(valid):
        problems.append(f"command indices {indices} repeat")
    if len(names) != len(indices):
        problems.append(f"{len(names)} command names for {len(indices)} command indices")
    if command.get("representation") not in REPRESENTATIONS:
        problems.append(f"command representation {command.get('representation')!r} is not one of "
                        f"{sorted(REPRESENTATIONS)}")
    units = command.get("units") or {}
    unknown = [name for name in names if str(units.get(name, "")).strip().lower() in UNKNOWN_UNITS]
    if unknown:
        problems.append(f"no known unit for command dimensions {unknown}")
    problems.extend(_evidence_problems("command", command.get("evidence")))

    derived_raw = list((declaration.get("derived") or {}).get("indices") or [])
    derived = [i for i in derived_raw if _is_index(i, n_dims)]
    if len(derived) != len(derived_raw):
        problems.append(f"derived indices {derived_raw} are not all dimensions 0..{n_dims - 1}")
    both = sorted(set(derived) & set(valid))
    if both:
        problems.append(f"dimensions {both} are declared both command and derived")
    undeclared = sorted(set(range(n_dims)) - set(derived) - set(valid))
    if undeclared:
        problems.append(f"action dimensions {undeclared} are declared neither command nor derived")

    gripper = declaration.get("gripper") or {}
    if gripper.get("action_index") not in valid:
        problems.append(f"gripper action_index {gripper.get('action_index')!r} is not a command dimension")
    if not _is_index(gripper.get("state_index"), n_dims):
        problems.append(f"gripper state_index {gripper.get('state_index')!r} is not a state dimension")
    if str(gripper.get("units", "")).strip().lower() in UNKNOWN_UNITS:
        problems.append("no known unit for the gripper")
    if not (_is_number(gripper.get("open_value")) and _is_number(gripper.get("closed_value"))):
        problems.append("the gripper's open_value and closed_value are not both given")
    elif float(gripper["open_value"]) == float(gripper["closed_value"]):
        problems.append("the gripper's open_value equals its closed_value")
    problems.extend(_evidence_problems("gripper", gripper.get("evidence")))

    for position, accepted in enumerate(declaration.get("accepted_contradictions") or []):
        accepted = accepted or {}
        if not str(accepted.get("check") or "").strip() or not str(accepted.get("reason") or "").strip():
            problems.append(f"accepted_contradictions {position} needs both a check and a reason")
    return problems


def measured_contradictions(declaration: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
                            gripper: Optional[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Where the recordings disagree with the declaration. Disagreement blocks; agreement proves nothing."""
    found: List[Dict[str, str]] = []
    by_dim = {int(r["dim"]): r for r in actions}
    command = declaration.get("command") or {}
    expected = REPRESENTATIONS.get(command.get("representation"))
    names = list(command.get("names") or [])
    for position, index in enumerate(command.get("indices") or []):
        measured = by_dim.get(index) if _is_index(index, len(by_dim)) else None
        if measured is None:
            continue
        name = names[position] if position < len(names) else str(index)
        kind = measured["classification"]
        if kind == "derived_next_state":
            found.append({"check": f"command_is_not_derived:{name}",
                          "detail": "declared a command, but it equals the next recorded state exactly"})
        elif expected is not None and kind in REPRESENTATIONS.values() and kind != expected:
            found.append({"check": f"command_representation:{name}",
                          "detail": f"declared {command.get('representation')}, measured {kind}"})
    for index in (declaration.get("derived") or {}).get("indices") or []:
        measured = by_dim.get(index) if _is_index(index, len(by_dim)) else None
        if measured is not None and measured["classification"] != "derived_next_state":
            found.append({"check": f"derived_equals_next_state:{measured['name']}",
                          "detail": "declared derived, but max |action[t] - state[t+1]| is "
                                    f"{measured['max_abs_action_minus_next_state']:.3g}"})
    declared = declaration.get("gripper") or {}
    if gripper is not None and _is_number(declared.get("open_value")) and _is_number(declared.get("closed_value")):
        open_value, closed_value = float(declared["open_value"]), float(declared["closed_value"])
        direction = "decreasing" if closed_value < open_value else "increasing"
        if direction != gripper["closing_direction"]:
            found.append({"check": "gripper_closing_direction",
                          "detail": f"declared closing by {direction} values, measured {gripper['closing_direction']}"})
        span = abs(open_value - closed_value)
        for label, value, measured in (("open", open_value, gripper["open_value"]),
                                       ("closed", closed_value, gripper["closed_value"])):
            if span > 0 and abs(value - float(measured)) > GRIPPER_VALUE_TOLERANCE * span:
                found.append({"check": f"gripper_{label}_value",
                              "detail": f"declared {value:.4g}, recordings sit at {float(measured):.4g}"})
    return found


def resolve_action_spec(declaration: Mapping[str, Any], declaration_record: Mapping[str, Any], dims: Sequence[str],
                        actions: Sequence[Mapping[str, Any]], gripper: Mapping[str, Any],
                        integrity_problems: Sequence[str], all_actions: np.ndarray, episodes: int,
                        margin: float) -> Dict[str, Any]:
    """The action specification: the declared semantics, the checks against the recordings, and the normaliser."""
    problems = list(declaration_problems(declaration, len(dims)))
    found = measured_contradictions(declaration, actions, gripper)
    accepted = {str((c or {}).get("check")): str((c or {}).get("reason"))
                for c in declaration.get("accepted_contradictions") or []}
    blocking = [c for c in found if c["check"] not in accepted]
    problems.extend(f"measurement contradicts the declaration ({c['check']}): {c['detail']}" for c in blocking)
    problems.extend(integrity_problems)

    command = declaration.get("command") or {}
    indices = [i for i in command.get("indices") or [] if _is_index(i, len(dims))]
    names = list(command.get("names") or [])
    warnings = [f"{r['name']}: the measurements could not classify it; the declaration stands on its evidence"
                for r in actions if r["dim"] in indices and r["classification"] == "unresolved"]
    declared_gripper = declaration.get("gripper") or {}
    if _is_number(declared_gripper.get("open_value")) and _is_number(declared_gripper.get("closed_value")):
        open_value, closed_value = float(declared_gripper["open_value"]), float(declared_gripper["closed_value"])
        gripper_source = "declared"
    else:
        open_value, closed_value = float(gripper["open_value"]), float(gripper["closed_value"])
        gripper_source = "measured; not declared, so not verified"
        warnings.append("the gripper's open and closed values are measured estimates until action_mapping.yaml "
                        "declares them")
    commands = all_actions[:, indices] if indices and all_actions.size else np.zeros((0, 0))
    lags = [r["best_lag_frames"] for r in actions if r["dim"] in indices]
    return {
        "format": ACTION_SPEC_FORMAT,
        "created": utc_now(),
        "verified": not problems,
        "problems": problems,
        "warnings": warnings,
        "declaration": dict(declaration_record),
        "command_indices": indices,
        "command_names": names if len(names) == len(indices) else [dims[i] for i in indices],
        "representation": command.get("representation"),
        "units": {str(k): str(v) for k, v in (command.get("units") or {}).items()},
        "derived_indices": [i for i in (declaration.get("derived") or {}).get("indices") or []
                            if _is_index(i, len(dims))],
        "derived_rule": (declaration.get("derived") or {}).get("rule"),
        "gripper": {
            "index": declared_gripper.get("action_index", gripper["index"]),
            "state_index": declared_gripper.get("state_index", gripper["index"]),
            "units": declared_gripper.get("units"),
            "open_value": open_value,
            "closed_value": closed_value,
            "closing_direction": "decreasing" if closed_value < open_value else "increasing",
            "source": gripper_source,
        },
        "contradictions": found,
        "accepted_contradictions": [{"check": k, "reason": v} for k, v in accepted.items()],
        "measured": {
            "classifications": {r["name"]: r["classification"] for r in actions},
            "state_action_lag_frames": int(np.median(lags)) if lags else None,
            "gripper": dict(gripper),
        },
        "normalization": {
            "type": "minmax",
            "fit_episodes": "all",
            "episodes": int(episodes),
            "low": [float(v) for v in commands.min(axis=0)] if commands.size else [],
            "high": [float(v) for v in commands.max(axis=0)] if commands.size else [],
            "margin": float(margin),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource

    parser = argparse.ArgumentParser(description="Audit the recorded fields against the declared action mapping.")
    parser.add_argument("--video-check", choices=("none", "metadata", "decode"), default="decode")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    dataset_cfg = configs["dataset"]
    source = RawEpisodeSource(configs)
    fps = source.fps()
    episodes = source.available()
    tables = {episode: source.table(episode) for episode in episodes}
    # table() keeps one episode cached; the audit holds all of them itself.
    dims = list(dataset_cfg["source"]["dims"])

    declaration_path = repo_path(dataset_cfg["action"]["mapping"])
    if not os.path.isfile(declaration_path):
        raise SystemExit(f"no action mapping declaration at {declaration_path}")
    declaration = load_yaml(declaration_path)
    declaration_record = {
        "path": os.path.relpath(declaration_path, repo_path("")).replace(os.sep, "/"),
        "sha256": file_sha256(declaration_path),
        "hash": stable_hash(declaration),
        "version": declaration.get("version"),
        "confirmed": declaration.get("confirmed") is True,
        "confirmed_by": declaration.get("confirmed_by"),
        "confirmed_on": declaration.get("confirmed_on"),
    }
    gripper_state = (declaration.get("gripper") or {}).get("state_index")
    if not _is_index(gripper_state, len(dims)):
        gripper_state = 6

    actions = audit_actions(tables, dims)
    gripper = audit_gripper(tables, gripper_state)
    velocity = audit_velocity_effort(tables, fps)
    timing = audit_timing(tables, source.lengths(), fps)
    videos = audit_videos(source, episodes, args.video_check)
    endings = audit_endings(tables, fps)

    integrity = []
    if not timing["ok"]:
        integrity.append("timestamps or frame indices are irregular")
    if videos.get("checked") != "none" and not videos.get("ok", False):
        integrity.append("videos do not align with the recorded rows")
    all_actions = np.concatenate([tables[e]["action"] for e in episodes])
    spec = resolve_action_spec(declaration, declaration_record, dims, actions, gripper, integrity, all_actions,
                               len(episodes), float(dataset_cfg["action"]["normalization_margin"]))
    spec["velocity_effort"] = {"resolved": velocity.get("resolved", False),
                               "interpretation": velocity.get("interpretation")}
    spec["source_revision"] = source.source_record()["resolved_revision"]

    out_dir = repo_path(dataset_cfg["paths"]["audit"])
    write_json(os.path.join(out_dir, "audit_report.json"), {
        "created": spec["created"], "episodes": len(episodes), "fps": fps,
        "actions": actions, "gripper": gripper, "velocity_effort": velocity,
        "timing": timing, "videos": videos, "endings": endings,
        "declaration": declaration_record, "contradictions": spec["contradictions"], "problems": spec["problems"],
    })
    write_json(os.path.join(out_dir, "action_spec.json"), spec)

    print(f"[audit] {len(episodes)} episodes at {fps} fps")
    print("  measured (evidence for or against the declaration, never proof of it):")
    for r in actions:
        print(f"    {r['name']:13s} {r['classification']:22s} max|a-s'|={r['max_abs_action_minus_next_state']:.2e} "
              f"best lag={r['best_lag_frames']:+d}")
    print(f"    gripper: open~{gripper['open_value']:.3f} closed~{gripper['closed_value']:.3f} "
          f"({gripper['closing_direction']} closes)")
    print(f"  declared: commands {spec['command_names']} as {spec['representation']}, units {spec['units']}")
    print(f"  declaration confirmed: {declaration_record['confirmed']} by {declaration_record['confirmed_by']!r}")
    print(f"  velocity/effort resolved: {velocity.get('resolved')} ({velocity.get('interpretation')})")
    print(f"  timing ok: {timing['ok']}   videos: {videos.get('checked')} ok={videos.get('ok')}")
    print(f"  episodes ending in motion: {len(endings['episodes_ending_in_motion'])}")
    for warning in spec["warnings"]:
        print(f"  warning: {warning}")
    if spec["problems"]:
        print("[audit] not verified:\n  " + "\n  ".join(spec["problems"]))
    print(f"[audit] verified={spec['verified']} -> {os.path.join(out_dir, 'action_spec.json')}")


if __name__ == "__main__":
    main()
