"""Dense environment reward for banana-in-pot, lid-on-pot.

Six stages, each a continuous closeness inside a discrete milestone:

====  ================  ==========================================  ==============================
k     stage             within-stage score q                        leaves the stage when
====  ================  ==========================================  ==============================
0     reach banana      C(gripper -> banana grasp region)           banana grasped
1     transport banana  C(banana -> pot entry)                      banana over the pot opening
2     release banana    mean(C(placement), released, settled)       banana in pot, released, settled
3     reach lid         C(gripper -> lid handle)                    lid grasped
4     seat lid          mean(C(lid lateral), C(lid height))         lid seated on the pot
5     release lid       mean(seating, released, settled)            task complete
====  ================  ==========================================  ==============================

with ``C(d; s) = 1 - tanh(d / s)``, ``S_t = (k_t + q_t) / 6`` and

    r_t = +1            on the transition that arrives at verified completion
    r_t = S_{t+1} - 1   otherwise.

What each input is for:

* **Gemini milestone labels** decide the discrete facts: grasp holds, the pot
  contains the banana, the pot supports the lid.
* **The recorded gripper** verifies them: a grasp needs the gripper closing,
  a release needs it to have opened.
* **Saved geometry** supplies every continuous term.

The stage is recomputed from the current frame's facts on every frame. It is
never the maximum reached so far, so a dropped banana falls back to stage 0 and
a lifted lid back to stage 3. The only memory is what verification itself needs
-- whether the gripper has opened since the object was last held, and whether
the object has settled since -- and it resets the moment the object is grasped
again, leaves the pot, or (for settling) is measured moving. An object the lid
or the pot's walls hide after it has settled therefore stays settled; one that
was never seen to settle is settled only as ``settle.unknown_speed`` says.

Nothing depends on the episode's length or on how far through it a frame is:
the reward at ``t`` reads frames up to ``t + 1`` only, which the prefix check
asserts.

Missing geometry is explicit, never a quiet pass:

* ``lid_seated.unknown_geometry`` says what an unknown lid position means --
  ``not_seated`` (the default: seating needs geometry that agrees) or
  ``defer_to_label`` (Gemini's support label alone decides when geometry is
  unknown);
* ``settle.unknown_speed`` says what an unmeasured speed means before an
  object has settled -- ``not_settled`` (the default) or ``defer_to_label``;
* a frame whose stage score needs a distance, or a speed to judge settling,
  that is unknown is flagged ``q_imputed``, and :func:`fallback_report` counts
  those frames overall, by stage and as the longest unbroken run.

:func:`reward_checks` marks the checks whose failure makes an episode unusable
for training as ``critical``; ``build_dataset`` refuses such an episode rather
than packing a reward it cannot trust.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

N_STAGES = 6
STAGE_NAMES = ("reach_banana", "transport_banana", "release_banana", "reach_lid", "seat_lid", "release_lid")
SCALE_NAMES = ("reach_banana", "transport_banana", "placement", "reach_lid", "lid_lateral", "lid_height")
UNKNOWN_GEOMETRY = ("not_seated", "defer_to_label")
UNKNOWN_SPEED = ("not_settled", "defer_to_label")
# A failure of any of these makes an episode's reward unusable for training.
CRITICAL_CHECKS = (
    "transport_requires_grasp", "empty_pot_is_not_completion", "no_dependence_on_episode_length",
    "completion_not_before_observed_event", "truncation_is_not_termination", "failure_is_not_a_shortcut",
    "observed_outcome_agrees", "distance_fallbacks_within_limit",
)
GEOMETRY_KEYS = (
    "d_gripper_banana", "d_banana_pot_entry", "banana_placement_error",
    "d_gripper_lid_handle", "lid_lateral_error", "lid_height_above_rim",
    "banana_speed", "lid_speed",
)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class RewardInputs:
    """Per-frame inputs for one episode. Geometry is metres, NaN where unknown."""

    d_gripper_banana: np.ndarray
    d_banana_pot_entry: np.ndarray
    banana_placement_error: np.ndarray
    d_gripper_lid_handle: np.ndarray
    lid_lateral_error: np.ndarray
    lid_height_above_rim: np.ndarray
    banana_speed: np.ndarray
    lid_speed: np.ndarray
    banana_grasp_label: np.ndarray
    lid_grasp_label: np.ndarray
    banana_in_pot_label: np.ndarray
    lid_on_pot_label: np.ndarray
    gripper_closure: np.ndarray
    # Gemini's own completion frame, or -1. Completion is never marked earlier.
    observed_completion_frame: int = -1
    observed_success: bool = False
    failure_frame: int = -1

    @property
    def n_frames(self) -> int:
        return int(len(self.gripper_closure))

    def prefix(self, length: int) -> "RewardInputs":
        values = {}
        for name, value in asdict(self).items():
            values[name] = value[:length] if isinstance(value, np.ndarray) else value
        if values["observed_completion_frame"] >= length:
            values["observed_completion_frame"] = -1
        if values["failure_frame"] >= length:
            values["failure_frame"] = -1
        return RewardInputs(**values)


@dataclass
class RewardScales:
    reach_banana: float
    transport_banana: float
    placement: float
    reach_lid: float
    lid_lateral: float
    lid_height: float
    lid_seated_offset: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_defaults(cls, cfg: Mapping[str, Any]) -> "RewardScales":
        defaults = cfg["scales"]["defaults_m"]
        return cls(**{name: float(defaults[name]) for name in SCALE_NAMES},
                   lid_seated_offset=float(cfg["scales"].get("lid_seated_offset_m", 0.0)),
                   provenance={"source": "defaults"})

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RewardScales":
        return cls(**{name: float(data[name]) for name in SCALE_NAMES},
                   lid_seated_offset=float(data.get("lid_seated_offset", 0.0)),
                   provenance=dict(data.get("provenance", {})))

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def identity(self) -> Dict[str, float]:
        out = {name: round(float(getattr(self, name)), 6) for name in SCALE_NAMES}
        out["lid_seated_offset"] = round(float(self.lid_seated_offset), 6)
        return out


@dataclass
class RewardResult:
    stage: np.ndarray
    q: np.ndarray
    staged_score: np.ndarray        # k + q, the raw ManiSkill-style staged score in [0, 6)
    S: np.ndarray                   # (k + q) / 6
    completion_frame: int
    failure_frame: int
    reward: np.ndarray              # transition t -> t + 1; NaN where the transition is invalid
    done: np.ndarray                # true task termination on transition t -> t + 1
    reward_valid: np.ndarray        # the transition exists and precedes termination
    terms: Dict[str, np.ndarray]
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def closeness(distance: np.ndarray, scale: float) -> np.ndarray:
    """``1 - tanh(d / s)``; NaN stays NaN so a missing distance is never 'far'."""
    distance = np.asarray(distance, dtype=np.float64)
    return 1.0 - np.tanh(np.maximum(distance, 0.0) / float(scale))


def trailing_all(mask: np.ndarray, frames: int) -> np.ndarray:
    """True where the last ``frames`` values up to and including ``t`` all hold."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    run = 0
    for t, value in enumerate(mask):
        run = run + 1 if value else 0
        out[t] = run >= frames
    return out


def gripper_closure(values: np.ndarray, open_value: float, closed_value: float) -> np.ndarray:
    """0 fully open, 1 fully closed, whichever direction the gripper counts."""
    span = float(open_value) - float(closed_value)
    if abs(span) < 1e-9:
        raise ValueError("gripper open and closed values coincide")
    return np.clip((float(open_value) - np.asarray(values, dtype=np.float64)) / span, 0.0, 1.0)


def compute_rewards(inputs: RewardInputs, scales: RewardScales, cfg: Mapping[str, Any]) -> RewardResult:
    n = inputs.n_frames
    eps = float(cfg["closeness_eps"])
    closing = inputs.gripper_closure >= float(cfg["gripper"]["closing_fraction"])
    opened = inputs.gripper_closure <= float(cfg["gripper"]["released_fraction"])
    settle_frames = int(cfg["settle"]["frames"])
    speed_limit = float(cfg["settle"]["speed_m_per_s"])
    unknown_speed = str(cfg["settle"]["unknown_speed"])
    if unknown_speed not in UNKNOWN_SPEED:
        raise ValueError(f"settle.unknown_speed must be one of {UNKNOWN_SPEED}, got {unknown_speed!r}")
    banana_speed = np.asarray(inputs.banana_speed, dtype=np.float64)
    lid_speed = np.asarray(inputs.lid_speed, dtype=np.float64)
    banana_speed_known, lid_speed_known = np.isfinite(banana_speed), np.isfinite(lid_speed)
    with np.errstate(invalid="ignore"):
        banana_moving = banana_speed_known & (banana_speed >= speed_limit)
        lid_moving = lid_speed_known & (lid_speed >= speed_limit)
        banana_slow = banana_speed_known & (banana_speed < speed_limit)
        lid_slow = lid_speed_known & (lid_speed < speed_limit)
    if unknown_speed == "defer_to_label":
        banana_slow = banana_slow | ~banana_speed_known
        lid_slow = lid_slow | ~lid_speed_known
    banana_still = trailing_all(banana_slow, settle_frames)
    lid_still = trailing_all(lid_slow, settle_frames)
    banana_settled = np.zeros(n, dtype=bool)
    lid_settled = np.zeros(n, dtype=bool)

    height_error = np.asarray(inputs.lid_height_above_rim, dtype=np.float64) - scales.lid_seated_offset
    lateral = np.asarray(inputs.lid_lateral_error, dtype=np.float64)
    geometry_known = np.isfinite(height_error) & np.isfinite(lateral)
    with np.errstate(invalid="ignore"):
        geometry_ok = (lateral <= float(cfg["lid_seated"]["lateral_m"])) & (
            np.abs(height_error) <= float(cfg["lid_seated"]["height_m"]))
    unknown_geometry = str(cfg["lid_seated"]["unknown_geometry"])
    if unknown_geometry not in UNKNOWN_GEOMETRY:
        raise ValueError(f"lid_seated.unknown_geometry must be one of {UNKNOWN_GEOMETRY}, got {unknown_geometry!r}")
    lid_seated = inputs.lid_on_pot_label.astype(bool)
    if unknown_geometry == "not_seated":
        # Seating needs known geometry, and the geometry has to agree.
        lid_seated = lid_seated & geometry_known & geometry_ok
    else:
        # Unknown geometry leaves the decision to the label; known geometry must agree.
        lid_seated = lid_seated & (geometry_ok | ~geometry_known)

    banana_grasped = inputs.banana_grasp_label.astype(bool) & closing
    lid_grasped = inputs.lid_grasp_label.astype(bool) & closing

    c_reach_banana = closeness(inputs.d_gripper_banana, scales.reach_banana)
    c_transport = closeness(inputs.d_banana_pot_entry, scales.transport_banana)
    c_place = closeness(inputs.banana_placement_error, scales.placement)
    c_reach_lid = closeness(inputs.d_gripper_lid_handle, scales.reach_lid)
    c_lateral = closeness(lateral, scales.lid_lateral)
    c_height = closeness(np.abs(height_error), scales.lid_height)

    stage = np.zeros(n, dtype=np.int64)
    q = np.zeros(n, dtype=np.float64)
    imputed = np.zeros(n, dtype=bool)
    banana_release = np.zeros(n, dtype=bool)
    lid_release = np.zeros(n, dtype=bool)
    banana_placed = np.zeros(n, dtype=bool)
    complete_now = np.zeros(n, dtype=bool)
    placement_region = float(cfg["placement_region_m"])

    released_b = False
    released_l = False
    rest_b = False
    rest_l = False
    previous_q = None
    previous_stage = None
    for t in range(n):
        in_pot = bool(inputs.banana_in_pot_label[t])
        if banana_grasped[t] or not in_pot:
            released_b = False
        if in_pot and opened[t] and not inputs.banana_grasp_label[t]:
            released_b = True
        if lid_grasped[t] or not lid_seated[t]:
            released_l = False
        if lid_seated[t] and opened[t] and not inputs.lid_grasp_label[t]:
            released_l = True
        banana_release[t], lid_release[t] = released_b, released_l
        # Settled once still for the window while in (on) the pot and unheld; it
        # holds until grasped, out of (off) the pot, or measured moving -- not
        # merely because the object is hidden.
        if not in_pot or inputs.banana_grasp_label[t] or banana_moving[t]:
            rest_b = False
        elif banana_still[t]:
            rest_b = True
        if not lid_seated[t] or inputs.lid_grasp_label[t] or lid_moving[t]:
            rest_l = False
        elif lid_still[t]:
            rest_l = True
        banana_settled[t], lid_settled[t] = rest_b, rest_l
        placed = in_pot and released_b and banana_settled[t] and not inputs.banana_grasp_label[t]
        banana_placed[t] = placed

        if not in_pot:
            if banana_grasped[t]:
                entry = inputs.d_banana_pot_entry[t]
                k = 2 if (np.isfinite(entry) and entry <= placement_region) else 1
            else:
                k = 0
        elif not placed:
            k = 2
        elif lid_seated[t]:
            k = 5
        elif lid_grasped[t]:
            k = 4
        else:
            k = 3

        settle_unknown = False
        if k == 0:
            parts = [c_reach_banana[t]]
        elif k == 1:
            parts = [c_transport[t]]
        elif k == 2:
            parts = [c_place[t], float(released_b), float(banana_settled[t])]
            # Only an unheld banana in the pot can settle; otherwise the term is 0 whatever the speed.
            settle_unknown = (in_pot and not inputs.banana_grasp_label[t] and not banana_settled[t]
                              and not banana_speed_known[t])
        elif k == 3:
            parts = [c_reach_lid[t]]
        elif k == 4:
            parts = [c_lateral[t], c_height[t]]
        else:
            seating = np.nanmean([c_lateral[t], c_height[t]]) if np.isfinite([c_lateral[t], c_height[t]]).any() else np.nan
            parts = [seating, float(released_l), float(lid_settled[t])]
            settle_unknown = not inputs.lid_grasp_label[t] and not lid_settled[t] and not lid_speed_known[t]

        finite = bool(np.all(np.isfinite(parts)))
        if finite:
            value = float(np.mean(parts))
        elif previous_stage == k and previous_q is not None:
            # A missing distance holds the last score within the same stage
            # visit, and otherwise scores the known parts with the unknown one
            # at zero. Either way the frame is flagged, never silently guessed.
            value = previous_q
        else:
            value = float(np.mean([p if np.isfinite(p) else 0.0 for p in parts]))
        # A settling term that could not be judged, for want of a speed, is flagged too.
        imputed[t] = not finite or settle_unknown
        value = min(max(value, 0.0), 1.0 - eps)
        stage[t], q[t] = k, value
        previous_stage, previous_q = k, value
        complete_now[t] = placed and bool(lid_seated[t]) and released_l and bool(lid_settled[t])

    completion = -1
    notes: List[str] = []
    verified = np.flatnonzero(complete_now)
    if verified.size:
        if not inputs.observed_success:
            notes.append(
                f"geometry and labels verify completion at frame {int(verified[0])}, but Gemini's "
                "outcome is not a success; no completion is marked"
            )
        else:
            # Not before the observed event: the later of the two frames that
            # each independently say the task is done.
            candidates = verified[verified >= max(int(inputs.observed_completion_frame), 0)]
            if candidates.size:
                completion = int(candidates[0])
            else:
                notes.append(
                    f"verification holds only before Gemini's completion frame "
                    f"{inputs.observed_completion_frame}; no completion is marked"
                )
    elif inputs.observed_success:
        notes.append(
            f"Gemini reports success at frame {inputs.observed_completion_frame}, but completion is "
            "never verified by labels, gripper and settling; no completion is marked"
        )

    failure = -1
    if bool(cfg.get("failure_termination", False)) and inputs.failure_frame >= 0 and completion < 0:
        failure = int(inputs.failure_frame)
        check_failure_penalty(cfg)

    staged = stage.astype(np.float64) + q
    S = staged / N_STAGES
    reward = np.full(n, np.nan, dtype=np.float64)
    done = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    terminal = completion if completion >= 0 else failure
    for t in range(max(n - 1, 0)):
        arrival = t + 1
        if terminal >= 0 and arrival > terminal:
            break
        valid[t] = True
        if completion >= 0 and arrival == completion:
            reward[t] = float(cfg["completion_reward"])
            done[t] = True
        elif failure >= 0 and arrival == failure:
            reward[t] = float(cfg["failure_penalty"])
            done[t] = True
        else:
            reward[t] = S[arrival] - 1.0

    terms = {
        "closeness_reach_banana": c_reach_banana,
        "closeness_transport_banana": c_transport,
        "closeness_placement": c_place,
        "closeness_reach_lid": c_reach_lid,
        "closeness_lid_lateral": c_lateral,
        "closeness_lid_height": c_height,
        "banana_grasped": banana_grasped,
        "lid_grasped": lid_grasped,
        "banana_release_verified": banana_release,
        "banana_settled": banana_settled,
        "banana_placed": banana_placed,
        "lid_seated": lid_seated,
        "lid_geometry_known": geometry_known,
        "lid_release_verified": lid_release,
        "lid_settled": lid_settled,
        "gripper_closure": np.asarray(inputs.gripper_closure, dtype=np.float64),
        "q_imputed": imputed,
    }
    return RewardResult(stage=stage, q=q, staged_score=staged, S=S, completion_frame=completion,
                        failure_frame=failure, reward=reward, done=done, reward_valid=valid,
                        terms=terms, notes=notes)


def check_failure_penalty(cfg: Mapping[str, Any]) -> None:
    """Refuse a failure penalty that makes ending an episode attractive.

    Every non-terminal reward is at least -1, so continuing forever returns at
    least ``-1 / (1 - gamma)``. A failure paying more than that would be a
    better outcome than trying.
    """
    gamma = float(cfg["gamma"])
    floor = -1.0 / (1.0 - gamma)
    if float(cfg["failure_penalty"]) > floor + 1e-9:
        raise ValueError(
            f"failure_penalty={cfg['failure_penalty']} exceeds -1/(1-gamma)={floor:.3f}: "
            "terminating on failure would beat continuing"
        )


# --------------------------------------------------------------------------- #
# Scales
# --------------------------------------------------------------------------- #
def stage_entry_values(inputs: RewardInputs, result: RewardResult) -> Dict[str, List[float]]:
    """Each stage's distance on the first frame of every visit to that stage."""
    stage = result.stage
    entries = {name: [] for name in SCALE_NAMES}
    sources = {
        0: [("reach_banana", inputs.d_gripper_banana)],
        1: [("transport_banana", inputs.d_banana_pot_entry)],
        2: [("placement", inputs.banana_placement_error)],
        3: [("reach_lid", inputs.d_gripper_lid_handle)],
        4: [("lid_lateral", inputs.lid_lateral_error), ("lid_height", None)],
    }
    height = np.abs(np.asarray(inputs.lid_height_above_rim, dtype=np.float64))
    for t in range(len(stage)):
        if t > 0 and stage[t] == stage[t - 1]:
            continue
        for name, series in sources.get(int(stage[t]), []):
            value = height[t] if series is None else float(series[t])
            if np.isfinite(value):
                entries[name].append(float(value))
    return entries


def fit_lid_offset(inputs_list: Sequence[RewardInputs]) -> Tuple[float, int]:
    """Median lid height above the rim on frames labelled seated and not held."""
    values: List[float] = []
    for inputs in inputs_list:
        mask = inputs.lid_on_pot_label.astype(bool) & ~inputs.lid_grasp_label.astype(bool)
        series = np.asarray(inputs.lid_height_above_rim, dtype=np.float64)[mask]
        values.extend(float(v) for v in series if np.isfinite(v))
    if not values:
        return 0.0, 0
    return float(np.median(values)), len(values)


def fallback_report(result: RewardResult) -> Dict[str, Any]:
    """How much of the reward trace scores a stage without the distance it needs.

    Counted over the frames that belong to the episode for learning -- up to
    termination, if there is one.
    """
    imputed = np.asarray(result.terms["q_imputed"], dtype=bool)
    terminal = result.completion_frame if result.completion_frame >= 0 else result.failure_frame
    span = imputed[: terminal + 1] if terminal >= 0 else imputed
    stage = result.stage[: len(span)]
    longest = run = 0
    for value in span:
        run = run + 1 if value else 0
        longest = max(longest, run)
    by_stage = {}
    for k in range(N_STAGES):
        frames = stage == k
        if frames.any():
            by_stage[STAGE_NAMES[k]] = float(np.mean(span[frames]))
    return {"frames": int(len(span)), "imputed_frames": int(span.sum()),
            "imputed_fraction": float(span.mean()) if len(span) else 0.0,
            "longest_imputed_run": int(longest), "imputed_fraction_by_stage": by_stage}


def fit_scales(inputs_list: Sequence[RewardInputs], cfg: Mapping[str, Any],
               episodes: Sequence[int], inputs_identity: Optional[Mapping[str, Any]] = None) -> RewardScales:
    """Scales so each stage's median entry distance scores the target closeness.

    Stages do not depend on the scales, only on labels, gripper and the lid
    offset, so one pass with any scales finds every entry frame.
    """
    offset, offset_samples = fit_lid_offset(inputs_list)
    provisional = RewardScales.from_defaults(cfg)
    provisional.lid_seated_offset = offset
    pooled = {name: [] for name in SCALE_NAMES}
    for inputs in inputs_list:
        result = compute_rewards(inputs, provisional, cfg)
        for name, values in stage_entry_values(inputs, result).items():
            pooled[name].extend(values)
    target = float(cfg["scales"]["target_entry_closeness"])
    if not 0.0 < target < 1.0:
        raise ValueError("scales.target_entry_closeness must lie in (0, 1)")
    divisor = math.atanh(1.0 - target)
    fitted: Dict[str, float] = {}
    samples: Dict[str, int] = {}
    minimum = int(cfg["scales"]["min_samples"])
    for name in SCALE_NAMES:
        values = [v for v in pooled[name] if v > 0]
        samples[name] = len(values)
        if len(values) >= minimum:
            fitted[name] = max(float(np.median(values)) / divisor, 1e-3)
        else:
            fitted[name] = float(cfg["scales"]["defaults_m"][name])
    return RewardScales(**fitted, lid_seated_offset=offset, provenance={
        "source": "fitted",
        "episodes": [int(e) for e in episodes],
        "samples": samples,
        "lid_offset_samples": offset_samples,
        "target_entry_closeness": target,
        "reward_version": cfg["version"],
        # The annotation and geometry each episode's inputs were read from.
        "inputs": dict(inputs_identity or {}),
    })


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def reward_checks(inputs: RewardInputs, result: RewardResult, scales: RewardScales,
                  cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The behaviours the reward must have, measured on one episode.

    ``passed`` is None for a measurement that informs rather than decides.
    """
    checks: List[Dict[str, Any]] = []

    def record(name: str, passed: Optional[bool], detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail, "critical": name in CRITICAL_CHECKS})

    stage, S, terms = result.stage, result.S, result.terms
    n = len(stage)
    # Approach, transport and seating are pure closeness terms. Release stages
    # step by a third when release or settling is verified, which is a
    # milestone and not a lack of smoothness, so they are not measured here.
    continuous = {0, 1, 3, 4}
    jumps = [abs(S[t] - S[t - 1]) for t in range(1, n)
             if stage[t] == stage[t - 1] and int(stage[t]) in continuous
             and not terms["q_imputed"][t] and not terms["q_imputed"][t - 1]]
    worst = max(jumps) if jumps else 0.0
    record("smooth_during_approach_and_transport", worst <= 0.05,
           f"largest frame-to-frame change in S inside a continuous stage: {worst:.4f} (limit 0.05)")

    transport = (stage == 1) | ((stage == 2) & ~inputs.banana_in_pot_label.astype(bool))
    ungrasped = int(np.sum(transport & ~terms["banana_grasped"]))
    seat = stage == 4
    unheld_lid = int(np.sum(seat & ~terms["lid_grasped"]))
    record("transport_requires_grasp", ungrasped == 0 and unheld_lid == 0,
           f"{ungrasped} transport frame(s) without a verified banana grasp, "
           f"{unheld_lid} seating frame(s) without a verified lid grasp")

    drops = [int(t) for t in range(1, n) if stage[t] < stage[t - 1]]
    record("regressions", None,
           f"stage decreases at frames {drops[:20]}" if drops else "no stage decreases")

    empty = RewardInputs(**{**asdict(inputs), "banana_in_pot_label": np.zeros(n, dtype=bool)})
    empty_result = compute_rewards(empty, scales, cfg)
    record("empty_pot_is_not_completion", empty_result.completion_frame < 0,
           "with every contain label removed, completion frame is "
           f"{empty_result.completion_frame}")

    half = n // 2
    if half >= 2:
        prefix = compute_rewards(inputs.prefix(half), scales, cfg)
        same_stage = np.array_equal(prefix.stage, stage[:half])
        same_S = np.allclose(prefix.S, S[:half])
        compare = slice(0, half - 1)
        mask = prefix.reward_valid[compare] & result.reward_valid[compare]
        same_reward = np.allclose(prefix.reward[compare][mask], result.reward[compare][mask])
        record("no_dependence_on_episode_length", bool(same_stage and same_S and same_reward),
               f"first {half} frames recomputed alone: stage equal={same_stage}, "
               f"S equal={same_S}, rewards equal={same_reward}")

    if result.completion_frame >= 0 and inputs.observed_completion_frame >= 0:
        record("completion_not_before_observed_event",
               result.completion_frame >= inputs.observed_completion_frame,
               f"completion {result.completion_frame}, Gemini completion {inputs.observed_completion_frame}")
    else:
        record("completion_not_before_observed_event", None,
               f"completion {result.completion_frame}, Gemini completion {inputs.observed_completion_frame}")

    if result.completion_frame < 0 and result.failure_frame < 0:
        record("truncation_is_not_termination", not result.done.any(),
               f"no completion; {int(result.done.sum())} terminal transition(s)")
    else:
        record("truncation_is_not_termination", int(result.done.sum()) == 1,
               f"terminal at frame {max(result.completion_frame, result.failure_frame)}; "
               f"{int(result.done.sum())} terminal transition(s)")

    if bool(cfg.get("failure_termination", False)):
        try:
            check_failure_penalty(cfg)
            record("failure_is_not_a_shortcut", True, "failure_penalty <= -1/(1-gamma)")
        except ValueError as exc:
            record("failure_is_not_a_shortcut", False, str(exc))
    else:
        record("failure_is_not_a_shortcut", True, "failure termination is disabled")

    verified = result.completion_frame >= 0
    record("observed_outcome_agrees", bool(inputs.observed_success) == verified,
           f"Gemini reports success={bool(inputs.observed_success)} (frame {inputs.observed_completion_frame}); "
           f"labels, gripper and settling verify completion at frame {result.completion_frame}"
           + ("; " + "; ".join(result.notes) if result.notes and bool(inputs.observed_success) != verified else ""))

    fallbacks = fallback_report(result)
    limits = cfg["fallbacks"]
    within = (fallbacks["imputed_fraction"] <= float(limits["max_fraction"])
              and fallbacks["longest_imputed_run"] <= int(limits["max_run_frames"]))
    record("distance_fallbacks_within_limit", within,
           f"{fallbacks['imputed_frames']} of {fallbacks['frames']} frames ({fallbacks['imputed_fraction']:.1%}) "
           f"score a stage without its distance, longest run {fallbacks['longest_imputed_run']} frames "
           f"(limits {float(limits['max_fraction']):.0%} and {int(limits['max_run_frames'])} frames); by stage "
           + ", ".join(f"{k} {v:.0%}" for k, v in fallbacks["imputed_fraction_by_stage"].items()))
    return checks


def critical_failures(checks: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [c for c in checks if c.get("critical") and c.get("passed") is False]


# --------------------------------------------------------------------------- #
# Glue from saved artifacts
# --------------------------------------------------------------------------- #
def inputs_from_artifacts(spec, annotation, geometry: Mapping[str, np.ndarray],
                          gripper_values: np.ndarray, gripper: Mapping[str, Any]) -> RewardInputs:
    """Reward inputs from a validated annotation, saved geometry and the state."""
    n = annotation.n_frames
    for key in GEOMETRY_KEYS:
        if len(geometry[key]) != n:
            raise ValueError(f"geometry {key} covers {len(geometry[key])} frames, annotation {n}")
    if len(gripper_values) != n:
        raise ValueError(f"gripper covers {len(gripper_values)} frames, annotation {n}")
    outcome = annotation.outcome
    return RewardInputs(
        **{key: np.asarray(geometry[key], dtype=np.float64) for key in GEOMETRY_KEYS},
        banana_grasp_label=annotation.holds(spec, "grasp", "ee", "banana"),
        lid_grasp_label=annotation.holds(spec, "grasp", "ee", "lid"),
        banana_in_pot_label=annotation.held_by(spec, "contain", "pot", "banana"),
        lid_on_pot_label=annotation.held_by(spec, "support", "pot", "lid"),
        gripper_closure=gripper_closure(gripper_values, gripper["open_value"], gripper["closed_value"]),
        observed_completion_frame=int(outcome.get("completion_frame", -1)),
        observed_success=bool(outcome.get("success", False)),
        failure_frame=int(outcome.get("failure_frame", -1)) if "failure_frame" in outcome else -1,
    )


def load_scales(cfg: Mapping[str, Any], required: bool = True) -> RewardScales:
    from ..common import read_json, repo_path
    import os

    path = repo_path(cfg["scales"]["path"])
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(
                f"no fitted reward scales at {path}. Run "
                "`python -m real_robot.rewards.kitchen fit-scales --episodes all` first."
            )
        return RewardScales.from_defaults(cfg)
    return RewardScales.from_json(read_json(path))


def reward_identity(cfg: Mapping[str, Any], scales: RewardScales) -> Dict[str, Any]:
    from ..common import stable_hash

    settings = {key: value for key, value in cfg.items() if key != "scales"}
    return {"version": cfg["version"], "settings": stable_hash(settings), "scales": scales.identity()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..common import add_config_arguments, load_configs, repo_path, write_json
    from ..data.episode_dataset import RawEpisodeSource

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-scales", help="fit and freeze scales (on every training episode, for training)")
    fit.add_argument("--episodes", default="all",
                     help="all for the scales training uses; a smaller set only for the pilot plots")
    fit.add_argument("--mode", default=None, help="annotation mode (default: annotation.yaml)")
    fit.add_argument("--force", action="store_true", help="replace an existing scales file")
    add_config_arguments(fit)
    args = parser.parse_args(argv)
    from ..preprocessing.artifacts import file_digest
    from ..preprocessing.freshness import ArtifactChain, problems_text

    configs = load_configs(["dataset", "annotation", "graph", "reward"], args.overrides)
    source = RawEpisodeSource(configs, mode=args.mode)
    episodes = source.select(args.episodes)
    path = repo_path(configs["reward"]["scales"]["path"])
    import os
    if os.path.isfile(path) and not args.force:
        raise SystemExit(f"{path} exists; scales are frozen. Pass --force to refit.")
    chain = ArtifactChain(configs, source)
    stale = {e: chain.geometry_chain(e) for e in episodes}
    stale = {e: p for e, p in stale.items() if p}
    if stale:
        raise SystemExit("[reward] scales are fitted only on current annotations and geometry:\n  "
                         + problems_text(stale))
    inputs_list = []
    identity = {}
    for episode in episodes:
        inputs_list.append(source.reward_inputs(episode))
        identity[str(int(episode))] = {"annotation": file_digest(source.annotation_path(episode)),
                                       "geometry": file_digest(source.geometry_path(episode))}
    scales = fit_scales(inputs_list, configs["reward"], episodes, identity)
    write_json(path, scales.to_json())
    covered = sorted(episodes) == sorted(source.available())
    print(f"[reward] fitted on {len(episodes)} episodes -> {path}")
    if not covered:
        print(f"[reward] these scales cover {len(episodes)} of {len(source.available())} episodes: fine for "
              "inspecting the pilot, refused by build_dataset. Refit with --episodes all --force before building.")
    for name in SCALE_NAMES:
        print(f"  {name:18s} s={getattr(scales, name):.4f} m  "
              f"({scales.provenance['samples'][name]} entries)")
    print(f"  lid_seated_offset  {scales.lid_seated_offset:+.4f} m "
          f"({scales.provenance['lid_offset_samples']} frames)")


if __name__ == "__main__":
    main()
