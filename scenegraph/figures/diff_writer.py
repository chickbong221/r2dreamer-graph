"""Output layout for a figure made of one human view and two camera differences.

:mod:`scenegraph.figures.multicamera_writer` writes head, wrist and a graph per
step, which is the shape the ManiSkill tabletop figure needs. This writes a
different figure: the third-person human view of what happened, and -- for the
two robot-mounted cameras -- the *pixel-wise difference between consecutive
frames* rather than the frames themselves. A difference image is the picture of
what moved, which is what a temporal figure is about; the frames it was
computed from are kept beside it so the difference can be recomputed and
checked rather than taken on trust.

Two copies of every difference are written, for two different readers:

``diff/``      the exact ``|f[t] - f[t-1]|`` as uint8. Lossless and
               reproducible: nothing here is scaled, so the true per-pixel
               magnitudes are recoverable from the file.
``diff_vis/``  the same magnitudes multiplied by one episode-wide gain. A raw
               consecutive-frame difference in a mostly static scene is almost
               black on paper, and per-frame normalisation would make every
               panel of a printed strip use a different scale -- so the gain is
               derived once from the whole episode and recorded in the manifest.

Index 0 has no predecessor and therefore no difference: the frame directories
hold ``n`` entries and the difference directories ``n - 1``, starting at 0001.
That asymmetry is deliberate, and the manifest's per-step records name the
files, so a figure script never pairs a difference with the wrong frame.

Same staging discipline as the other two writers, for the same reason: a
half-written episode under ``data/paper_figures`` is indistinguishable from a
finished one.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The two file conventions -- lossless uint8 PNG, and JSON that tolerates
# whatever numpy scalar came in from the simulator -- are already settled for
# figures in this package. Re-deriving them here would let two figures from the
# same repo disagree about what an exported frame looks like.
from .writer import save_json, save_png

FRAME_DIR = "frames"
DIFF_DIR = "diff"
DIFF_VIS_DIR = "diff_vis"
HUMAN_ROLE = "human"
MANIFEST = "episode.json"
# Distinct from the other writers' prefixes so none of the three can stage, or
# delete, another's work.
STAGING_PREFIX = ".staging_diff_"
# uint8 difference magnitudes: |a - b| for two uint8 pixels never exceeds 255,
# so one bin per level covers the histogram exactly and the gain below is
# computed from counts rather than from a sample.
LEVELS = 256


def episode_path(root: "Path | str", name: str) -> Path:
    """Where a committed episode lands.

    Public because the caller needs it before the rollout runs: refusing a name
    collision after two passes of simulation have already been paid for wastes
    both of them.
    """
    return Path(root) / str(name)


def abs_diff(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """``|current - previous|`` per channel, as uint8.

    Promoted to int16 first: subtracting two uint8 arrays wraps, so the
    unpromoted version turns a drop of one level into a difference of 255 and
    the resulting figure is noise.
    """
    return np.abs(
        current.astype(np.int16) - previous.astype(np.int16)
    ).astype(np.uint8)


def gain_from_histogram(
    histogram: np.ndarray, *, percentile: float, max_gain: float
) -> float:
    """One multiplier that makes this episode's differences legible in print.

    The magnitude at ``percentile`` is mapped to full scale, so the amplified
    image saturates only on the tail the percentile excluded. Both guards fire
    on real episodes: a scene where almost nothing moved puts the percentile at
    zero and would ask for an infinite gain, and any gain past ``max_gain`` is
    amplifying sensor noise rather than motion.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 1.0
    fraction = np.cumsum(counts) / total
    target = min(max(float(percentile), 0.0), 100.0) / 100.0
    reference = int(np.searchsorted(fraction, target))
    reference = max(1, min(reference, LEVELS - 1))
    return float(min(float(max_gain), 255.0 / reference))


def amplify(diff: np.ndarray, gain: float, *, invert: bool = False) -> np.ndarray:
    """``diff * gain``, clipped to uint8; optionally on a white ground."""
    scaled = np.clip(np.asarray(diff, dtype=np.float32) * float(gain), 0, 255)
    out = scaled.astype(np.uint8)
    return (255 - out) if invert else out


def load_png(path: "Path | str") -> np.ndarray:
    """Read back a PNG this module wrote, as ``[H, W, 3]`` uint8."""
    from PIL import Image

    with Image.open(str(path)) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


class DiffEpisodeWriter:
    """Accumulates one episode's frames and differences, then commits or drops them.

    ``roles`` names the sensor cameras the differences are about, in the order
    the caller wants them on disk. The human view is not one of them: it is a
    single third-person camera, written whole, and its difference would mostly
    show the arm sweeping past a static room.
    """

    def __init__(
        self,
        root: "Path | str",
        name: str,
        *,
        roles: Sequence[str],
        human_size: Optional[Sequence[int]] = None,
        sensor_size: Optional[Sequence[int]] = None,
        save_human: bool = True,
        save_frames: bool = True,
        max_gain: float = 16.0,
        overwrite: bool = False,
    ):
        # A name carrying a separator would point ``commit``'s replace step at
        # some directory other than a child of ``root``; rejected at
        # construction so the deletion below cannot be aimed by a caller.
        if not name or Path(name).name != name:
            raise ValueError(
                f"episode name must be a single directory name, got {name!r}"
            )
        if not roles:
            raise ValueError("a difference figure needs at least one camera role")
        self.root = Path(root)
        self.name = str(name)
        self.roles = [str(role) for role in roles]
        self.human_size = None if human_size is None else tuple(
            int(v) for v in human_size
        )
        self.sensor_size = None if sensor_size is None else tuple(
            int(v) for v in sensor_size
        )
        self.save_human = bool(save_human)
        # The raw frames are what the differences are computed from, so they are
        # kept by default; a caller that wants only the differences says so, and
        # the manifest it writes records that the frames are absent.
        self.save_frames = bool(save_frames)
        self.max_gain = float(max_gain)
        self.overwrite = bool(overwrite)
        self.staging = self.root / f"{STAGING_PREFIX}{self.name}"
        self.records: List[Dict[str, Any]] = []
        self._index = 0
        self._previous: Dict[str, np.ndarray] = {}
        self._histogram: Dict[str, np.ndarray] = {}

    @property
    def count(self) -> int:
        return self._index

    @property
    def final(self) -> Path:
        return episode_path(self.root, self.name)

    def open(self) -> None:
        """Start (or restart) the staging directory for this episode."""
        self._refuse_existing()
        shutil.rmtree(self.staging, ignore_errors=True)
        subs: List[str] = []
        if self.save_human:
            subs.append(f"{FRAME_DIR}/{HUMAN_ROLE}")
        for role in self.roles:
            if self.save_frames:
                subs.append(f"{FRAME_DIR}/{role}")
            subs.append(f"{DIFF_DIR}/{role}")
            subs.append(f"{DIFF_VIS_DIR}/{role}")
        for sub in subs:
            (self.staging / sub).mkdir(parents=True, exist_ok=True)
        self.records = []
        self._index = 0
        self._previous = {}
        self._histogram = {
            role: np.zeros(LEVELS, dtype=np.int64) for role in self.roles
        }

    def write_step(
        self,
        *,
        step: int,
        sensors: Dict[str, np.ndarray],
        human: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Export one instant: the human view, the sensor frames, the differences.

        Everything for this instant is written in one call under one shared
        stem, so a figure can never pair a human view with the difference of a
        different step.
        """
        stem = f"{self._index:04d}"
        record: Dict[str, Any] = {"index": self._index, "step": int(step)}
        record.update(dict(extra or {}))

        if self.save_human:
            if human is None:
                raise ValueError(
                    "this writer was opened with save_human; pass the human "
                    "render frame, or construct it with save_human=False"
                )
            self._check_frame(HUMAN_ROLE, human, self.human_size)
            path = f"{FRAME_DIR}/{HUMAN_ROLE}/frame_{stem}.png"
            save_png(self.staging / path, human)
            record[HUMAN_ROLE] = path

        for role in self.roles:
            if role not in sensors:
                raise KeyError(
                    f"no frame for camera role {role!r}; have {sorted(sensors)}"
                )
            frame = np.ascontiguousarray(np.asarray(sensors[role])[..., :3])
            self._check_frame(role, frame, self.sensor_size)
            if self.save_frames:
                path = f"{FRAME_DIR}/{role}/frame_{stem}.png"
                save_png(self.staging / path, frame)
                record[role] = path
            previous = self._previous.get(role)
            if previous is not None:
                diff = abs_diff(frame, previous)
                path = f"{DIFF_DIR}/{role}/diff_{stem}.png"
                save_png(self.staging / path, diff)
                record[f"{role}_diff"] = path
                self._histogram[role] += np.bincount(
                    diff.ravel(), minlength=LEVELS
                ).astype(np.int64)
                # Two numbers per difference, so a caption can say how much of
                # the frame moved without anyone re-opening the PNGs.
                record[f"{role}_diff_mean"] = float(diff.mean())
                record[f"{role}_diff_max"] = int(diff.max())
            self._previous[role] = frame

        self.records.append(record)
        self._index += 1
        return self._index - 1

    def write_amplified(
        self, *, percentile: float, gain: float = 0.0, invert: bool = False
    ) -> Dict[str, float]:
        """Fill ``diff_vis`` from the staged raw differences. Returns the gains.

        Run after the last step, because the gain is a property of the whole
        episode: a per-frame scale would make every panel of a printed strip
        mean something different. The raw differences are re-read from staging
        rather than held in memory -- an episode of 500px differences is
        hundreds of megabytes, and they are already on disk.

        ``gain > 0`` overrides the automatic one, for a figure that has to share
        a scale with another episode.
        """
        gains: Dict[str, float] = {}
        for role in self.roles:
            role_gain = (
                float(gain) if gain > 0 else gain_from_histogram(
                    self._histogram[role],
                    percentile=percentile,
                    max_gain=self.max_gain,
                )
            )
            gains[role] = role_gain
            for record in self.records:
                source = record.get(f"{role}_diff")
                if not source:
                    continue
                target = f"{DIFF_VIS_DIR}/{role}/{Path(source).stem}.png"
                save_png(
                    self.staging / target,
                    amplify(
                        load_png(self.staging / source), role_gain, invert=invert
                    ),
                )
                record[f"{role}_diff_vis"] = target
        return gains

    def commit(self, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Publish the staged episode under its real name and return the path."""
        self._refuse_existing()
        payload = dict(metadata or {})
        payload["name"] = self.name
        payload["exported_frames"] = len(self.records)
        payload["exported_diffs"] = max(0, len(self.records) - 1)
        payload["steps"] = self.records
        save_json(self.staging / MANIFEST, payload)
        final = self.final
        if final.exists():
            self._replace(final)
        self.staging.rename(final)
        return final

    def discard(self) -> None:
        """Delete everything this episode staged. Safe to call twice."""
        shutil.rmtree(self.staging, ignore_errors=True)
        self.records = []
        self._index = 0
        self._previous = {}

    # ------------------------------------------------------------- internals
    def _refuse_existing(self) -> None:
        """Stop before an episode nobody asked to replace is replaced.

        ``data/paper_figures`` holds figures that have already been placed in a
        paper; overwriting one silently would change a published picture and
        leave no trace of what it used to be.
        """
        if self.overwrite:
            return
        final = self.final
        if final.exists():
            raise FileExistsError(
                f"{final} already exists; pass --overwrite to replace it, or "
                "write to a different --out"
            )

    def _replace(self, final: Path) -> None:
        """Remove the episode being overwritten, and nothing else.

        The two facts that make this ``rmtree`` safe -- it is a directory, and
        it is the direct child of ``root`` this writer is named for -- are
        checked rather than assumed. Without them a name assembled elsewhere
        could aim the deletion at ``data/paper_figures`` itself.
        """
        if final.parent != self.root or final.name != self.name:
            raise RuntimeError(
                f"refusing to remove {final}: not the episode directory "
                f"{self.name!r} under {self.root}"
            )
        if not final.is_dir():
            raise RuntimeError(f"refusing to remove {final}: not a directory")
        shutil.rmtree(final)

    def _check_frame(
        self, role: str, image: np.ndarray, expected: Optional[Tuple[int, ...]]
    ) -> None:
        """Reject a frame that is not the picture this figure was sized for.

        Nothing here crops or resizes. A frame arriving at another size means
        the env was built with different camera configs, and the failure should
        name that rather than produce a figure whose panels do not line up --
        or, worse, a difference between two images of different sizes.
        """
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{role} frame must be [H, W, 3] RGB, got shape {arr.shape}"
            )
        if expected is not None and arr.shape[:2] != expected:
            raise ValueError(
                f"{role} frame is {arr.shape[:2]}, expected {expected}"
            )
