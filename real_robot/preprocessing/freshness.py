"""Whether an artifact is current, all the way up the chain it was made from.

Each artifact records the digests of the files it read, so a stage can tell
that an input has not changed since. That alone does not make the input
current: unchanged tracks made from an annotation whose prompt has since
changed are stale, and so is geometry measured with an alignment whose own
tracks were made again. Every check here walks the chain to its roots --

    prepared videos  <- source videos, video settings
    annotation       <- graph, frozen bins, prompts and schemas, Gemini settings,
                        prepared videos, validation rules; and valid
    tracks           <- the annotation, tracking settings, source videos
    depth            <- source video, Depth Pro weights, the fixed focal length
                        estimated with those weights
    camera check     <- source video, the reference episode
    alignment        <- its episodes' tracks, depth and camera check, settings
    geometry         <- annotation, tracks, depth, camera check, alignment,
                        settings
    reward scales    <- the annotation and geometry of every episode they were
                        fitted on

-- and returns every reason something is not current, each naming the stage
that fixes it. Stages that build on an artifact (alignment, measurement,
tracking, the reward scales, the dataset build) refuse on any of them; reports
print them as warnings.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from ..common import read_json, stable_hash


def problems_text(problems: Mapping[int, Sequence[str]], limit: int = 40) -> str:
    """``episode N: reason; reason`` lines for a refusal message."""
    lines = [f"episode {episode}: " + "; ".join(reasons) for episode, reasons in sorted(problems.items())]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more episode(s)"]
    return "\n  ".join(lines)


class ArtifactChain:
    """Memoised currency checks over one source. Make a new one after writing artifacts."""

    def __init__(self, configs: Mapping[str, Mapping[str, Any]], source):
        self.configs = configs
        self.source = source
        self._memo: Dict[Tuple[Any, ...], List[str]] = {}
        self._annotator = None
        self._geometry = None

    def _once(self, key: Tuple[Any, ...], compute: Callable[[], List[str]]) -> List[str]:
        if key not in self._memo:
            self._memo[key] = list(dict.fromkeys(compute()))
        return list(self._memo[key])

    @property
    def geometry(self):
        if self._geometry is None:
            from .estimate_geometry import GeometryStage

            self._geometry = GeometryStage(self.configs, mode=self.source.mode, source=self.source)
        return self._geometry

    @property
    def annotator(self):
        if self._annotator is None:
            from .annotate_episode import EpisodeAnnotator

            self._annotator = EpisodeAnnotator(self.configs, mode=self.source.mode, source=self.source)
        return self._annotator

    # ------------------------------------------------------ annotation
    def annotation(self, episode: int) -> List[str]:
        return self._once(("annotation", int(episode)), lambda: self._annotation(int(episode)))

    def _annotation(self, episode: int) -> List[str]:
        from ..graphs.validate import ANNOTATION_FORMAT
        from .artifacts import mismatched_fields
        from .prepare_videos import prepared_status

        path = self.source.annotation_path(episode)
        if not os.path.isfile(path):
            return ["no annotation; run annotate_episode"]
        stored = read_json(path)
        if stored.get("format") != ANNOTATION_FORMAT:
            return ["the annotation was written by an earlier version of this package; run annotate_episode"]
        problems = []
        if stored.get("status") != "valid":
            problems.append(f"the annotation is invalid ({len(stored.get('issues') or [])} issue(s), see {path})")
        prepared, reason = prepared_status(self.source, episode, self.configs["annotation"]["videos"])
        if prepared is None:
            problems.append(f"the videos Gemini watched are not current ({reason}); run prepare_videos, then "
                            "annotate_episode")
            return problems
        try:
            expected = self.annotator.input_identity(episode, prepared=prepared)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"the annotation's inputs cannot be checked: {exc}")
            return problems
        if stored.get("input_hash") != stable_hash(expected):
            fields = mismatched_fields(expected, stored.get("input_identity") or {})
            problems.append("the annotation was made from other inputs (" + (", ".join(fields) or "unknown")
                            + "); run annotate_episode")
        return problems

    # ---------------------------------------------------------- tracks
    def tracks(self, episode: int) -> List[str]:
        return self._once(("tracks", int(episode)), lambda: self._tracks(int(episode)))

    def _tracks(self, episode: int) -> List[str]:
        from .artifacts import stale_reason, stored_inputs
        from .track_objects import tracks_inputs

        problems = [f"annotation: {p}" for p in self.annotation(episode)]
        path = self.source.tracks_path(episode)
        if not os.path.isfile(path):
            return problems + ["no tracks; run track_objects"]
        reason = stale_reason(tracks_inputs(self.source, episode, self.configs["annotation"]["tracking"]),
                              stored_inputs(path))
        if reason:
            problems.append(f"the tracks are stale ({reason}); run track_objects")
        return problems

    # ---------------------------------------------------- depth, camera
    def depth(self, episode: int) -> List[str]:
        return self._once(("depth", int(episode)), lambda: self._depth(int(episode)))

    def _depth(self, episode: int) -> List[str]:
        from ..common import repo_path
        from .artifacts import file_digest, stale_reason, stored_inputs

        stage = self.geometry
        camera_path = stage.camera_geometry_path()
        if not os.path.isfile(camera_path):
            return ["no fixed focal length; run estimate_geometry depth"]
        camera = read_json(camera_path)
        weights = file_digest(repo_path(stage.cfg["checkpoint"]))
        if weights is None:
            return [f"the Depth Pro weights are missing at {stage.cfg['checkpoint']}, so depth cannot be checked"]
        if camera.get("checkpoint") != weights:
            return ["the fixed focal length was estimated with other Depth Pro weights; run estimate_geometry depth"]
        path = stage.depth_path(episode)
        if not os.path.isfile(path):
            return ["no depth; run estimate_geometry depth"]
        reason = stale_reason(stage.depth_inputs(episode, float(camera["focal_px"])), stored_inputs(path))
        return [f"the depth is stale ({reason}); run estimate_geometry depth"] if reason else []

    def camera(self, episode: int) -> List[str]:
        return self._once(("camera", int(episode)), lambda: self._camera(int(episode)))

    def _camera(self, episode: int) -> List[str]:
        fixed, reason = self.geometry.camera_status(episode)
        return [] if fixed else [reason]

    # ------------------------------------------------------- alignment
    def alignment(self) -> List[str]:
        return self._once(("alignment",), self._alignment)

    def _alignment(self) -> List[str]:
        stage = self.geometry
        if not os.path.isfile(stage.alignment_path()):
            return ["no camera alignment; run estimate_geometry align"]
        current, reason = stage.alignment_current()
        if not current:
            return [f"the camera alignment is stale ({reason}); run estimate_geometry align"]
        problems = []
        for episode in read_json(stage.alignment_path())["inputs"]["episodes"]:
            for problem in self.tracks(episode) + self.depth(episode) + self.camera(episode):
                problems.append(f"alignment episode {episode}: {problem}")
        return problems

    # -------------------------------------------------------- geometry
    def measurement_inputs(self, episode: int) -> List[str]:
        """Everything measuring an episode reads, current: camera, tracks (so the annotation), depth, alignment."""
        return self._once(("measurement_inputs", int(episode)), lambda: (
            self.camera(episode) + self.tracks(episode) + self.depth(episode)
            + [f"alignment: {p}" for p in self.alignment()]))

    def geometry_chain(self, episode: int) -> List[str]:
        """The saved geometry, and everything it was measured from, current."""
        return self._once(("geometry", int(episode)), lambda: self._geometry_chain(int(episode)))

    def _geometry_chain(self, episode: int) -> List[str]:
        from .artifacts import stale_reason, stored_inputs

        problems = self.measurement_inputs(episode)
        path = self.source.geometry_path(episode)
        if not os.path.isfile(path):
            return problems + ["no geometry; run estimate_geometry measure"]
        reason = stale_reason(self.geometry.geometry_inputs(episode), stored_inputs(path))
        if reason:
            problems.append(f"the geometry is stale ({reason}); run estimate_geometry measure")
        return problems

    # ---------------------------------------------------------- scales
    def scales(self, scales) -> Dict[int, List[str]]:
        """Per fitted episode, why the reward scales cannot be trusted; empty when they can."""
        from .artifacts import file_digest

        inputs = scales.provenance.get("inputs") or {}
        out: Dict[int, List[str]] = {}
        for episode in scales.provenance.get("episodes", []):
            episode = int(episode)
            recorded = inputs.get(str(episode))
            problems = self.geometry_chain(episode)
            current = {"annotation": file_digest(self.source.annotation_path(episode)),
                       "geometry": file_digest(self.source.geometry_path(episode))}
            if recorded != current:
                problems.append("the reward scales were fitted on another version of this episode's annotation or "
                                "geometry; run rewards.kitchen fit-scales --episodes all --force")
            if problems:
                out[episode] = list(dict.fromkeys(problems))
        return out


def warn_stale(chain: ArtifactChain, episodes: Sequence[int], what: str = "geometry",
               prefix: str = "[stale]") -> Dict[int, List[str]]:
    """Print, without refusing, why each episode's artifacts are not current. Returns the problems."""
    check = {"annotation": chain.annotation, "tracks": chain.tracks, "geometry": chain.geometry_chain}[what]
    stale = {int(e): check(e) for e in episodes}
    stale = {e: p for e, p in stale.items() if p}
    for episode, problems in sorted(stale.items()):
        print(f"{prefix} episode {episode}: this report reads artifacts that are not current: " + "; ".join(problems),
              flush=True)
    return stale
