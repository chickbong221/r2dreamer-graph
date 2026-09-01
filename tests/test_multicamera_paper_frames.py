"""Two-camera paper export: the camera contract, the layout, the isolation.

No simulator. The observation is a dict of numpy arrays shaped the way
ManiSkill shapes one, and the graph source is a stub -- what can go wrong here
is wiring (the wrist camera quietly feeding the graph), pairing (a head frame
committed beside another step's wrist frame), file layout (a failed attempt left
on disk, or a published figure overwritten), and isolation (this exporter
reaching into the single-view one).
"""

import ast
import json
import pathlib
import re
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from scenegraph.core.mask_extractor import extract_camera_obs
from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.figures.graph_source import FigureGraphSource
from scenegraph.figures.multicamera_writer import (
    MulticameraEpisodeWriter, STAGING_PREFIX, episode_path,
)
from scenegraph.figures.writer import EpisodeWriter
from scenegraph.tools.render_multicamera_paper_frames import (
    DEFAULT_HEAD_CAMERA, DEFAULT_WRIST_CAMERA, MulticameraSession,
    PAPER_SENSOR_SIZE, parse_args, preflight,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEAD, WRIST = DEFAULT_HEAD_CAMERA, DEFAULT_WRIST_CAMERA
# Small enough that a test can render a real matplotlib diagram per step; the
# one case that cares about the printed resolution asks for the real 500px.
H = W = 16


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
def _camera(height=H, width=W, *, value=10, segmentation=True):
    """One camera's entry in ``sensor_data``, batched the way ManiSkill is."""
    entry = {"rgb": np.full((1, height, width, 3), value, dtype=np.uint8)}
    if segmentation:
        entry["segmentation"] = np.zeros((1, height, width, 1), dtype=np.int32)
    return entry


def _obs(**cameras):
    return {"sensor_data": dict(cameras)}


def _default_obs(height=H, width=W):
    """Head and wrist as the real env returns them, with different pixels.

    The wrist entry deliberately carries segmentation here: a preflight that
    only passed because the wrist was malformed would prove nothing.
    """
    return _obs(
        **{
            HEAD: _camera(height, width, value=10),
            WRIST: _camera(height, width, value=200),
        }
    )


def _graph(frame=0, camera=HEAD):
    return Graph(
        frame=int(frame), env_id="PegInsertionSide-v1", camera=camera,
        nodes=[
            Node("ee", "ee", "end_effector", pose_world=[0, 0, 1, 1, 0, 0, 0]),
            Node("actor:peg", "object", "peg", pose_world=[0, 0, 1, 1, 0, 0, 0]),
        ],
        edges=[Edge("ee", "actor:peg", "planar-distance", "near")],
    )


class _GraphSource:
    """Stands in for ``FigureGraphSource``: the members the session touches.

    ``step`` reads segmentation for its declared cameras exactly as the real
    source does, so a session that widened the camera list would be caught by
    the read rather than by an assertion about it.
    """

    def __init__(self, cameras=(HEAD,)):
        self._cameras = [str(c) for c in cameras]
        self.frame = 0
        self.resets = 0
        self.seen = []

    @property
    def cameras(self):
        return list(self._cameras)

    @property
    def whitelist_dir(self):
        return "scenegraph/configs/subtask_whitelists/PegInsertionSide-v1"

    def on_reset(self):
        self.frame = 0
        self.resets += 1

    def step(self, obs):
        for camera in self._cameras:
            self.seen.append(camera)
            extract_camera_obs(obs, camera)
        graph = _graph(self.frame)
        self.frame += 1
        return graph


class _Env:
    """Enough of an env for ``FigureGraphSource``'s constructor."""

    @property
    def unwrapped(self):
        return self


# --------------------------------------------------------------------------- #
# The output layout
# --------------------------------------------------------------------------- #
class MulticameraWriterTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.graph = _graph()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _writer(self, name="PegInsertionSide-v1_seed0000", **kw):
        kw.setdefault("graph_image", False)
        return MulticameraEpisodeWriter(self.root, name, **kw)

    def _frame(self, value, height=H, width=W):
        return np.full((height, width, 3), value, dtype=np.uint8)

    def test_each_index_holds_both_views_and_exactly_one_graph(self):
        writer = self._writer(graph_image=True)
        writer.open()
        writer.write_step(step=0, head=self._frame(10), wrist=self._frame(200),
                          graph=self.graph)
        path = writer.commit()

        head = path / "frames" / "head" / "frame_0000.png"
        wrist = path / "frames" / "wrist" / "frame_0000.png"
        for item in (head, wrist,
                     path / "graphs" / "graph_0000.png",
                     path / "graph_json" / "graph_0000.json"):
            self.assertTrue(item.exists(), item)
        self.assertNotEqual(head.read_bytes(), wrist.read_bytes())
        self.assertEqual(len(list((path / "graphs").iterdir())), 1)
        self.assertEqual(len(list((path / "graph_json").iterdir())), 1)
        # The single-view layout is not also written: one flat frame directory
        # would leave a reader unable to tell which camera it came from.
        self.assertFalse((path / "frames" / "frame_0000.png").exists())

    def test_the_frames_are_written_at_the_full_sensor_resolution(self):
        """The figure prints these pixels; nothing resizes or crops them."""
        height, width = PAPER_SENSOR_SIZE
        writer = self._writer(frame_size=PAPER_SENSOR_SIZE)
        writer.open()
        writer.write_step(
            step=0,
            head=self._frame(10, height, width),
            wrist=self._frame(200, height, width),
            graph=self.graph,
        )
        path = writer.commit()
        for role in ("head", "wrist"):
            with Image.open(path / "frames" / role / "frame_0000.png") as img:
                self.assertEqual(img.size, (width, height))
                self.assertEqual(img.mode, "RGB")

    def test_a_frame_that_is_not_the_declared_size_is_refused(self):
        """A wrong-sized frame means the env was built with other sensor
        configs; the failure has to name that, not print a lopsided figure."""
        writer = self._writer(frame_size=(H, W))
        writer.open()
        with self.assertRaises(ValueError):
            writer.write_step(step=0, head=self._frame(10),
                              wrist=self._frame(200, H * 2, W), graph=self.graph)

    def test_a_frame_that_is_not_rgb_is_refused(self):
        writer = self._writer()
        writer.open()
        with self.assertRaises(ValueError):
            writer.write_step(step=0, head=np.zeros((H, W), dtype=np.uint8),
                              wrist=self._frame(200), graph=self.graph)

    def test_the_manifest_pairs_every_artifact_with_one_step(self):
        writer = self._writer()
        writer.open()
        writer.write_step(step=0, head=self._frame(10), wrist=self._frame(200),
                          graph=_graph(0))
        writer.write_step(step=5, head=self._frame(11), wrist=self._frame(201),
                          graph=_graph(5))
        path = writer.commit({"env_id": "PegInsertionSide-v1"})

        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["exported_frames"], 2)
        self.assertEqual([s["step"] for s in manifest["steps"]], [0, 5])
        second = manifest["steps"][1]
        self.assertEqual(second["head"], "frames/head/frame_0001.png")
        self.assertEqual(second["wrist"], "frames/wrist/frame_0001.png")
        self.assertEqual(second["graph_json"], "graph_json/graph_0001.json")
        self.assertEqual(second["n_nodes"], 2)
        self.assertEqual(second["n_edges"], 1)
        for record in manifest["steps"]:
            for key in ("head", "wrist", "graph_json"):
                self.assertTrue((path / record[key]).exists(), record[key])

    def test_no_callout_metadata_is_recorded(self):
        """These frames are raw sensor pixels. A chip list in the manifest
        would describe labels that were never drawn."""
        writer = self._writer()
        writer.open()
        writer.write_step(step=0, head=self._frame(10), wrist=self._frame(200),
                          graph=self.graph)
        path = writer.commit()
        manifest = json.loads((path / "episode.json").read_text())
        for record in manifest["steps"]:
            self.assertNotIn("callouts", record)
        self.assertFalse((path / "frames_clean").exists())

    def test_a_discarded_episode_leaves_nothing_behind(self):
        writer = self._writer()
        writer.open()
        writer.write_step(step=0, head=self._frame(10), wrist=self._frame(200),
                          graph=self.graph)
        writer.discard()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_an_existing_episode_is_not_replaced_without_overwrite(self):
        name = "PegInsertionSide-v1_seed0000"
        existing = episode_path(self.root, name)
        (existing / "frames").mkdir(parents=True)
        (existing / "episode.json").write_text("{\"keep\": true}")

        writer = self._writer(name)
        with self.assertRaises(FileExistsError):
            writer.open()
        self.assertEqual(
            json.loads((existing / "episode.json").read_text()), {"keep": True}
        )
        self.assertFalse((self.root / f"{STAGING_PREFIX}{name}").exists())

    def test_a_commit_onto_an_episode_that_appeared_late_is_refused(self):
        """The directory can be created while the episode is being captured;
        the gate at ``open`` is not the one that decides."""
        name = "PegInsertionSide-v1_seed0000"
        writer = self._writer(name)
        writer.open()
        writer.write_step(step=0, head=self._frame(10), wrist=self._frame(200),
                          graph=self.graph)
        episode_path(self.root, name).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            writer.commit()

    def test_overwrite_replaces_that_episode_and_nothing_beside_it(self):
        name = "PegInsertionSide-v1_seed0000"
        stale = episode_path(self.root, name)
        (stale / "graphs").mkdir(parents=True)
        (stale / "graphs" / "graph_9999.png").write_bytes(b"stale")
        # The single-view figure already published from this same root.
        neighbour = self.root / "PlaceSphere-v1_seed0006"
        (neighbour / "frames").mkdir(parents=True)
        (neighbour / "frames" / "frame_0000.png").write_bytes(b"published")

        writer = self._writer(name, overwrite=True)
        writer.open()
        writer.write_step(step=0, head=self._frame(10), wrist=self._frame(200),
                          graph=self.graph)
        path = writer.commit()

        self.assertFalse((path / "graphs" / "graph_9999.png").exists())
        self.assertTrue((path / "frames" / "head" / "frame_0000.png").exists())
        self.assertEqual(
            (neighbour / "frames" / "frame_0000.png").read_bytes(), b"published"
        )

    def test_an_episode_name_cannot_point_out_of_the_output_root(self):
        """``commit`` deletes the directory it is replacing; the name that
        aims it is checked before anything is written."""
        for name in ("../escape", "nested/name", "", "."):
            with self.assertRaises(ValueError, msg=name):
                MulticameraEpisodeWriter(self.root, name)

    def test_the_two_writers_cannot_stage_into_the_same_directory(self):
        single = EpisodeWriter(self.root, "PegInsertionSide-v1_seed0000")
        multi = self._writer("PegInsertionSide-v1_seed0000")
        self.assertNotEqual(single.staging, multi.staging)


# --------------------------------------------------------------------------- #
# The camera contract
# --------------------------------------------------------------------------- #
class PreflightTest(unittest.TestCase):
    def _check(self, obs, graphs=None, **kw):
        kw.setdefault("head", HEAD)
        kw.setdefault("wrist", WRIST)
        kw.setdefault("graph_camera", HEAD)
        kw.setdefault("sensor_size", (H, W))
        preflight(obs, graphs or _GraphSource(), **kw)

    def test_the_intended_wiring_passes(self):
        self._check(_default_obs())

    def test_an_observation_without_sensors_is_refused(self):
        with self.assertRaises(SystemExit):
            self._check({})

    def test_a_missing_wrist_camera_is_named(self):
        obs = _obs(**{HEAD: _camera()})
        with self.assertRaises(SystemExit) as caught:
            self._check(obs)
        self.assertIn(WRIST, str(caught.exception))

    def test_a_camera_without_segmentation_is_named(self):
        obs = _obs(**{HEAD: _camera(segmentation=False), WRIST: _camera()})
        with self.assertRaises(SystemExit) as caught:
            self._check(obs)
        self.assertIn("segmentation", str(caught.exception))

    def test_a_camera_at_the_wrong_resolution_is_named(self):
        """A task's default sensor size is far below print resolution, and the
        export would otherwise succeed at that size."""
        obs = _obs(**{HEAD: _camera(), WRIST: _camera(H // 2, W)})
        with self.assertRaises(SystemExit) as caught:
            self._check(obs)
        message = str(caught.exception)
        self.assertIn(WRIST, message)
        self.assertIn("rgb", message)

    def test_every_mismatch_is_reported_by_one_run(self):
        obs = _obs(**{HEAD: _camera(segmentation=False)})
        with self.assertRaises(SystemExit) as caught:
            self._check(obs)
        message = str(caught.exception)
        self.assertIn(HEAD, message)
        self.assertIn(WRIST, message)

    def test_a_graph_built_from_both_cameras_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self._check(_default_obs(), _GraphSource((HEAD, WRIST)))
        self.assertIn(WRIST, str(caught.exception))

    def test_a_graph_built_from_the_wrist_camera_is_refused(self):
        """The wrist swings with the gripper: its node set would change with
        arm motion alone."""
        with self.assertRaises(SystemExit):
            self._check(_default_obs(), _GraphSource((WRIST,)),
                        graph_camera=WRIST)

    def test_one_camera_cannot_serve_as_both_views(self):
        with self.assertRaises(SystemExit):
            self._check(_default_obs(), wrist=HEAD)


class GraphCameraTest(unittest.TestCase):
    def test_an_explicit_camera_list_pins_the_graph_to_one_sensor(self):
        """The exporter passes ``cameras=[graph_camera]`` for this reason: the
        default is every sensor the task renders, which now includes the wrist."""
        source = FigureGraphSource(
            _Env(), env_id="PegInsertionSide-v1", cameras=[HEAD]
        )
        self.assertEqual(source.cameras, [HEAD])


# --------------------------------------------------------------------------- #
# The capture session
# --------------------------------------------------------------------------- #
class SessionTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.graphs = _GraphSource()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _session(self, *extra):
        args = parse_args([
            "--out", str(self.root), "--sensor-size", str(H), str(W),
            "--no-graph-image", *extra,
        ])
        return MulticameraSession(args, self.graphs)

    def _episode(self):
        return episode_path(self.root, "PegInsertionSide-v1_seed0000")

    def test_a_captured_step_writes_both_views_and_one_graph(self):
        session = self._session()
        session.prepare(0)
        session.on_reset(_default_obs())
        session.on_step(_default_obs(), {})
        path = session.close(commit=True)

        self.assertEqual(path, self._episode())
        for index in ("0000", "0001"):
            for role in ("head", "wrist"):
                self.assertTrue(
                    (path / "frames" / role / f"frame_{index}.png").exists()
                )
            self.assertTrue((path / "graph_json" / f"graph_{index}.json").exists())

    def test_the_wrist_camera_never_reaches_the_graph_source(self):
        """The wrist frame is exported, so the wrist data is in hand; what must
        not happen is it being handed to the builder."""
        session = self._session()
        session.prepare(0)
        session.on_reset(_default_obs())
        session.on_step(_default_obs(), {})
        path = session.close(commit=True)

        self.assertEqual(set(self.graphs.seen), {HEAD})
        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["graph_cameras"], [HEAD])
        self.assertEqual(
            manifest["frame_cameras"], {"head": HEAD, "wrist": WRIST}
        )
        self.assertIs(manifest["frame_annotations"], False)
        self.assertTrue((path / "frames" / "wrist" / "frame_0000.png").exists())

    def test_the_three_artifacts_of_an_index_carry_one_step_number(self):
        session = self._session()
        session.prepare(0)
        session.on_reset(_default_obs())
        for _ in range(3):
            session.on_step(_default_obs(), {})
        path = session.close(commit=True)

        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual([s["step"] for s in manifest["steps"]], [0, 1, 2, 3])
        for record in manifest["steps"]:
            stem = f"{record['index']:04d}"
            self.assertTrue(record["head"].endswith(f"{stem}.png"))
            self.assertTrue(record["wrist"].endswith(f"{stem}.png"))
            self.assertTrue(record["graph_json"].endswith(f"{stem}.json"))
            data = json.loads((path / record["graph_json"]).read_text())
            self.assertEqual(data["frame"], record["step"])
            self.assertEqual(data["camera"], HEAD)

    def test_stride_thins_the_export_but_not_the_graph(self):
        """Temporal relations difference over the last K frames, so the builder
        has to see every step even when one in three is written."""
        session = self._session("--stride", "3")
        session.prepare(0)
        session.on_reset(_default_obs())
        for _ in range(6):
            session.on_step(_default_obs(), {})
        path = session.close(commit=True)

        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual([s["step"] for s in manifest["steps"]], [0, 3, 6])
        self.assertEqual(len(self.graphs.seen), 7)

    def test_max_frames_stops_the_export_after_its_count(self):
        session = self._session("--max-frames", "2")
        session.prepare(0)
        session.on_reset(_default_obs())
        for _ in range(4):
            session.on_step(_default_obs(), {})
        path = session.close(commit=True)

        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["exported_frames"], 2)

    def test_a_failed_attempt_leaves_no_episode_and_no_staging(self):
        session = self._session()
        session.prepare(0)
        session.on_reset(_default_obs())
        session.on_step(_default_obs(), {})
        self.assertIsNone(session.close(commit=False))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_second_attempt_drops_what_the_first_one_staged(self):
        session = self._session()
        session.prepare(0)
        session.on_reset(_default_obs())
        session.on_step(_default_obs(), {})
        session.prepare(1)
        session.on_reset(_default_obs())
        path = session.close(commit=True)

        self.assertEqual(path.name, "PegInsertionSide-v1_seed0001")
        self.assertEqual([p.name for p in self.root.iterdir()], [path.name])
        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["exported_frames"], 1)

    def test_an_existing_episode_stops_the_run_before_the_attempt(self):
        """Refused here rather than inside a capture hook: the runner turns a
        hook's exception into a failed attempt, and the loop would walk to the
        next seed instead of reporting the collision."""
        self._episode().mkdir(parents=True)
        session = self._session()
        with self.assertRaises(SystemExit):
            session.prepare(0)

    def test_overwrite_is_what_authorises_the_replacement(self):
        self._episode().mkdir(parents=True)
        session = self._session("--overwrite")
        session.prepare(0)
        session.on_reset(_default_obs())
        self.assertEqual(session.close(commit=True), self._episode())


# --------------------------------------------------------------------------- #
# Isolation from the single-view figure
# --------------------------------------------------------------------------- #
class IsolationTest(unittest.TestCase):
    EXPORTER = ROOT / "scenegraph" / "tools" / "render_multicamera_paper_frames.py"
    WRITER = ROOT / "scenegraph" / "figures" / "multicamera_writer.py"
    OLD_EXPORTER = ROOT / "scenegraph" / "tools" / "render_paper_frames.py"

    def _imported(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        return modules

    def _imported_names(self, path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) for alias in node.names}

    def test_the_exporter_never_reaches_the_callout_pipeline(self):
        """No chips, no leader lines, no boxes: whatever the sensor returned is
        what the figure prints."""
        for path in (self.EXPORTER, self.WRITER):
            for module in self._imported(path):
                self.assertNotIn("annotate", module, f"{path.name}: {module}")

    def test_the_exporter_does_not_reuse_the_single_view_writer(self):
        """It shares the PNG and JSON conventions, and nothing else: the
        layouts differ, so one class serving both would have to grow a mode."""
        self.assertNotIn("EpisodeWriter", self._imported_names(self.EXPORTER))
        # Word-anchored: ``MulticameraEpisodeWriter(`` is the class this
        # exporter is supposed to build.
        source = self.EXPORTER.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\bEpisodeWriter\(", source))
        self.assertEqual(
            self._imported_names(self.WRITER) & {"EpisodeWriter", "FRAME_DIR"},
            set(),
        )

    def test_the_single_view_exporter_does_not_know_about_this_one(self):
        """The dependency runs one way. The PlaceSphere figure has to keep
        rendering exactly as it did."""
        for module in self._imported(self.OLD_EXPORTER):
            self.assertNotIn("multicamera", module)

    def test_the_single_view_writer_still_writes_its_own_layout(self):
        root = pathlib.Path(tempfile.mkdtemp())
        try:
            writer = EpisodeWriter(root, "PlaceSphere-v1_seed0000",
                                   graph_image=False)
            writer.open()
            writer.write_step(
                step=0, frame=np.zeros((H, W, 3), dtype=np.uint8), graph=_graph()
            )
            path = writer.commit()
            self.assertTrue((path / "frames" / "frame_0000.png").exists())
            self.assertFalse((path / "frames" / "head").exists())
            self.assertFalse((path / "frames" / "wrist").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
