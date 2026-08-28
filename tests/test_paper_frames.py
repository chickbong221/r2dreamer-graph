"""Paper-figure capture: projection, label placement, and the output layout.

No simulator. The render camera is a pair of matrices and entities are stubs
holding collision shapes, the same fakes ``test_camera_projection`` uses -- what
can go wrong here is geometry (an object behind the camera labelled in the
middle of the frame), layout (two names stacked on one another), and file
layout (a failed episode left half-written on disk).
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.figures.annotate import (
    Callout, DEFAULT_LABELS, build_callouts, draw_callouts, fixed_labels,
    label_every_node,
)
from scenegraph.figures.render_camera import FigureCamera
from scenegraph.figures.writer import EpisodeWriter
from scenegraph.viz.palette import ColorMap

W = H = 400
# 90-degree field of view, principal point at the centre.
K = np.array([[200.0, 0.0, 200.0], [0.0, 200.0, 200.0], [0.0, 0.0, 1.0]])
# Camera at the origin looking down +z, world axes aligned.
EYE = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _Geom:
    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class _Shape:
    def __init__(self, geometry):
        self.geometry = geometry
        self.local_pose = None


class _Body:
    def __init__(self, shapes):
        self._shapes = list(shapes)

    def get_collision_shapes(self):
        return self._shapes


class _Obj:
    def __init__(self, shapes):
        self.components = [_Body(shapes)]


class _Entity:
    def __init__(self, half):
        self._objs = [_Obj([_Shape(_Geom(half_size=np.asarray(half, float)))])]
        self._scene_idxs = [0]


class _RenderCamera:
    def __init__(self, intrinsic=K, extrinsic=EYE, width=W, height=H):
        self._k, self._ext = intrinsic, extrinsic
        self.width, self.height = width, height

    def get_intrinsic_matrix(self):
        return self._k

    def get_extrinsic_matrix(self):
        return self._ext


class _Scene:
    def __init__(self, camera):
        self.human_render_cameras = {"render_camera": camera}


class _Env:
    def __init__(self, camera=None, frame=None):
        self.scene = _Scene(camera or _RenderCamera())
        self._frame = frame

    @property
    def unwrapped(self):
        return self

    def render(self):
        return self._frame[None]


def _pose(xyz):
    return [*xyz, 1.0, 0.0, 0.0, 0.0]


def _node(node_id, name, xyz, node_type="object"):
    return Node(node_id=node_id, node_type=node_type, name=name,
                pose_world=_pose(xyz))


def _graph(nodes, edges=()):
    return Graph(frame=0, env_id="env0", camera="base_camera",
                 nodes=list(nodes), edges=list(edges))


def _blank():
    return np.full((H, W, 3), 90, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
class FigureCameraTest(unittest.TestCase):
    def setUp(self):
        self.camera = FigureCamera(_Env())

    def test_a_point_on_the_axis_lands_on_the_principal_point(self):
        self.assertEqual(self.camera.point_pixel([0.0, 0.0, 2.0]), (200.0, 200.0))

    def test_a_point_behind_the_camera_has_no_pixel(self):
        """A negative depth flips the sign of the projection, so a ghost of an
        object behind the robot would otherwise be labelled inside the frame."""
        self.assertIsNone(self.camera.point_pixel([0.0, 0.0, -2.0]))

    def test_the_frame_picks_the_render_camera_by_name(self):
        self.assertEqual(self.camera.name, "render_camera")

    def test_capture_drops_the_batch_row(self):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        camera = FigureCamera(_Env(frame=frame))
        self.assertEqual(camera.capture().shape, (H, W, 3))

    def test_a_box_projects_to_a_clipped_pixel_rectangle(self):
        box = self.camera.entity_box(_Entity((0.1, 0.1, 0.1)), _pose((0, 0, 1.0)))
        self.assertIsNotNone(box)
        x0, y0, x1, y1 = box
        self.assertLess(x0, 200.0)
        self.assertGreater(x1, 200.0)
        self.assertGreaterEqual(x0, 0.0)
        self.assertLessEqual(y1, float(H))

    def test_an_object_spanning_the_view_is_clipped_not_dropped(self):
        """The table runs past both edges; a figure still has to label it."""
        box = self.camera.entity_box(_Entity((5.0, 5.0, 0.02)), _pose((0, 0, 1.0)))
        self.assertEqual(box, (0.0, 0.0, float(W), float(H)))

    def test_an_object_entirely_behind_the_camera_has_no_box(self):
        self.assertIsNone(
            self.camera.entity_box(_Entity((0.1, 0.1, 0.1)), _pose((0, 0, -2.0)))
        )

    def test_the_geometry_cache_is_dropped_on_invalidate(self):
        entity = _Entity((0.1, 0.1, 0.1))
        self.camera.entity_box(entity, _pose((0, 0, 1.0)))
        self.assertTrue(self.camera._aabb)
        self.camera.invalidate()
        self.assertFalse(self.camera._aabb)


# --------------------------------------------------------------------------- #
# Callouts
# --------------------------------------------------------------------------- #
class CalloutTest(unittest.TestCase):
    def setUp(self):
        self.camera = FigureCamera(_Env())
        self.entities = {
            "actor:sphere": _Entity((0.04, 0.04, 0.04)),
            "actor:bin": _Entity((0.1, 0.1, 0.05)),
            "actor:table-workspace": _Entity((0.6, 0.6, 0.02)),
            "actor:goal_site": _Entity((0.02, 0.02, 0.02)),
        }
        self.graph = _graph([
            _node("ee", "end_effector", (0.0, -0.2, 1.0), node_type="ee"),
            _node("actor:sphere", "sphere", (-0.2, 0.0, 1.2)),
            _node("actor:bin", "bin", (0.2, 0.0, 1.2)),
            _node("actor:table-workspace", "table-workspace", (0.0, 0.3, 1.5)),
            _node("actor:goal_site", "goal_site", (0.0, 0.0, 1.1)),
        ])

    def test_only_the_named_entities_are_labelled(self):
        calls = build_callouts(self.graph, self.camera, self.entities)
        self.assertEqual(
            {c.node_id for c in calls},
            {"ee", "actor:sphere", "actor:bin", "actor:table-workspace"},
        )
        self.assertEqual(
            {c.text for c in calls}, {"ee", "sphere", "bin", "table"}
        )

    def test_the_gripper_is_labelled_without_an_entity(self):
        """``ee`` is a tool-centre pose, not an actor, so it has no AABB to
        project. Its visual target is raised from the TCP onto the housing."""
        calls = build_callouts(self.graph, self.camera, self.entities)
        ee = next(c for c in calls if c.node_id == "ee")
        self.assertIsNone(ee.box)
        self.assertTrue(all(np.isfinite(ee.anchor)))
        tcp = self.camera.point_pixel([0.0, -0.2, 1.0])
        self.assertAlmostEqual(ee.anchor[0], tcp[0])
        self.assertAlmostEqual(ee.anchor[1], tcp[1] - 0.05 * H)

    def test_labels_all_names_every_vertex(self):
        calls = build_callouts(
            self.graph, self.camera, self.entities, labels=label_every_node
        )
        self.assertEqual(len(calls), len(self.graph.nodes))
        self.assertIn("goal_site", {c.text for c in calls})

    def test_a_custom_map_overrides_the_default(self):
        calls = build_callouts(
            self.graph, self.camera, self.entities,
            labels=fixed_labels({"actor:sphere": "ball"}),
        )
        self.assertEqual([c.text for c in calls], ["ball"])

    def test_chip_colours_match_the_node_link_diagram(self):
        """One ColorMap drives both pictures, so the chip beside the sphere is
        the colour of the sphere's circle in the graph printed next to it."""
        shared = ColorMap()
        calls = build_callouts(
            self.graph, self.camera, self.entities, colormap=shared
        )
        sphere = next(c for c in calls if c.node_id == "actor:sphere")
        expected = tuple(
            int(round(255 * v)) for v in shared.color("actor:sphere")
        )
        self.assertEqual(sphere.color, expected)

    def test_a_node_with_no_pose_is_skipped(self):
        graph = _graph([Node("actor:sphere", "object", "sphere")])
        self.assertEqual(build_callouts(graph, self.camera, {}), [])

    def test_an_offset_moves_only_its_own_chip(self):
        base = draw_callouts(_blank(), build_callouts(
            self.graph, self.camera, self.entities))
        moved = draw_callouts(_blank(), build_callouts(
            self.graph, self.camera, self.entities,
            offsets={"actor:sphere": (0.0, 60.0)}))
        self.assertFalse(np.array_equal(base, moved))


class LabelLayoutTest(unittest.TestCase):
    def setUp(self):
        self.camera = FigureCamera(_Env())

    def _draw(self, graph, entities, **kw):
        calls = build_callouts(graph, self.camera, entities)
        image = draw_callouts(_blank(), calls, **kw)
        return calls, image

    def test_the_frame_keeps_its_size_and_dtype(self):
        graph = _graph([_node("actor:sphere", "sphere", (0.0, 0.0, 1.0))])
        _calls, image = self._draw(graph, {})
        self.assertEqual(image.shape, (H, W, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_drawing_does_not_modify_the_captured_frame(self):
        frame = _blank()
        graph = _graph([_node("actor:sphere", "sphere", (0.0, 0.0, 1.0))])
        draw_callouts(frame, build_callouts(graph, self.camera, {}))
        np.testing.assert_array_equal(frame, _blank())

    def test_chips_for_objects_at_the_same_pixel_do_not_overlap(self):
        """Two objects a hand's width apart in the world can be a few pixels
        apart in a third-person view; two overlapping names name neither."""
        graph = _graph([
            _node("actor:sphere", "sphere", (0.0, 0.0, 1.0)),
            _node("actor:bin", "bin", (0.001, 0.0, 1.0)),
            _node("ee", "end_effector", (0.002, 0.0, 1.0), node_type="ee"),
        ])
        calls, _image = self._draw(graph, {})
        self.assertEqual(len(calls), 3)
        chips = [c.chip for c in calls]
        self.assertTrue(all(chip is not None for chip in chips))
        for i, a in enumerate(chips):
            for b in chips[i + 1:]:
                overlap = not (a[2] <= b[0] or b[2] <= a[0]
                               or a[3] <= b[1] or b[3] <= a[1])
                self.assertFalse(overlap, f"{a} overlaps {b}")

    def test_a_name_prefers_the_side_of_its_object(self):
        call = Callout(
            "actor:sphere", "sphere", (200.0, 200.0), (10, 20, 30),
            box=(180.0, 180.0, 220.0, 220.0),
        )
        draw_callouts(_blank(), [call], font_size=20, lift=10)
        self.assertGreater(call.chip[0], call.box[2])

    def test_the_ee_chip_is_left_of_its_visual_target(self):
        call = Callout("ee", "ee", (200.0, 200.0), (10, 20, 30))
        image = draw_callouts(_blank(), [call], font_size=20, lift=10)
        self.assertLess(call.chip[2], call.anchor[0])
        self.assertLess(call.chip[1], call.anchor[1])
        self.assertGreater(call.chip[3], call.anchor[1])
        np.testing.assert_array_equal(image[200, 200], call.color)

    def test_a_wide_surface_is_labelled_at_its_right_edge(self):
        call = Callout(
            "actor:table-workspace", "table", (200.0, 280.0), (10, 20, 30),
            box=(0.0, 120.0, 400.0, 400.0),
        )
        image = draw_callouts(_blank(), [call], font_size=20, lift=10)
        self.assertGreater(call.chip[1], call.box[1])
        self.assertLess(call.chip[3], call.anchor[1])
        self.assertGreater(call.chip[0], 0.75 * W)
        dot_x = int(0.5 * (call.chip[0] + call.chip[2]))
        np.testing.assert_array_equal(image[280, dot_x], call.color)

    def test_every_chip_stays_inside_the_frame(self):
        graph = _graph([
            _node("actor:sphere", "sphere", (-0.9, -0.9, 1.0)),
            _node("actor:bin", "bin", (0.9, 0.9, 1.0)),
        ])
        calls, _image = self._draw(graph, {})
        for call in calls:
            x0, y0, x1, y1 = call.chip
            self.assertGreaterEqual(x0, 0.0)
            self.assertGreaterEqual(y0, 0.0)
            self.assertLessEqual(x1, float(W))
            self.assertLessEqual(y1, float(H))

    def test_an_object_at_the_top_edge_is_labelled_below_it(self):
        """There is no room above, and a chip clamped to the top edge would sit
        on the object instead of pointing at it."""
        graph = _graph([_node("actor:sphere", "sphere", (0.0, -0.95, 1.0))])
        calls, _image = self._draw(graph, {})
        self.assertGreater(calls[0].chip[1], calls[0].anchor[1])

    def test_the_chip_is_actually_painted_in_the_node_colour(self):
        graph = _graph([_node("actor:sphere", "sphere", (0.0, 0.0, 1.0))])
        calls, image = self._draw(graph, {})
        x0, y0, x1, y1 = calls[0].chip
        patch = image[int(y0) + 4:int(y1) - 4, int(x0) + 4:int(x1) - 4]
        self.assertTrue(
            (patch == np.asarray(calls[0].color, dtype=np.uint8)).all(axis=-1).any()
        )


# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #
class EpisodeWriterTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.graph = _graph(
            [_node("ee", "end_effector", (0, 0, 1), node_type="ee"),
             _node("actor:sphere", "sphere", (0, 0, 1))],
            [Edge("ee", "actor:sphere", "planar-distance", "near")],
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _writer(self, **kw):
        return EpisodeWriter(self.root, "PlaceSphere-v1_seed0000", **kw)

    def test_frame_and_graph_are_separate_files(self):
        writer = self._writer()
        writer.open()
        writer.write_step(step=0, frame=_blank(), graph=self.graph)
        path = writer.commit({"env_id": "PlaceSphere-v1"})

        frame = path / "frames" / "frame_0000.png"
        diagram = path / "graphs" / "graph_0000.png"
        data = path / "graph_json" / "graph_0000.json"
        for item in (frame, diagram, data):
            self.assertTrue(item.exists(), item)
        self.assertNotEqual(frame.read_bytes(), diagram.read_bytes())
        self.assertEqual(
            json.loads(data.read_text())["nodes"][0]["node_id"], "ee"
        )

    def test_the_manifest_pairs_each_frame_with_its_graph(self):
        writer = self._writer()
        writer.open()
        writer.write_step(step=0, frame=_blank(), graph=self.graph)
        writer.write_step(step=5, frame=_blank(), graph=self.graph)
        path = writer.commit({"env_id": "PlaceSphere-v1", "stride": 5})

        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["exported_frames"], 2)
        self.assertEqual([s["step"] for s in manifest["steps"]], [0, 5])
        self.assertEqual(manifest["steps"][1]["frame"],
                         "frames/frame_0001.png")
        self.assertEqual(manifest["steps"][1]["graph"],
                         "graphs/graph_0001.png")

    def test_a_discarded_episode_leaves_nothing_behind(self):
        """Success is only known at the end of an episode, so a failed attempt
        must not leave a directory that looks exactly like a kept one."""
        writer = self._writer()
        writer.open()
        writer.write_step(step=0, frame=_blank(), graph=self.graph)
        writer.discard()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_the_clean_frame_is_opt_in(self):
        writer = self._writer(save_clean=True)
        writer.open()
        writer.write_step(step=0, frame=_blank(), graph=self.graph,
                          clean=_blank())
        path = writer.commit()
        self.assertTrue((path / "frames_clean" / "frame_0000.png").exists())

    def test_numpy_values_from_the_simulator_still_serialise(self):
        """Raw relation values and node boxes arrive as numpy scalars; one
        float32 must not abort the export of the frame it belongs to."""
        graph = _graph(
            [Node("actor:sphere", "object", "sphere",
                  pose_world=_pose((0, 0, 1)),
                  bbox=np.zeros((1, 4), dtype=np.float32),
                  attributes={"radius": np.float32(0.02)})],
            [Edge("ee", "actor:sphere", "planar-distance", "near",
                  raw_value=np.float32(0.11))],
        )
        graph.meta["n_visible"] = np.int64(1)

        writer = self._writer(graph_image=False)
        writer.open()
        writer.write_step(step=0, frame=_blank(), graph=graph)
        path = writer.commit()

        data = json.loads((path / "graph_json" / "graph_0000.json").read_text())
        self.assertAlmostEqual(data["edges"][0]["raw_value"], 0.11, places=5)
        self.assertEqual(data["meta"]["n_visible"], 1)

    def test_skipping_the_diagram_still_writes_the_data(self):
        writer = self._writer(graph_image=False)
        writer.open()
        writer.write_step(step=0, frame=_blank(), graph=self.graph)
        path = writer.commit()
        self.assertFalse((path / "graphs").exists())
        self.assertTrue((path / "graph_json" / "graph_0000.json").exists())


class DefaultLabelTest(unittest.TestCase):
    def test_the_default_map_covers_the_place_sphere_vertices(self):
        """The whitelist mined for PlaceSphere-v1 admits exactly these three
        actors; with ``ee`` that is every vertex the task's graph can hold."""
        members = json.loads(
            (Path(__file__).resolve().parents[1] / "scenegraph" / "configs"
             / "subtask_whitelists" / "PlaceSphere-v1" / "task_all.json")
            .read_text()
        )["members"]
        self.assertEqual(set(DEFAULT_LABELS) - {"ee"}, set(members))


if __name__ == "__main__":
    unittest.main()
