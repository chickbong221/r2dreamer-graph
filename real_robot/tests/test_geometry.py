"""Geometry primitives: alignment, planes, robust depth, gap filling and smoothing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from ..common import load_configs
from ..preprocessing.estimate_geometry import (
    GeometryStage,
    apply_similarity,
    backproject,
    convex_hull,
    derive_tcp,
    euler_zyx,
    fill_series,
    fit_alignment,
    fit_plane_ransac,
    inside_hull,
    measurements,
    plane_degeneracy,
    robust_depth,
    smooth_series,
    stationary_position,
    table_frame,
    umeyama,
)
from . import synthetic as syn


def rotation(angle_z: float, angle_x: float) -> np.ndarray:
    cz, sz, cx, sx = np.cos(angle_z), np.sin(angle_z), np.cos(angle_x), np.sin(angle_x)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ rx


class Alignment(unittest.TestCase):
    def test_umeyama_recovers_a_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(50, 3))
        R, s, t = rotation(0.7, -0.4), 0.8, np.array([0.1, -0.2, 0.5])
        dst = s * src @ R.T + t
        s2, R2, t2 = umeyama(src, dst)
        self.assertAlmostEqual(s2, s, places=6)
        np.testing.assert_allclose(R2, R, atol=1e-6)
        np.testing.assert_allclose(t2, t, atol=1e-6)

    def test_camera_to_robot_with_a_fingertip_offset_and_outliers(self):
        rng = np.random.default_rng(1)
        n = 300
        eef = rng.uniform([0.5, -0.3, 0.0], [1.0, 0.3, 0.4], size=(n, 3))
        rpy = rng.uniform(-0.6, 0.6, size=(n, 3))
        offset = np.array([0.0, 0.0, 0.08])
        tips = eef + np.einsum("nij,j->ni", euler_zyx(rpy[:, 0], rpy[:, 1], rpy[:, 2]), offset)
        R, s, t = rotation(2.0, 1.2), 1.15, np.array([0.3, 0.1, 0.9])
        # camera point p with robot tip = s R p + t
        camera = (tips - t) @ R / s
        camera += rng.normal(scale=0.003, size=camera.shape)
        camera[:30] += rng.normal(scale=0.3, size=(30, 3))
        fit = fit_alignment(camera, eef, rpy, iterations=500, inlier=0.02, rng=np.random.default_rng(2))
        self.assertAlmostEqual(fit["scale"], s, delta=0.02)
        self.assertTrue(fit["tool_offset_used"])
        np.testing.assert_allclose(fit["tool_offset_gripper_frame"], offset, atol=0.01)
        self.assertLess(fit["median_residual_m"], 0.01)
        mapped = apply_similarity(fit, camera[30:])
        self.assertLess(np.median(np.linalg.norm(mapped - tips[30:], axis=1)), 0.01)


class Surfaces(unittest.TestCase):
    def test_plane_ransac_ignores_objects_on_the_table(self):
        rng = np.random.default_rng(3)
        normal = np.array([0.0, -0.8, -0.6])
        normal /= np.linalg.norm(normal)
        basis = np.linalg.svd(normal[None])[2][1:]
        table = rng.uniform(-0.3, 0.3, size=(200, 2)) @ basis + 1.0 * normal * -1.2
        clutter = table[:40] - normal * 0.1
        found, offset, inliers = fit_plane_ransac(np.vstack([table, clutter]), 300, 0.005, rng)
        self.assertGreater(abs(found @ normal), 0.999)
        self.assertGreaterEqual(inliers[:200].mean(), 0.99)
        self.assertLessEqual(inliers[200:].mean(), 0.05)

    def test_table_frame_puts_the_table_at_zero_height_with_z_up(self):
        normal = np.array([0.0, -0.8, -0.6])
        normal /= np.linalg.norm(normal)
        offset = 1.2
        frame = table_frame(normal, offset)
        on_table = np.array([[0.1, 0.2, 0.0]])
        on_table = on_table - (on_table @ normal + offset) * normal
        camera = np.zeros((1, 3))
        self.assertAlmostEqual(float(apply_similarity(frame, on_table)[0, 2]), 0.0, places=6)
        self.assertGreater(float(apply_similarity(frame, camera)[0, 2]), 0.0)

    def test_depth_window_rejects_the_background_behind_a_thin_object(self):
        depth = np.full((20, 20), 1.0, dtype=np.float32)
        depth[9:11, 9:11] = 0.6
        self.assertAlmostEqual(robust_depth(depth, 10 / 19, 10 / 19, radius=3, background_tol=0.05), 0.6, places=5)
        self.assertTrue(np.isnan(robust_depth(depth, np.nan, 0.5, 3, 0.05)))

    def test_backprojection_centre_and_edge(self):
        np.testing.assert_allclose(backproject(0.5, 0.5, 2.0, 500.0, 640, 480), [0, 0, 2.0])
        np.testing.assert_allclose(backproject(1.0, 0.5, 1.0, 320.0, 640, 480), [1.0, 0, 1.0])


class Series(unittest.TestCase):
    def test_past_only_filling_holds_and_never_reads_ahead(self):
        values = np.array([[0.0], [np.nan], [np.nan], [3.0]])
        valid = np.array([True, False, False, True])
        filled, known, gap = fill_series(values, valid, "past_only", max_gap=5)
        np.testing.assert_allclose(filled[:, 0], [0.0, 0.0, 0.0, 3.0])
        self.assertEqual(gap.tolist(), [0, 1, 2, 0])
        whole, _, whole_gap = fill_series(values, valid, "full_episode", max_gap=5)
        np.testing.assert_allclose(whole[:, 0], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(whole_gap.tolist(), [0, 2, 2, 0])
        leading, known_leading, _ = fill_series(np.array([[np.nan], [1.0]]), np.array([False, True]),
                                                "full_episode", max_gap=5)
        self.assertTrue(np.isnan(leading[0, 0]))
        self.assertEqual(known_leading.tolist(), [False, True])

    def test_one_measurement_does_not_become_a_known_episode(self):
        # The review's case: a single measured position in 100 frames.
        values = np.full((100, 3), np.nan)
        values[0] = [0.1, 0.2, 0.3]
        valid = np.isfinite(values).all(axis=1)
        for mode in ("full_episode", "past_only"):
            filled, known, gap = fill_series(values, valid, mode, max_gap=5)
            self.assertEqual(int(known.sum()), 6, mode)             # the measurement and five held frames
            self.assertTrue(np.isnan(filled[6:]).all(), mode)
            self.assertEqual(int((gap == 0).sum()), 1, mode)        # gap == 0 is the measurement mask
            self.assertEqual(int(gap.max()), 99, mode)

    def test_a_long_gap_between_measurements_stays_unknown(self):
        values = np.full((51, 1), np.nan)
        values[0], values[50] = 0.0, 5.0
        valid = np.isfinite(values[:, 0])
        filled, known, gap = fill_series(values, valid, "full_episode", max_gap=5)
        self.assertFalse(known[1:50].any())
        self.assertTrue((gap[1:50] == 49).all())
        speed = measurements({"ee": np.zeros((51, 3)), "banana:center": np.repeat(filled, 3, axis=1),
                              "banana:grasp_region": np.full((51, 3), np.nan),
                              "pot:rim_center": np.zeros((51, 3)), "lid:center": np.zeros((51, 3)),
                              "lid:handle": np.zeros((51, 3))}, fps=15.0, entry_height=0.05)["banana_speed"]
        self.assertTrue(np.isnan(speed[1:51]).all())                 # no speed, so no false settling
        bridged, bridged_known, _ = fill_series(values, valid, "full_episode", max_gap=49)
        self.assertTrue(bridged_known.all())
        self.assertAlmostEqual(float(bridged[25, 0]), 2.5)

    def test_the_pot_is_held_through_gaps_only_when_verified_stationary(self):
        rng = np.random.default_rng(5)
        still = np.tile([0.4, 0.0, 0.1], (60, 1)) + rng.normal(scale=0.002, size=(60, 3))
        measured = np.zeros(60, dtype=bool)
        measured[[3, 10, 20, 40]] = True
        rim, known, report = stationary_position(still, measured, "full_episode", 0.02, 3)
        self.assertTrue(report["verified"] and known.all())
        np.testing.assert_allclose(rim[0], [0.4, 0.0, 0.1], atol=0.005)
        moved = still.copy()
        moved[40] += [0.2, 0.0, 0.0]
        rim, known, report = stationary_position(moved, measured, "full_episode", 0.02, 3)
        self.assertIsNone(rim)
        self.assertFalse(report["verified"] or known.any())
        # Past only: verified from the third measurement on, and never from a later one.
        rim, known, report = stationary_position(moved, measured, "past_only", 0.02, 3)
        self.assertEqual(report["verified_from_frame"], 20)
        self.assertFalse(known[:20].any())
        self.assertTrue(known[20:40].all())
        self.assertFalse(known[40:].any())

    def test_causal_smoothing_ignores_the_future(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(30, 3))
        b = a.copy()
        b[20:] += 5.0
        valid = np.ones(30, dtype=bool)
        settings = {"method": "ema", "alpha": 0.4}
        np.testing.assert_allclose(smooth_series(a, valid, settings)[:20], smooth_series(b, valid, settings)[:20])

    def test_measurements_from_positions(self):
        n = 3
        points = {
            "ee": np.array([[0.0, 0.0, 0.1]] * n),
            "banana:center": np.array([[0.1, 0.0, 0.0]] * n),
            "banana:grasp_region": np.array([[0.0, 0.0, 0.0]] * n),
            "pot:rim_center": np.array([[0.4, 0.0, 0.1]] * n),
            "lid:center": np.array([[0.4, 0.03, 0.12]] * n),
            "lid:handle": np.full((n, 3), np.nan),
        }
        values = measurements(points, fps=15.0, entry_height=0.05)
        self.assertAlmostEqual(values["d_gripper_banana"][0], 0.1)
        self.assertAlmostEqual(values["lid_lateral_error"][0], 0.03)
        self.assertAlmostEqual(values["lid_height_above_rim"][0], 0.02)
        self.assertAlmostEqual(values["d_gripper_lid_handle"][0], np.linalg.norm([0.4, 0.03, 0.02]))
        self.assertTrue(np.isnan(values["banana_speed"][0]))
        self.assertEqual(values["banana_speed"][1], 0.0)



class TablePlane(unittest.TestCase):
    def stage(self):
        configs = load_configs(["dataset", "annotation", "graph"])
        stage = GeometryStage.__new__(GeometryStage)
        stage.configs, stage.cfg = configs, configs["annotation"]["geometry"]
        stage.spec = syn.graph_spec()
        stage.camera = stage.spec.cameras[0]
        stage.source = SimpleNamespace(mode="full_episode")
        return stage

    def lifted(self, stage, n=30):
        entities, cameras = len(stage.spec.entity_ids), len(stage.spec.cameras)
        return {"depth": {"depth": np.ones((n, 60, 80), dtype=np.float16), "frames": np.arange(n)},
                "hw": (480, 640), "focal": 500.0,
                "tracks": {"boxes": np.zeros((n, entities, cameras, 4), np.float32),
                           "visible": np.zeros((n, entities, cameras), bool)}}

    @staticmethod
    def keyframe(frame, points):
        return {"frame": frame, "camera": "high", "object": "table", "visible": True, "box": [0.0, 1.0, 0.5, 1.0],
                "points": points, "hidden_points": []}

    def test_the_plane_comes_from_a_keyframe_whose_points_span_one(self):
        # The review's case: the first table keyframe has one point, a later one three good points.
        stage = self.stage()
        one = self.keyframe(0, {"surface_1": [0.5, 0.8]})
        line = self.keyframe(10, {"surface_1": [0.1, 0.8], "surface_2": [0.5, 0.8], "surface_3": [0.9, 0.8]})
        spread = self.keyframe(20, {"surface_1": [0.15, 0.7], "surface_2": [0.85, 0.7], "surface_3": [0.5, 0.95]})
        samples, key = stage._table_samples(0, self.lifted(stage), SimpleNamespace(keyframes=[one, line, spread]))
        self.assertEqual(key["frame"], 20)
        self.assertIsNone(plane_degeneracy(samples, 0.01))
        normal, _, _ = fit_plane_ransac(samples, 100, 0.01, np.random.default_rng(0))
        self.assertGreater(abs(normal[2]), 0.999)      # constant depth: the plane faces the camera

    def test_a_point_or_a_line_is_refused_not_fitted(self):
        stage = self.stage()
        one = self.keyframe(0, {"surface_1": [0.5, 0.8]})
        line = self.keyframe(10, {"surface_1": [0.1, 0.8], "surface_2": [0.5, 0.8], "surface_3": [0.9, 0.8]})
        with self.assertRaises(ValueError):
            stage._table_samples(0, self.lifted(stage), SimpleNamespace(keyframes=[one, line]))
        same = np.tile([0.1, 0.2, 1.0], (145, 1))
        self.assertIn("1 distinct", plane_degeneracy(same, 0.01))
        with self.assertRaises(ValueError):
            fit_plane_ransac(same, 50, 0.01, np.random.default_rng(0))
        collinear = np.stack([np.linspace(0, 1, 20), np.linspace(0, 1, 20), np.ones(20)], axis=1)
        self.assertIn("line", plane_degeneracy(collinear, 0.01))

    def test_the_sampling_grid_stays_inside_the_points(self):
        hull = convex_hull(np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.5, 0.3]]))
        self.assertEqual(len(hull), 3)
        self.assertTrue(inside_hull(hull, 0.5, 0.4))
        self.assertFalse(inside_hull(hull, 0.12, 0.8))    # inside the bounding box, outside the triangle


class GripperPoint(unittest.TestCase):
    def test_the_closing_point_is_the_midpoint_of_the_lifted_fingertips(self):
        names = ["banana:center", "ee:fingertip_1", "ee:fingertip_2"]
        points = np.full((3, 3, 3), np.nan)
        points[0, 1], points[0, 2] = [0.10, 0.0, 0.50], [0.14, 0.02, 0.52]
        points[1, 1] = [0.1, 0.1, 0.1]                     # one fingertip hidden
        out_names, out = derive_tcp(names, points)
        self.assertEqual(out_names[-1], "ee:tcp")
        np.testing.assert_allclose(out[0, -1], [0.12, 0.01, 0.51])
        self.assertTrue(np.isnan(out[1, -1]).all())
        self.assertTrue(np.isnan(out[2, -1]).all())

    def test_the_depth_between_open_fingertips_is_not_the_gripper(self):
        # Table 1.00 m away; two fingers 0.80 m away, 20 pixels apart. The image midpoint reads the table.
        depth = np.full((120, 160), 1.0, dtype=np.float32)
        depth[55:65, 60:64] = 0.8
        depth[55:65, 96:100] = 0.8
        midpoint = robust_depth(depth, 80 / 159, 60 / 119, 3, 0.03)
        fingertip = robust_depth(depth, 62 / 159, 60 / 119, 3, 0.03)
        self.assertAlmostEqual(midpoint, 1.0, places=5)
        self.assertAlmostEqual(fingertip, 0.8, places=5)


class CameraCheck(unittest.TestCase):
    def test_an_unmeasured_or_moved_camera_is_not_fixed(self):
        self.assertTrue(GeometryStage._fixed({"across_px": 1.0, "within_px": 2.5}, 3.0))
        self.assertFalse(GeometryStage._fixed({"across_px": 4.0, "within_px": 0.5}, 3.0))
        self.assertFalse(GeometryStage._fixed({"across_px": float("nan"), "within_px": 0.5}, 3.0))


if __name__ == "__main__":
    unittest.main()
