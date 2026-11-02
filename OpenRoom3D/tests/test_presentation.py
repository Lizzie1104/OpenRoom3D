"""Synthetic checks for reversible demo background operations."""
import copy
import unittest

import numpy as np

from openroom3d.presentation import (
    cleanup_mask,
    object_has_moved,
    plane_coordinates,
    table_patch,
)


def plane_config():
    return {
        "normal": [0., 0., 1.],
        "origin": [2., 3., 4.],
        "offset": -4.,
        "tangent_axes": [[1., 0.], [0., 1.], [0., 0.]],
        "footprint_min": [-1., -.5],
        "footprint_max": [1., .5],
        "cleanup_scope_margin": .25,
        "background_min_height": -.08,
        "column_min_height": -.02,
        "column_max_height": 1.,
        "patch_height": .0003,
    }


def world_points(local, config):
    local = np.asarray(local, dtype=np.float64)
    return (np.asarray(config["origin"])
            + local[:, :2] @ np.asarray(config["tangent_axes"]).T
            + local[:, 2, None] * np.asarray(config["normal"]))


class PresentationTest(unittest.TestCase):
    def test_plane_coordinates_use_fixed_world_plane(self):
        config = plane_config()
        local = np.array([[.2, -.3, .7], [-1., .5, -.1], [0., 0., 0.]])
        points = world_points(local, config)
        original = points.copy()
        uv, height = plane_coordinates(points, config)
        np.testing.assert_allclose(uv, local[:, :2], atol=1e-12)
        np.testing.assert_allclose(height, local[:, 2], atol=1e-12)
        np.testing.assert_array_equal(points, original)

    def test_under_plane_cleanup_changes_only_local_scope(self):
        config = plane_config()
        # The first two lie within the padded footprint but outside the
        # original object column. Distant lower geometry stays untouched.
        points = world_points([
            [-1.1, 0., -.1],
            [-1.1, 0., .1],
            [-1.3, 0., -10.],
            [0., .8, -10.],
            [0., 0., -.05],
        ], config)
        original = points.copy()
        config_before = copy.deepcopy(config)
        np.testing.assert_array_equal(
            cleanup_mask(points, config), [False, True, True, True, True])
        np.testing.assert_array_equal(points, original)
        self.assertEqual(config, config_before)

    def test_unknown_column_is_limited_to_original_footprint_and_height(self):
        config = plane_config()
        points = world_points([
            [0., 0., .3],
            [.9, .4, .9],
            [0., 0., 1.1],
            [0., 0., -.05],
            [1.1, 0., .3],
            [0., .6, .3],
        ], config)
        np.testing.assert_array_equal(
            cleanup_mask(points, config), [False, False, True, True, True, True])

    def test_stationary_patch_matches_tilted_plane_and_valid_faces(self):
        config = plane_config()
        normal = np.array([1., 2., 3.])
        normal /= np.linalg.norm(normal)
        tangent = np.cross([0., 0., 1.], normal)
        tangent /= np.linalg.norm(tangent)
        axes = np.column_stack((tangent, np.cross(normal, tangent)))
        origin = np.array(config["origin"])
        config.update(normal=normal.tolist(), tangent_axes=axes.tolist(),
                      offset=float(-origin @ normal))
        config_before = copy.deepcopy(config)
        vertices, faces = table_patch(config)
        uv, height = plane_coordinates(vertices, config)
        np.testing.assert_allclose(height, config["patch_height"], atol=3e-7)
        np.testing.assert_allclose(uv, [[-1., -.5], [1., -.5], [1., .5], [-1., .5]],
                                   atol=3e-7)
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(faces.dtype, np.uint32)
        self.assertEqual(faces.shape, (2, 3))
        self.assertTrue(np.all(faces < len(vertices)))
        for face in faces:
            self.assertEqual(len(np.unique(face)), 3)
            triangle = vertices[face]
            area_vector = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            self.assertGreater(float(area_vector @ normal), 0.)
        self.assertEqual(config, config_before)
        # Motion is a separate decision: it cannot transform patch vertices.
        self.assertTrue(object_has_moved(np.eye(3), [.3, -.2, .1]))
        fixed_vertices, fixed_faces = table_patch(config)
        np.testing.assert_array_equal(fixed_vertices, vertices)
        np.testing.assert_array_equal(fixed_faces, faces)

    def test_nonorthonormal_plane_is_rejected_before_cleanup(self):
        bad_frames = [
            {"normal": [0., 0., 2.]},
            {"tangent_axes": [[1., .5], [0., 1.], [0., 0.]]},
            {"tangent_axes": [[1., 0.], [0., 1.], [.1, 0.]]},
            {"normal": [0., 0., float("nan")]},
        ]
        for change in bad_frames:
            with self.subTest(change=change):
                config = plane_config()
                config.update(change)
                with self.assertRaises(ValueError):
                    plane_coordinates(np.zeros((1, 3)), config)
                with self.assertRaises(ValueError):
                    cleanup_mask(np.zeros((1, 3)), config)

    def test_invalid_footprint_is_rejected(self):
        config = plane_config()
        config["footprint_max"] = config["footprint_min"].copy()
        with self.assertRaises(ValueError):
            cleanup_mask(np.zeros((1, 3)), config)

    def test_motion_detection_identity_translation_rotation_and_reset(self):
        identity = np.eye(3)
        zero = np.zeros(3)
        self.assertFalse(object_has_moved(identity, zero))
        self.assertFalse(object_has_moved(identity, [1e-8, 0., 0.]))
        self.assertTrue(object_has_moved(identity, [1e-5, 0., 0.]))
        angle = .1
        rotated = np.array([[np.cos(angle), -np.sin(angle), 0.],
                            [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        self.assertTrue(object_has_moved(rotated, zero))
        self.assertFalse(object_has_moved(identity, zero))
        np.testing.assert_array_equal(identity, np.eye(3))
        np.testing.assert_array_equal(zero, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
