"""Meaningful invariants for shape retention, PCA pose and Gaussian rigid edits."""
from __future__ import annotations

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from openroom3d.geometry import constrain_to_plane, gaussian_covariances, load_support_plane, pca_pose, rigid_gaussians, support_plane_pose
from openroom3d.manipulation import ObjectState
from openroom3d.shape_retention import retain_by_shape


class ShapeRetentionTests(unittest.TestCase):
    def setUp(self):
        self.corners = np.array([[x, y, z] for x in (-1., 1.) for y in (-.7, .7) for z in (-.4, .4)])
        # Interior singleton, near exterior extent overlap, far exterior noise.
        xyz = np.vstack((self.corners, [[0, 0, 0], [1.12, 0, 0], [8, 8, 8], [0, .1, 0]]))
        self.raw = {"xyz": xyz, "gaussian_index": np.arange(len(xyz)),
                    "similarity": np.array([.9] * 8 + [.4, .8, .95, .1]),
                    "scale": np.tile([.1, .02, .01], (len(xyz), 1)),
                    "quat": np.tile([1., 0, 0, 0], (len(xyz), 1)),
                    "rgb": np.full((len(xyz), 3), 70, dtype=np.uint8), "opacity": np.ones(len(xyz)) * .8}
        self.seed = {key: values[:8].copy() for key, values in self.raw.items()}

    def test_sparse_interior_retained_without_connected_neighbours(self):
        fields, pose, _ = retain_by_shape(self.raw, self.seed, padding=0, extent_sigma=2, max_scale=.2)
        ids = fields["gaussian_index"]
        self.assertIn(8, ids)
        self.assertIn(9, ids)
        self.assertNotIn(10, ids)
        self.assertNotIn(11, ids)
        self.assertEqual(fields["retention_reason"][ids == 8].item(), 1)
        self.assertEqual(fields["retention_reason"][ids == 9].item(), 2)
        for key in self.raw:
            np.testing.assert_array_equal(fields[key], self.raw[key][ids])
        self.assertEqual(pose["pose_support_points"], 8)

    def test_extent_overlap_uses_orientation(self):
        raw = {key: value.copy() for key, value in self.raw.items()}
        raw["quat"][9] = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        fields, _, _ = retain_by_shape(raw, self.seed, padding=0, extent_sigma=2, max_scale=.2)
        self.assertNotIn(9, fields["gaussian_index"])

    def test_rotation_of_scene_and_prior_preserves_selection(self):
        baseline, _, _ = retain_by_shape(self.raw, self.seed, padding=.02, extent_sigma=2, max_scale=.2)
        R = Rotation.from_euler("xyz", [24, 35, 18], degrees=True).as_matrix()
        transformed = {key: value.copy() for key, value in self.raw.items()}
        transformed["xyz"], transformed["quat"] = rigid_gaussians(self.raw["xyz"], self.raw["quat"], R, [2, 3, 4], [0, 0, 0])
        seed = {key: values[:8].copy() for key, values in transformed.items()}
        other, _, _ = retain_by_shape(transformed, seed, padding=.02, extent_sigma=2, max_scale=.2)
        np.testing.assert_array_equal(other["gaussian_index"], baseline["gaussian_index"])

    def test_reject_wrong_source_and_invalid_threshold(self):
        seed = {key: value.copy() for key, value in self.seed.items()}
        seed["xyz"][0, 0] += .02
        with self.assertRaises(ValueError):
            retain_by_shape(self.raw, seed)
        with self.assertRaises(ValueError):
            retain_by_shape(self.raw, self.seed, interior_threshold=.8, boundary_threshold=.5)


class PoseAndManipulationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.xyz = rng.normal(size=(60, 3)) * [.7, .2, .04] + [.2, -.1, -.4]
        self.quat = Rotation.random(60, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
        self.scale = np.tile([.02, .007, .001], (60, 1))
        self.data = {"xyz": self.xyz.astype(np.float32), "quat": self.quat.astype(np.float32),
                     "scale": self.scale.astype(np.float32), "rgb": np.ones((60, 3), dtype=np.uint8) * 90,
                     "opacity": np.full(60, .7, dtype=np.float32), "gaussian_index": np.arange(60)}
        self.pose = pca_pose(self.xyz)

    def test_pose_is_right_handed_and_box_contains_centers(self):
        axes = np.array(self.pose["rotation_matrix_columns"])
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(axes), 1.)
        local = (self.xyz - self.pose["obb_center"]) @ axes
        self.assertTrue(np.all(np.abs(local) <= np.array(self.pose["obb_extent"]) / 2 + 1e-12))

    def test_covariance_follows_rigid_rotation(self):
        R = Rotation.from_euler("xyz", [10, -35, 80], degrees=True).as_matrix()
        original = gaussian_covariances(self.scale, self.quat)
        _, q = rigid_gaussians(self.xyz, self.quat, R, [.1, .2, .3], self.pose["centroid"])
        updated = gaussian_covariances(self.scale, q)
        np.testing.assert_allclose(updated, R @ original @ R.T, atol=1e-12)

    def test_absolute_transform_distance_pose_and_exact_reset(self):
        state = ObjectState(self.data, self.pose)
        R = Rotation.from_euler("z", 35, degrees=True).as_matrix()
        state.set_transform(R, [.1, .2, .3])
        first = state.arrays()
        state.set_transform(R, [.1, .2, .3])
        np.testing.assert_array_equal(first["xyz"], state.arrays()["xyz"])
        np.testing.assert_allclose(np.linalg.norm(first["xyz"][1:] - first["xyz"][0], axis=1),
                                   np.linalg.norm(self.data["xyz"][1:] - self.data["xyz"][0], axis=1), atol=2e-7)
        for key in ("scale", "rgb", "opacity"):
            np.testing.assert_array_equal(first[key], self.data[key])
        np.testing.assert_allclose(state.pose()["centroid"], np.array(self.pose["centroid"]) + [.1, .2, .3])
        state.reset()
        for key in self.data:
            np.testing.assert_array_equal(state.arrays()[key], self.data[key])

    def test_bad_rotation_leaves_state_unchanged(self):
        state = ObjectState(self.data, self.pose)
        with self.assertRaises(ValueError):
            state.set_transform(np.eye(3) * 2, [0, 0, 0])
        np.testing.assert_array_equal(state.translation, np.zeros(3))
        for key in self.data:
            np.testing.assert_array_equal(state.arrays()[key], self.data[key])



class SupportPlaneTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        # A flat box resting on a tilted table: long axis along u, thin along the normal.
        self.normal = np.array([.1, -.2, 1.]) / np.linalg.norm([.1, -.2, 1.])
        u = np.cross(self.normal, [0, 0, 1.]) + [1, 0, 0]
        u -= u @ self.normal * self.normal
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u)
        local = rng.uniform(-1, 1, (400, 3)) * [.3, .2, .02] + [0, 0, .02]
        self.xyz = local @ np.stack([u, v, self.normal])
        self.u = u

    def test_third_axis_is_normal_and_first_is_long_side(self):
        pose = support_plane_pose(self.xyz, self.normal)
        axes = np.array(pose["rotation_matrix_columns"])
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(axes), 1.)
        np.testing.assert_allclose(axes[:, 2], self.normal, atol=1e-12)
        self.assertGreater(abs(axes[:, 0] @ self.u), .99)
        local = (self.xyz - pose["obb_center"]) @ axes
        self.assertTrue(np.all(np.abs(local) <= np.array(pose["obb_extent"]) / 2 + 1e-12))
        self.assertFalse(pose["orientation_ambiguous"])

    def test_constraint_keeps_spin_about_normal_and_drops_tilt_and_lift(self):
        spin = Rotation.from_rotvec(self.normal * .7).as_matrix()
        rotation, translation = constrain_to_plane(spin, self.u * .1 + self.normal * .05, self.normal)
        np.testing.assert_allclose(rotation, spin, atol=1e-10)
        np.testing.assert_allclose(translation, self.u * .1, atol=1e-12)
        tilt = Rotation.from_rotvec(self.u * .4).as_matrix()
        rotation, _ = constrain_to_plane(tilt @ spin, np.zeros(3), self.normal)
        np.testing.assert_allclose(rotation @ self.normal, self.normal, atol=1e-10)
        flip = Rotation.from_rotvec(self.u * np.pi).as_matrix()
        np.testing.assert_allclose(constrain_to_plane(flip, np.zeros(3), self.normal)[0], np.eye(3), atol=1e-10)

    def test_plane_files_in_both_formats_agree(self):
        import tempfile
        from pathlib import Path
        point = np.array([.1, .2, -.3])
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory, "a.npz"), Path(directory, "b.npz")
            np.savez(a, normal=self.normal, center=point)
            np.savez(b, n=self.normal, d=self.normal @ point)
            (na, pa), (nb, pb) = load_support_plane(a), load_support_plane(b)
        np.testing.assert_allclose(na, nb)
        self.assertAlmostEqual(float((pa - pb) @ na), 0., places=12)

if __name__ == "__main__":
    unittest.main()
