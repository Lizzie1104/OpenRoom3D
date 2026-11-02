"""Geometry helpers; arrays use world-space row vectors and wxyz quaternions."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_matrices(quaternions):
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all() or (np.linalg.norm(q, axis=1) < 1e-12).any():
        raise ValueError("Expected finite nonzero wxyz quaternions [N,4]")
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def gaussian_covariances(scales, quaternions):
    scales = np.asarray(scales, dtype=np.float64)
    rotations = quaternion_matrices(quaternions)
    if scales.shape != (len(rotations), 3) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Expected finite positive Gaussian scales [N,3]")
    factors = rotations * scales[:, None, :]
    return factors @ factors.swapaxes(1, 2)


def pca_pose(points, support_points=None):
    """Centroid/PCA frame, plus an OBB containing every supplied center.

    Optional support_points determine the frame, while all points determine
    the box. This stops sparse recovered contributors from changing the axes.
    PCA axis signs follow a deterministic convention, not object semantics.
    """
    points = np.asarray(points, dtype=np.float64)
    support = points if support_points is None else np.asarray(support_points, dtype=np.float64)
    for array in (points, support):
        if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3 or not np.isfinite(array).all():
            raise ValueError("Pose requires at least three finite xyz points")
    centroid = support.mean(axis=0)
    centered = support - centroid
    values, vectors = np.linalg.eigh(centered.T @ centered / (len(support) - 1))
    order = np.argsort(values)[::-1]
    values, axes = values[order], vectors[:, order]
    for i in range(2):
        dominant = np.argmax(np.abs(axes[:, i]))
        if axes[dominant, i] < 0:
            axes[:, i] *= -1
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    projected = (points - centroid) @ axes
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = axes, centroid
    gaps = np.abs(np.diff(values)) / max(float(values[0]), 1e-12)
    return {"pose_type": "geometry-derived local object pose",
            "coordinate_system": "normalized reconstruction units; not meters",
            "centroid": centroid.tolist(), "rotation_matrix_columns": axes.tolist(),
            "homogeneous_transform": transform.tolist(), "principal_variances": values.tolist(),
            "obb_center": (centroid + axes @ ((lo + hi) / 2)).tolist(), "obb_extent": (hi - lo).tolist(),
            "num_points": len(points), "pose_support_points": len(support),
            "axis_sign_convention": "dominant component positive for first two axes, right handed third axis",
            "relative_eigenvalue_gaps": gaps.tolist(), "orientation_ambiguous": bool((gaps < .1).any()),
            "pca_direction_ambiguity": True}


def box_edges(center, rotation, extent):
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    corners = np.asarray(center) + (signs * np.asarray(extent) / 2) @ np.asarray(rotation).T
    return np.array([[corners[i], corners[j]] for i in range(8) for j in range(i + 1, 8)
                     if np.count_nonzero(signs[i] != signs[j]) == 1])


def rigid_gaussians(xyz, quaternions, rotation, translation, pivot):
    """Rotate means about pivot; left-compose Gaussian orientation in world frame."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation, pivot = np.asarray(translation, dtype=np.float64), np.asarray(pivot, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7):
        raise ValueError("Rotation must be a finite proper rotation [3,3]")
    if translation.shape != (3,) or pivot.shape != (3,) or not np.isfinite(translation).all() or not np.isfinite(pivot).all():
        raise ValueError("Translation and pivot must be finite [3]")
    matrices = quaternion_matrices(quaternions)
    positions = (np.asarray(xyz) - pivot) @ rotation.T + pivot + translation
    q = Rotation.from_matrix(rotation @ matrices).as_quat()[:, [3, 0, 1, 2]]
    return positions, q
