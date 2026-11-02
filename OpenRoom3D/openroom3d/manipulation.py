"""Reversible, absolute object transforms. Never accumulate from prior clicks."""
from __future__ import annotations

import copy
import numpy as np
from .geometry import rigid_gaussians


class ObjectState:
    def __init__(self, arrays, pose):
        self.original = {key: np.array(value, copy=True) for key, value in arrays.items()}
        self.original_pose = copy.deepcopy(pose)
        self.pivot = np.asarray(pose["centroid"], dtype=np.float64)
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)

    def set_transform(self, rotation, translation):
        # Validate before changing state; invalid input leaves previous transform.
        rigid_gaussians(self.original["xyz"][:1], self.original["quat"][:1], rotation, translation, self.pivot)
        self.rotation = np.asarray(rotation, dtype=np.float64).copy()
        self.translation = np.asarray(translation, dtype=np.float64).copy()

    def reset(self):
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)

    def arrays(self):
        result = {key: value.copy() for key, value in self.original.items()}
        if np.array_equal(self.rotation, np.eye(3)) and np.array_equal(self.translation, np.zeros(3)):
            return result
        xyz, quat = rigid_gaussians(result["xyz"], result["quat"], self.rotation, self.translation, self.pivot)
        # Provenance fields identify original checkpoint even after a rigid edit.
        for key in ("xyz", "quat", "scale", "rgb", "opacity"):
            result.setdefault("source_" + key, result[key].copy())
        result["xyz"] = xyz.astype(result["xyz"].dtype)
        result["quat"] = quat.astype(result["quat"].dtype)
        return result

    def pose(self):
        pose = copy.deepcopy(self.original_pose)
        centroid = self.pivot + self.translation
        axes = self.rotation @ np.array(pose["rotation_matrix_columns"])
        box_center = (np.array(pose["obb_center"]) - self.pivot) @ self.rotation.T + self.pivot + self.translation
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = axes, centroid
        pose.update(centroid=centroid.tolist(), rotation_matrix_columns=axes.tolist(), obb_center=box_center.tolist(),
                    homogeneous_transform=matrix.tolist(), edit_rotation_world=self.rotation.tolist(),
                    edit_translation_world=self.translation.tolist(), edit_pivot_original=self.pivot.tolist(),
                    geometry_edit_type="rigid object transformation; no surface completion")
        return pose
