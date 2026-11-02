"""Explicit, reversible demo priors for unobserved background.

These are presentation operations, not measured reconstruction/segmentation.
Source Gaussian arrays stay immutable. Approximate table vertices are a
stationary background layer and never belong to the movable object.
"""
from __future__ import annotations

import numpy as np


def plane_coordinates(points, config):
    normal = np.asarray(config["normal"], dtype=np.float64)
    origin = np.asarray(config["origin"], dtype=np.float64)
    axes = np.asarray(config["tangent_axes"], dtype=np.float64)
    if normal.shape != (3,) or origin.shape != (3,) or axes.shape != (3, 2):
        raise ValueError("Invalid presentation plane dimensions")
    if not np.isfinite(normal).all() or not np.isfinite(origin).all() or not np.isfinite(axes).all():
        raise ValueError("Presentation plane must be finite")
    if not np.isclose(np.linalg.norm(normal), 1) or not np.allclose(axes.T @ axes, np.eye(2), atol=1e-6) or not np.allclose(normal @ axes, 0, atol=1e-6):
        raise ValueError("Presentation plane frame must be orthonormal")
    xyz = np.asarray(points, dtype=np.float64)
    return (xyz - origin) @ axes, xyz @ normal + float(config["offset"])


def cleanup_mask(points, config):
    uv, height = plane_coordinates(points, config)
    lo, hi = np.asarray(config["footprint_min"]), np.asarray(config["footprint_max"])
    if lo.shape != (2,) or hi.shape != (2,) or not np.all(hi > lo):
        raise ValueError("Invalid footprint bounds")
    scope_margin = float(config["cleanup_scope_margin"])
    local_scope = np.all((uv >= lo - scope_margin) & (uv <= hi + scope_margin), axis=1)
    below = local_scope & (height < float(config["background_min_height"]))
    footprint = np.all((uv >= lo) & (uv <= hi), axis=1)
    # The original object column contains missed/occluded appearance clutter.
    # Hide it only in explicitly enabled demo cleanup; this is not true removal.
    unknown_column = footprint & (height > float(config["column_min_height"])) & (height < float(config["column_max_height"]))
    return ~(below | unknown_column)


def table_patch(config):
    lo, hi = np.asarray(config["footprint_min"]), np.asarray(config["footprint_max"])
    corner_uv = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    normal = np.asarray(config["normal"])
    origin = np.asarray(config["origin"])
    axes = np.asarray(config["tangent_axes"])
    vertices = origin + corner_uv @ axes.T + float(config.get("patch_height", .0003)) * normal
    return vertices.astype(np.float32), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)


def load_inferred_background(path, scene_count):
    """Load a stationary inferred layer; it never belongs to a movable object."""
    data = dict(np.load(path, allow_pickle=False))
    required = ("xyz", "rgb", "opacity", "scale", "quat", "inferred", "scene_cleared_index")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Inferred background is missing {missing}")
    count = len(data["xyz"])
    if count == 0 or any(len(data[key]) != count for key in required[:-1]):
        raise ValueError("Inferred background arrays must be nonempty and agree in length")
    shapes = {"xyz": (3,), "rgb": (3,), "scale": (3,), "quat": (4,), "opacity": (), "inferred": ()}
    if any(data[key].shape[1:] != shape for key, shape in shapes.items()):
        raise ValueError("Inferred background arrays have the wrong shape")
    if data["rgb"].dtype != np.uint8 or data["inferred"].dtype != bool:
        raise ValueError("Inferred background needs uint8 rgb and boolean inferred flags")
    if not all(np.isfinite(data[key]).all() for key in ("xyz", "scale", "quat", "opacity")):
        raise ValueError("Inferred background values must be finite")
    if (data["scale"] <= 0).any() or (data["opacity"] < 0).any() or (data["opacity"] > 1).any():
        raise ValueError("Inferred background needs positive scales and opacity in [0,1]")
    cleared = data["scene_cleared_index"]
    if not np.issubdtype(cleared.dtype, np.integer) or cleared.ndim != 1 or len(cleared) == 0:
        raise ValueError("Cleared indices must be a nonempty integer array")
    if cleared.min() < 0 or cleared.max() >= scene_count:
        raise ValueError("Cleared indices do not fit this scene")
    return data


def object_has_moved(rotation, translation, tolerance=1e-7):
    return bool(np.linalg.norm(translation) > tolerance or not np.allclose(rotation, np.eye(3), atol=tolerance, rtol=0))
