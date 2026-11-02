"""Keep weak/sparse Gaussian contributors supported by a seed's bounding shape.

A box is a selection prior, not an assertion that an object is a solid cuboid.
Projected ellipsoid extents provide conservative box-overlap support, not exact
ellipsoid-box intersection. No parameter is recolored, moved, or synthesized.
"""
from __future__ import annotations

import numpy as np
from .geometry import pca_pose, quaternion_matrices


def retain_by_shape(raw, seed, padding=.01, interior_threshold=.35,
                    boundary_threshold=.60, extent_sigma=1., max_scale=.06):
    for name, value in (("padding", padding), ("extent_sigma", extent_sigma), ("max_scale", max_scale)):
        if not np.isfinite(value) or value < 0 or (name == "max_scale" and value == 0):
            raise ValueError(f"Invalid {name}")
    if not (0 <= interior_threshold <= boundary_threshold <= 1):
        raise ValueError("Thresholds must satisfy 0 <= interior <= boundary <= 1")
    xyz = np.asarray(raw["xyz"], dtype=np.float64)
    index = np.asarray(raw["gaussian_index"])
    if not np.array_equal(index, np.arange(len(xyz))):
        raise ValueError("Raw scene must be complete and in checkpoint order")
    seed_ids = np.asarray(seed["gaussian_index"])
    if not np.issubdtype(seed_ids.dtype, np.integer) or seed_ids.ndim != 1 or len(np.unique(seed_ids)) != len(seed_ids):
        raise ValueError("Seed indices must be unique integers")
    if len(seed_ids) < 3 or seed_ids.min() < 0 or seed_ids.max() >= len(xyz):
        raise ValueError("Seed indices are invalid")
    if not np.array_equal(np.asarray(raw["xyz"])[seed_ids], seed["xyz"]):
        raise ValueError("Seed does not belong to this source model")
    if not np.isfinite(xyz).all():
        raise ValueError("Scene positions must be finite")
    scores = np.asarray(raw["similarity"])
    if scores.shape != (len(xyz),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected finite probabilities [N] in [0,1]")
    pose = pca_pose(seed["xyz"])
    axes = np.array(pose["rotation_matrix_columns"])
    center, half_extent = np.array(pose["obb_center"]), np.array(pose["obb_extent"]) / 2
    local = (xyz - center) @ axes
    scales = np.asarray(raw["scale"], dtype=np.float64)
    rotations = quaternion_matrices(raw["quat"])
    if scales.shape != (len(xyz), 3) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid scales")
    # sqrt(diag(A^T Sigma A)): Gaussian extent along each box axis.
    local_rotations = np.einsum("ab,nbc->nac", axes.T, rotations)
    projected_sigma = np.sqrt(((local_rotations * scales[:, None, :]) ** 2).sum(axis=2))
    center_inside = np.all(np.abs(local) <= half_extent + padding, axis=1)
    extent_overlaps = np.all(np.abs(local) <= half_extent + padding + extent_sigma * projected_sigma, axis=1)
    quality = scales.max(axis=1) <= max_scale
    core_keep = center_inside & (scores >= interior_threshold) & quality
    fringe_keep = extent_overlaps & ~center_inside & (scores >= boundary_threshold) & quality
    seed_mask = np.zeros(len(xyz), dtype=bool)
    seed_mask[seed_ids] = True
    selected = seed_mask | core_keep | fringe_keep
    # Retained contributions bypass DBSCAN noise removal inside the object prior.
    # Keep the original seed as pose support; all retained points bound the OBB.
    fields = {key: np.asarray(raw[key])[selected].copy() for key in raw.keys()
              if np.asarray(raw[key]).ndim > 0 and len(raw[key]) == len(xyz)}
    fields["selected_mask"] = np.ones(int(selected.sum()), dtype=bool)
    fields["pose_support_mask"] = seed_mask[selected]
    fields["retention_reason"] = np.where(seed_mask[selected], 0,
                                         np.where(center_inside[selected], 1, 2)).astype(np.uint8)
    result_pose = pca_pose(fields["xyz"], seed["xyz"])
    metadata = {"method": "shape-constrained retention of original Gaussian parameters",
                "parameters": {"padding": padding, "interior_threshold": interior_threshold,
                               "boundary_threshold": boundary_threshold, "extent_sigma": extent_sigma, "max_scale": max_scale},
                "counts": {"seed": len(seed_ids), "selected": int(selected.sum()),
                           "interior_added": int((core_keep & ~seed_mask).sum()),
                           "extent_overlap_added": int((fringe_keep & ~seed_mask).sum())},
                "reason_codes": {"0": "original seed", "1": "center inside shape", "2": "Gaussian extent overlaps shape"},
                "shape_prior": pose, "pose_support": "original seed only; all centers determine OBB extent",
                "gaussian_parameters_modified": False, "checkpoint_modified": False,
                "status": "pending_isolated_render_review",
                "limitations": "Box can include background/contact surfaces; sparse viewpoint-dependent contributors may still spill. No fabricated surface."}
    return fields, result_pose, metadata
