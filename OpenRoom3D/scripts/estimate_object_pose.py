#!/usr/bin/env python3
"""Estimate a geometry-derived local pose from a refined Gaussian NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def estimate(args: argparse.Namespace) -> Path:
    data = np.load(args.input)
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    mask = np.asarray(data["selected_mask"], dtype=bool)
    if "cluster_label" in data:
        clusters = np.unique(data["cluster_label"][mask])
        if len(clusters) != 1 or clusters[0] < 0:
            raise ValueError("Pose requires one verified candidate cluster. Use inspect_query_objects.py to split candidates first.")
    if not mask.any():
        raise ValueError("selected_mask contains no points")
    points = xyz[mask]
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Pose requires at least three finite points")
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1
    projected = centered @ rotation
    extent = projected.max(axis=0) - projected.min(axis=0)
    obb_center = center + rotation @ ((projected.max(axis=0) + projected.min(axis=0)) / 2)
    output = args.output or args.input.with_suffix(".pose.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pose_type": "geometry-derived local object pose",
        "semantic_verification": "not performed by this script",
        "coordinate_system": "reconstruction coordinate system",
        "source": str(args.input.resolve()),
        "num_points": int(len(points)),
        "centroid": center.tolist(),
        "rotation_matrix_columns": rotation.tolist(),
        "homogeneous_transform": np.vstack((np.column_stack((rotation, center)), [0.0, 0.0, 0.0, 1.0])).tolist(),
        "principal_variances": eigenvalues.tolist(),
        "obb_extent": extent.tolist(),
        "obb_center": obb_center.tolist(),
        "pca_direction_ambiguity": True,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="refined NPZ")
    parser.add_argument("--output", type=Path)
    estimate(parser.parse_args())
