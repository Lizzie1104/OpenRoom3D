#!/usr/bin/env python3
"""Scene-tuned recovery of a dark laptop above a tan support plane.

Uses existing Gaussian parameters only. Color assumptions and thresholds are
explicit: this is not a universal automatic object segmenter. Render review is
required, especially if object and table colors/heights are similar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def fit_support_plane(points, threshold, iterations=500):
    if len(points) < 30:
        raise ValueError("Insufficient local tan table points; support-plane assumptions do not hold")
    rng = np.random.default_rng(19)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        offset = -sample[0] @ normal
        mask = np.abs(points @ normal + offset) < threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 30:
        raise ValueError("Failed to fit a supported plane")
    centroid = points[best].mean(axis=0)
    _, _, vectors = np.linalg.svd(points[best] - centroid, full_matrices=False)
    normal = vectors[-1]
    return normal, -centroid @ normal, int(best.sum())


def recover(args):
    raw = np.load(args.raw, allow_pickle=False)
    seed = np.load(args.seed, allow_pickle=False)
    xyz = raw["xyz"].astype(np.float64)
    ids = raw["gaussian_index"]
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Input must be a complete scene export indexed in checkpoint order")
    seed_ids = seed["gaussian_index"]
    if not np.array_equal(raw["xyz"][seed_ids], seed["xyz"]):
        raise ValueError("Seed and raw source disagree")
    distance = cKDTree(seed["xyz"]).query(xyz)[0]
    rgb = raw["rgb"].astype(np.float64) / 255
    brightness = rgb.mean(axis=1)
    tan = ((rgb[:, 0] - rgb[:, 2] > .08) & (rgb[:, 1] - rgb[:, 2] > .03)
           & (brightness > .25) & (brightness < .85))
    local_table = (distance < args.plane_radius) & tan
    normal, offset, inliers = fit_support_plane(xyz[local_table], args.plane_threshold)
    if np.median(seed["xyz"] @ normal + offset) < 0:
        normal, offset = -normal, -offset
    height = xyz @ normal + offset
    pose = json.loads(args.pose.read_text())
    if Path(pose["source"]).resolve() != args.seed.resolve():
        raise ValueError("Pose must refer to seed export")
    rotation = np.array(pose["rotation_matrix_columns"])
    local = (xyz - np.array(pose["obb_center"])) @ rotation
    inside_box = np.all(np.abs(local) <= np.array(pose["obb_extent"]) / 2 + args.box_padding, axis=1)
    seed_mask = np.zeros(len(xyz), dtype=bool)
    seed_mask[seed_ids] = True
    recovered = ((distance <= args.recovery_radius) & inside_box & (brightness <= args.dark_limit)
                 & ~tan & (raw["similarity"] >= args.score_floor))
    selected = (seed_mask | recovered) & (height > args.support_margin)
    selected &= ~(tan & (height < args.support_margin + .008))
    selected &= raw["scale"].max(axis=1) <= args.max_scale
    if selected.sum() < 3:
        raise ValueError("No viable selection after support-plane filtering")
    fields = {key: raw[key][selected] for key in raw.files
              if raw[key].ndim > 0 and len(raw[key]) == len(xyz)}
    fields["selected_mask"] = np.ones(int(selected.sum()), dtype=bool)
    # No synthetic DBSCAN label: this selection is seed + explicit geometry recovery.
    fields["recovered_from_scene"] = (~seed_mask[selected]).astype(bool)
    fields["support_height"] = height[selected].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    metadata = {
        "label": "laptop", "method": "seed-guided dark-object recovery above tan support plane",
        "raw_source": str(args.raw.resolve()), "seed_source": str(args.seed.resolve()),
        "parameters": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "support_plane_normal": normal.tolist(), "support_plane_offset": float(offset),
        "support_plane_candidate_points": int(local_table.sum()), "support_plane_inliers": inliers,
        "counts": {"seed": int(seed_mask.sum()), "selected": int(selected.sum()),
                   "added": int((selected & ~seed_mask).sum()), "removed": int((seed_mask & ~selected).sum())},
        "coordinate_system": "reconstruction units, not meters", "status": "pending_render_review",
        "limitations": "Dark laptop, tan planar table and scene-specific thresholds; no universal segmentation claim.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.output}: {metadata['counts']}")
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--pose", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--support-margin", type=float, default=.004)
    p.add_argument("--recovery-radius", type=float, default=.025)
    p.add_argument("--dark-limit", type=float, default=.45)
    p.add_argument("--score-floor", type=float, default=.25)
    p.add_argument("--max-scale", type=float, default=.035)
    p.add_argument("--box-padding", type=float, default=.015)
    p.add_argument("--plane-radius", type=float, default=.08)
    p.add_argument("--plane-threshold", type=float, default=.003)
    args = p.parse_args()
    for key in ("recovery_radius", "max_scale", "plane_radius", "plane_threshold"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be finite and positive")
    if not 0 <= args.dark_limit <= 1 or not 0 <= args.score_floor <= 1 or args.support_margin < 0 or args.box_padding < 0:
        p.error("invalid color, score, support margin or padding")
    recover(args)
