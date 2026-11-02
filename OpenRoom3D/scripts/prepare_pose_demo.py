#!/usr/bin/env python3
"""Prepare a conservative original-Gaussian object and inferred static table.

The derived selection removes under-table/large contributions for a usable
demo; it may sacrifice palm detail. Background cleanup explicitly hides unknown
original-footprint clutter. The table patch is a prior, never measured truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.geometry import pca_pose
from openroom3d.presentation import cleanup_mask, table_patch


def prepare(args):
    raw = dict(np.load(args.scene, allow_pickle=False))
    source = dict(np.load(args.object, allow_pickle=False))
    plane = json.loads(args.plane.read_text())
    indices = source["gaussian_index"]
    for key in ("xyz", "quat", "scale", "rgb", "opacity"):
        if not np.array_equal(raw[key][indices], source[key]):
            raise ValueError(f"Source object mismatch: {key}")
    normal = np.array(plane["support_plane_normal"], dtype=np.float64)
    normal_length = np.linalg.norm(normal)
    if not np.isfinite(normal_length) or normal_length < 1e-8:
        raise ValueError("Invalid support plane")
    normal /= normal_length
    offset = float(plane["support_plane_offset"]) / normal_length
    height = raw["xyz"] @ normal + offset
    colors = raw["rgb"].astype(np.float64) / 255
    tan = ((colors[:, 0] - colors[:, 2] > .08) & (colors[:, 1] - colors[:, 2] > .03)
           & (colors.mean(axis=1) > .25) & (colors.mean(axis=1) < .85))
    keep = height[indices] > args.object_min_height
    keep &= raw["scale"][indices].max(axis=1) <= args.max_scale
    keep &= ~(tan[indices] & (height[indices] < .012))
    if keep.sum() < 3:
        raise ValueError("Presentation filtering removed object")
    fields = {key: value[keep].copy() for key, value in source.items() if value.ndim > 0 and len(value) == len(indices)}
    fields["selected_mask"] = np.ones(int(keep.sum()), dtype=bool)
    support = fields.get("pose_support_mask", np.ones(int(keep.sum()), dtype=bool))
    pose = pca_pose(fields["xyz"], fields["xyz"][support])
    seed_xyz = source["xyz"][source.get("pose_support_mask", np.ones(len(indices), dtype=bool))]
    origin = seed_xyz.mean(axis=0)
    origin -= normal * (origin @ normal + offset)
    u = np.array([1., 0, 0])
    u -= normal * (u @ normal)
    if np.linalg.norm(u) < 1e-7:
        u = np.array([0., 1, 0]) - normal * normal[1]
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    tangent = np.column_stack((u, v))
    source_height = seed_xyz @ normal + offset
    base = seed_xyz[(source_height > .004) & (source_height < .03)]
    if len(base) < 50:
        raise ValueError("Too few original base points to estimate occluded footprint")
    uv = (base - origin) @ tangent
    lo, hi = np.quantile(uv, [.005, .995], axis=0)
    lo -= args.patch_margin
    hi += args.patch_margin
    all_uv = (raw["xyz"] - origin) @ tangent
    ring = np.all((all_uv > lo - .08) & (all_uv < hi + .08), axis=1) & (abs(height) < .006) & tan
    if ring.sum() < 10:
        raise ValueError("Insufficient observed table color for demo background")
    config = {"schema_version": 1, "kind": "inferred demo presentation, not recovered ground truth",
              "normal": normal.tolist(), "offset": offset, "origin": origin.tolist(), "tangent_axes": tangent.tolist(),
              "footprint_min": lo.tolist(), "footprint_max": hi.tolist(), "rgb": np.median(raw["rgb"][ring], axis=0).round().astype(int).tolist(),
              "patch_height": .0003, "cleanup_scope_margin": .35, "background_min_height": -.008,
              "column_min_height": -.008, "column_max_height": float(np.quantile(source_height, .995) + .025),
              "scene_source": str(args.scene.resolve()), "object_source": str(args.object.resolve()),
              "support_plane_source": str(args.plane.resolve()), "observed_color_points": int(ring.sum()),
              "patch_visible_only_after_motion": True, "stationary_background_layer": True,
              "schematic_table_margin": .35,
              "limitations": "Fitted table plane and constant nearby color; no true hidden texture/shadows/underside. Unknown original-footprint clutter is hidden, not reconstructed."}
    config["cleanup_hidden_source_gaussians"] = int((~cleanup_mask(raw["xyz"], config)).sum())
    vertices, faces = table_patch(config)
    config["patch_vertices_world"] = vertices.tolist()
    config["patch_faces"] = faces.tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    object_path = args.output / "obj_001_laptop.npz"
    config["prepared_object_source"] = str(object_path.resolve())
    config["compatible_object_sources"] = [str(object_path.resolve()), str(args.object.resolve())] + [str(p.resolve()) for p in args.compatible_object]
    np.savez_compressed(object_path, **fields)
    pose.update(source=str(object_path.resolve()), semantic_verification="conservative demo cleanup, not complete segmentation")
    object_path.with_suffix(".pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    metadata = {"label": "laptop", "status": "demo selection; palm detail may be incomplete", "source": str(args.object.resolve()),
                "source_count": len(indices), "selected_count": int(keep.sum()), "object_min_height": args.object_min_height,
                "max_scale": args.max_scale, "original_parameters_preserved": True, "checkpoint_modified": False,
                "limitations": "Removing displaced under-table contributions can expose palm gap; no laptop surface completion."}
    object_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "presentation.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Prepared {object_path}: {int(keep.sum())}/{len(indices)} original Gaussians; patch is inferred stationary background")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--plane", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--object-min-height", type=float, default=.002)
    p.add_argument("--max-scale", type=float, default=.035)
    p.add_argument("--patch-margin", type=float, default=.025)
    p.add_argument("--compatible-object", action="append", type=Path, default=[], help="other known selections of the same physical laptop")
    args = p.parse_args()
    if not np.isfinite(args.object_min_height) or not np.isfinite(args.max_scale) or args.max_scale <= 0 or not np.isfinite(args.patch_margin) or args.patch_margin < 0:
        p.error("Invalid presentation thresholds")
    prepare(args)
