#!/usr/bin/env python3
"""Repair a planar surface from an explicit, scene-specific image recipe.

This is image-guided geometry completion, NOT a better semantic mask. It reuses
source Gaussian slots but changes their means, scales, rotations, RGB and
opacity in an independent object export. Original attributes and image pixels
are recorded. The training checkpoint is never edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def repair(recipe_path, output):
    recipe = json.loads(recipe_path.read_text())
    raw = np.load(recipe["raw"], allow_pickle=False)
    seed = np.load(recipe["seed"], allow_pickle=False)
    cleaned = np.load(recipe["cleaned"], allow_pickle=False)
    support = json.loads(Path(recipe["support_plane_metadata"]).read_text())
    xyz = raw["xyz"]
    if not np.array_equal(raw["gaussian_index"], np.arange(len(xyz))):
        raise ValueError("Raw export must contain the complete scene in checkpoint order")
    for source in (seed, cleaned):
        if not np.array_equal(source["xyz"], xyz[source["gaussian_index"]]):
            raise ValueError("Seed or cleaned export does not match source scene")
    normal = np.array(support["support_plane_normal"])
    offset = support["support_plane_offset"]
    if not np.isclose(np.linalg.norm(normal), 1):
        raise ValueError("Support-plane normal must be normalized")
    height = xyz @ normal + offset
    base_ids = cleaned["gaussian_index"]
    lower, upper = recipe["base_height_band"]
    band = ((height[base_ids] > lower) & (height[base_ids] < upper)
            & (raw["rgb"][base_ids].mean(axis=1) < 102))
    if not band.any():
        raise ValueError("No supported base plane")
    target_height = float(np.median(height[base_ids][band]))
    width, height_pixels = recipe["reference_width"], recipe["reference_height"]
    step = recipe["pixel_step"]
    if step <= 0:
        raise ValueError("Pixel step must be positive")
    yy, xx = np.mgrid[0:height_pixels:step, 0:width:step]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    inside = Polygon(recipe["palm_polygon"]).contains_points(pixels)
    pixels = pixels[inside]
    with Image.open(recipe["source_comparison"]) as image:
        photo = np.asarray(image.crop(recipe["photo_crop"])).astype(np.float64) / 255
        isolated = np.asarray(image.crop(recipe["isolated_crop"])).astype(np.float64) / 255
    if photo.shape[:2] != (height_pixels, width) or isolated.shape != photo.shape:
        raise ValueError("Reference crops do not match specified camera resolution")
    colors = photo[pixels[:, 1], pixels[:, 0]]
    keep = ((colors.mean(axis=1) < recipe["source_dark_limit"])
            & (isolated[pixels[:, 1], pixels[:, 0]].mean(axis=1) > recipe["white_selection_limit"]))
    pixels, colors = pixels[keep], colors[keep]
    view = np.array(recipe["view_matrix"])
    intrinsics = np.array(recipe["intrinsics"])
    camera = -view[:3, :3].T @ view[:3, 3]

    def intersect(image_pixels):
        rays = (np.column_stack((image_pixels, np.ones(len(image_pixels))))
                @ np.linalg.inv(intrinsics).T) @ view[:3, :3]
        denominator = rays @ normal
        if np.any(np.abs(denominator) < 1e-8):
            raise ValueError("Reference rays parallel to repair plane")
        distance = (target_height - (camera @ normal + offset)) / denominator
        if np.any(distance <= 0):
            raise ValueError("Repair plane lies behind reference camera")
        return camera + rays * distance[:, None]

    positions = intersect(pixels)
    neighbor_positions = intersect(pixels + np.array((step, 0)))
    sizes = np.linalg.norm(neighbor_positions - positions, axis=1) * recipe["scale_spacing_ratio"]
    surface_tree = cKDTree(cleaned["xyz"][height[base_ids] < .026])
    good = surface_tree.query(positions)[0] < recipe["max_surface_distance"]
    positions, sizes, colors, pixels = positions[good], sizes[good], colors[good], pixels[good]
    if len(positions) == 0:
        raise ValueError("No repair samples survived local surface constraint")
    distance = cKDTree(seed["xyz"]).query(xyz)[0]
    pool = np.flatnonzero((distance < recipe["source_pool_radius"])
                          & (height < recipe["source_pool_above_height"])
                          & (height > recipe["source_pool_below_height"])
                          & (raw["similarity"] > recipe["source_pool_score_floor"]))
    pool = pool[~np.isin(pool, base_ids)]
    if len(pool) < len(positions):
        raise ValueError("Insufficient source Gaussian slots for measured repair samples")
    indices = np.concatenate((base_ids, pool[:len(positions)]))
    fields = {key: raw[key][indices].copy() for key in raw.files
              if raw[key].ndim > 0 and len(raw[key]) == len(xyz)}
    for key in ("xyz", "scale", "quat", "rgb", "opacity"):
        fields["source_" + key] = fields[key].copy()
    tangent = np.cross(normal, (0, 1, 0))
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.cross(normal, (1, 0, 0))
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    rotation = np.column_stack((tangent, bitangent, normal))
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]  # gsplat wxyz
    start = len(base_ids)
    fields["xyz"][start:] = positions
    fields["scale"][start:] = np.column_stack((sizes, sizes, np.full(len(sizes), recipe["plane_thickness"])))
    fields["quat"][start:] = quat
    fields["rgb"][start:] = (colors * 255).round().astype(np.uint8)
    fields["opacity"][start:] = recipe["patch_opacity"]
    fields["selected_mask"] = np.ones(len(indices), dtype=bool)
    fields["surface_repair_mask"] = np.arange(len(indices)) >= start
    fields["reference_pixel"] = np.vstack((np.full((start, 2), -1), pixels)).astype(np.int32)
    fields["semantic_score_valid_after_repair"] = ~fields["surface_repair_mask"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **fields)
    metadata = {"method": recipe["method"], "recipe": str(recipe_path.resolve()),
                "object_id": output.stem, "label": "laptop", "coordinate_system": "reconstruction units, not meters",
                "num_gaussians": len(indices), "unmodified_geometry_gaussians": start,
                "surface_repaired_gaussians": len(positions), "repair_height_above_support": target_height,
                "reference_camera_index": recipe["camera_index"], "checkpoint_modified": False,
                "status": "pending_render_review", "manual_input": "scene-specific palm image polygon and camera recipe",
                "semantic_note": "Scores on repaired slots are source values, not predictions for new surface; validity mask provided.",
                "limitations": "Planar assumption, source-view fitting, incomplete backside coverage and remaining boundary artifacts."}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output}: {len(positions)} repaired surface slots / {len(indices)} Gaussians")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repair(args.recipe, args.output)
