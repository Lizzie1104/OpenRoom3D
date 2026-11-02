#!/usr/bin/env python3
"""CUDA diagnostic for cleaned background and inferred stationary table.

The browser uses a mesh for the table; this offline check samples the same
plane with explicitly synthetic thin Gaussians. Neither is measured geometry.
No source model or original Gaussian parameters are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.geometry import rigid_gaussians
from openroom3d.presentation import cleanup_mask


def sampled_patch(config, spacing=.004):
    lo, hi = np.array(config["footprint_min"]), np.array(config["footprint_max"])
    u = np.linspace(lo[0], hi[0], max(2, int(np.ceil((hi[0] - lo[0]) / spacing)) + 1))
    v = np.linspace(lo[1], hi[1], max(2, int(np.ceil((hi[1] - lo[1]) / spacing)) + 1))
    xx, yy = np.meshgrid(u, v)
    uv = np.column_stack((xx.ravel(), yy.ravel()))
    axes, normal = np.array(config["tangent_axes"]), np.array(config["normal"])
    xyz = np.array(config["origin"]) + uv @ axes.T + config["patch_height"] * normal
    matrix = np.column_stack((axes, normal))
    quat = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    return {"xyz": xyz.astype(np.float32), "quat": np.tile(quat, (len(xyz), 1)).astype(np.float32),
            "scale": np.tile([spacing * .9, spacing * .9, .00005], (len(xyz), 1)).astype(np.float32),
            "opacity": np.full(len(xyz), .98, dtype=np.float32),
            "rgb": np.tile(config["rgb"], (len(xyz), 1)).astype(np.uint8)}


def combine(*groups):
    return {key: np.concatenate([group[key] for group in groups], axis=0) for key in ("xyz", "quat", "scale", "opacity", "rgb")}


def render_arrays(arrays, view):
    quaternion = torch.as_tensor(arrays["quat"], device="cuda")
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True)
    height, width = view["mask"].shape
    with torch.no_grad():
        rgb, _, _ = rasterization(means=torch.as_tensor(arrays["xyz"], device="cuda"), quats=quaternion,
                                 scales=torch.as_tensor(arrays["scale"], device="cuda"),
                                 opacities=torch.as_tensor(arrays["opacity"], device="cuda"),
                                 colors=torch.as_tensor(arrays["rgb"], device="cuda", dtype=torch.float32) / 255,
                                 viewmats=torch.as_tensor(view["viewmat"], device="cuda")[None],
                                 Ks=torch.as_tensor(view["intrinsics"], device="cuda")[None],
                                 width=width, height=height, packed=False, render_mode="RGB",
                                 backgrounds=torch.ones((1, 3), device="cuda"))
    return rgb[0].cpu().numpy()


def render(args):
    raw = dict(np.load(args.scene, allow_pickle=False))
    object_data = dict(np.load(args.object, allow_pickle=False))
    config = json.loads(args.presentation.read_text())
    pose = json.loads(args.object.with_suffix(".pose.json").read_text())
    if Path(config["scene_source"]).resolve() != args.scene.resolve():
        raise ValueError("Presentation config source mismatch")
    for key in ("xyz", "quat", "scale", "rgb", "opacity"):
        if not np.array_equal(object_data[key], raw[key][object_data["gaussian_index"]]):
            raise ValueError("Object must retain original source parameters")
    remaining = np.ones(len(raw["xyz"]), dtype=bool)
    remaining[object_data["gaussian_index"]] = False
    background = {key: value[remaining & cleanup_mask(raw["xyz"], config)] for key, value in raw.items()
                  if value.ndim > 0 and len(value) == len(remaining)}
    normal = np.array(config["normal"])
    tangent = np.array(config["tangent_axes"])[:, 0]
    translation = normal * .05 + tangent * .10
    rotation = Rotation.from_rotvec(normal * np.deg2rad(20)).as_matrix()
    moved = {key: value.copy() for key, value in object_data.items()}
    moved["xyz"], moved["quat"] = rigid_gaussians(moved["xyz"], moved["quat"], rotation, translation, pose["centroid"])
    moved["xyz"], moved["quat"] = moved["xyz"].astype(np.float32), moved["quat"].astype(np.float32)
    patch = sampled_patch(config)
    schematic_config = dict(config)
    margin = float(config.get("schematic_table_margin", .35))
    schematic_config["footprint_min"] = (np.array(config["footprint_min"]) - margin).tolist()
    schematic_config["footprint_max"] = (np.array(config["footprint_max"]) + margin).tolist()
    schematic = sampled_patch(schematic_config)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index in args.camera_index:
        view = dict(np.load(args.masks / f"camera_{index:03d}.npz", allow_pickle=False))
        original = render_arrays(raw, view)
        moved_without = render_arrays(combine(background, moved), view)
        moved_with = render_arrays(combine(background, patch, moved), view)
        simplified_initial = render_arrays(combine(schematic, object_data), view)
        simplified_moved = render_arrays(combine(schematic, moved), view)
        np.savez_compressed(args.output / f"camera_{index:03d}.npz", original=original, moved_without_patch=moved_without,
                            moved_with_inferred_patch=moved_with, simplified_initial=simplified_initial,
                            simplified_moved=simplified_moved)
        height, width = original.shape[:2]
        canvas = Image.new("RGB", (width * 3, height + 35), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (image, label) in enumerate(zip((original, moved_without, moved_with),
                                               ("Original full scene", "Cleaned / moved / no inferred table", "Cleaned / moved / inferred table"))):
            canvas.paste(Image.fromarray((image.clip(0, 1) * 255).round().astype(np.uint8)), (col * width, 35))
            draw.text((col * width + 5, 8), label, fill="black")
        filename = f"camera_{index:03d}_demo.jpg"
        canvas.save(args.output / filename)
        simple_canvas = Image.new("RGB", (width * 2, height + 35), "white")
        simple_draw = ImageDraw.Draw(simple_canvas)
        for col, (image, label) in enumerate(((simplified_initial, "Schematic table / initial object"),
                                            (simplified_moved, "Same fixed table / moved object"))):
            simple_canvas.paste(Image.fromarray((image.clip(0, 1) * 255).round().astype(np.uint8)), (col * width, 35))
            simple_draw.text((col * width + 5, 8), label, fill="black")
        simple_name = f"camera_{index:03d}_simplified.jpg"
        simple_canvas.save(args.output / simple_name)
        results.append({"camera_index": index, "comparison": filename, "simplified_comparison": simple_name})
        print(f"saved {args.output / filename}", flush=True)
    metadata = {"type": "demo presentation diagnostic, not reconstruction accuracy",
                "source": str(args.scene.resolve()), "object": str(args.object.resolve()),
                "presentation": str(args.presentation.resolve()), "views": results,
                "synthetic_patch_gaussians": len(patch["xyz"]), "browser_patch": "independent stationary mesh",
                "translation": translation.tolist(), "rotation": rotation.tolist(), "background_layer_fixed": True,
                "source_object_parameters_modified": False, "checkpoint_modified": False,
                "limitations": "CUDA patch sample versus browser mesh differs; no true hidden texture and object details may be missing."}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--presentation", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera-index", nargs="+", type=int, default=[0, 30, 60])
    render(p.parse_args())
