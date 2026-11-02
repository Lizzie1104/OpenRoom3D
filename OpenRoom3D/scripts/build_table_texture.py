#!/usr/bin/env python3
"""Fuse a top-down table texture from photos and mark the never-observed hole.

Only source photos supply texels here; nothing is inpainted. Texels are kept when
a view sees them unoccluded and outside the dilated object mask. The object volume
is the multi-view visual hull, so hidden Gaussians inside it are cleared too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openroom3d.gaussian_scene import load_cameras, load_gaussians, render
from select_by_masks import hull_support, load_mask


def expected_depth(gaussians, result, c2w, width, height):
    """Alpha-weighted depth along the camera's forward axis."""
    forward = torch.as_tensor(-c2w[:3, 2], device=result["weight"].device, dtype=torch.float32)
    origin = torch.as_tensor(c2w[:3, 3], device=result["weight"].device, dtype=torch.float32)
    z = (gaussians["means"][result["gaussian"]] - origin) @ forward
    total = torch.zeros(width * height, device=z.device)
    total.index_add_(0, result["pixel"], result["weight"] * z)
    return (total / (result["alpha"].reshape(-1) + 1e-6)).reshape(height, width)


def fit_plane(points, tolerance=.004, iterations=3000):
    generator = torch.Generator(device=points.device).manual_seed(0)
    best, best_count = None, 0
    for _ in range(iterations):
        sample = points[torch.randint(len(points), (3,), device=points.device, generator=generator)]
        normal = torch.linalg.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = normal.norm()
        if length < 1e-9:
            continue
        normal = normal / length
        inliers = ((points - sample[0]) @ normal).abs() < tolerance
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    center = points[best].mean(0)
    normal = torch.linalg.svd(points[best] - center, full_matrices=False)[2][-1]
    return center, normal, best_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True, help="selected object NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--texel", type=float, default=.0015, help="reconstruction units per texel")
    parser.add_argument("--margin", type=float, default=.2, help="context around the object footprint")
    parser.add_argument("--hull", type=float, default=.8)
    parser.add_argument("--depth-tolerance", type=float, default=.03)
    parser.add_argument("--mask-dilate", type=int, default=4)
    parser.add_argument("--min-views", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gaussians = load_gaussians(args.run)
    device = gaussians["means"].device
    cameras, intrinsics, distortion, (full_width, full_height) = load_cameras(args.data, args.run)
    factor = args.width / full_width
    height = int(round(full_height * factor))
    scaled = intrinsics.copy()
    scaled[:2] *= factor
    train = [row for row in enumerate(cameras) if row[1]["split"] == "train"]
    means = gaussians["means"].detach()
    obj = np.load(args.object, allow_pickle=False)
    selected = torch.zeros(len(means), dtype=torch.bool, device=device)
    selected[torch.as_tensor(obj["gaussian_index"], device=device)] = True
    support, seen = hull_support(means, train, scaled, args.width, height, args.masks, 1)
    hull = (support >= args.hull) & (seen >= 5)

    # Fit the table locally from visible surface pixels around the object.
    samples = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for index, camera in train[::6]:
        mask = load_mask(args.masks, index, args.width, height, device).astype(np.uint8)
        if not mask.any():
            continue
        result = render(gaussians, camera["c2w"], scaled, args.width, height)
        depth = expected_depth(gaussians, result, camera["c2w"], args.width, height)
        ring = cv2.dilate(mask, kernel, iterations=40).astype(bool) & ~cv2.dilate(mask, kernel, iterations=6).astype(bool)
        rgb = result["rgb"]
        tan = (rgb[..., 0] > rgb[..., 2] + .12) & (rgb[..., 0] > .55) & torch.as_tensor(ring, device=device)
        v, u = tan.nonzero(as_tuple=True)
        K = torch.as_tensor(scaled, device=device, dtype=torch.float32)
        rays = torch.stack([(u + .5 - K[0, 2]) / K[0, 0], -(v + .5 - K[1, 2]) / K[1, 1], -torch.ones_like(u, dtype=torch.float32)], -1)
        c2w = torch.as_tensor(camera["c2w"], device=device, dtype=torch.float32)
        samples.append(c2w[:3, 3] + (rays * depth[v, u, None]) @ c2w[:3, :3].T)
    points = torch.cat(samples)
    points = points[torch.randperm(len(points), device=device)[:200000]]
    center, normal, inliers = fit_plane(points)
    if normal[2] < 0:
        normal = -normal
    print(f"local table plane: {inliers}/{len(points)} inliers", flush=True)

    # Texture grid spans the object footprint plus context.
    axis_u = torch.linalg.cross(normal, torch.tensor([1., 0, 0], device=device))
    axis_u = axis_u / axis_u.norm()
    axis_v = torch.linalg.cross(normal, axis_u)
    core = (hull | selected) & ((means - center) @ normal > -.03) & ((means - center) @ normal < .1)
    uv = torch.stack([(means[core] - center) @ axis_u, (means[core] - center) @ axis_v], -1)
    low = torch.quantile(uv, .005, dim=0) - args.margin
    high = torch.quantile(uv, .995, dim=0) + args.margin
    size = ((high - low) / args.texel).ceil().long()
    tex_w, tex_h = int(size[0]), int(size[1])
    grid_v, grid_u = torch.meshgrid(torch.arange(tex_h, device=device), torch.arange(tex_w, device=device), indexing="ij")
    world = (center + (low[0] + (grid_u.float() + .5) * args.texel)[..., None] * axis_u
             + (low[1] + (grid_v.float() + .5) * args.texel)[..., None] * axis_v).reshape(-1, 3)
    print(f"texture {tex_w}x{tex_h}", flush=True)

    K_full = torch.as_tensor(intrinsics, device=device, dtype=torch.float32)
    stacks, valids = [], []
    for index, camera in train:
        mask = cv2.dilate(load_mask(args.masks, index, args.width, height, device).astype(np.uint8), kernel, iterations=args.mask_dilate)
        mask_t = torch.as_tensor(mask.astype(bool), device=device)
        result = render(gaussians, camera["c2w"], scaled, args.width, height)
        depth = expected_depth(gaussians, result, camera["c2w"], args.width, height)
        c2w = torch.as_tensor(camera["c2w"], device=device, dtype=torch.float32).clone()
        c2w[:3, 1:3] *= -1
        local = (world - c2w[:3, 3]) @ c2w[:3, :3]
        z = local[:, 2]
        x = local[:, 0] / z.clamp_min(1e-6) * K_full[0, 0] + K_full[0, 2]
        y = local[:, 1] / z.clamp_min(1e-6) * K_full[1, 1] + K_full[1, 2]
        inside = (z > .01) & (x >= 0) & (x < full_width - 1) & (y >= 0) & (y < full_height - 1)
        sx, sy = (x * factor).long().clamp(0, args.width - 1), (y * factor).long().clamp(0, height - 1)
        view_dir = (world - c2w[:3, 3]) / (world - c2w[:3, 3]).norm(dim=1, keepdim=True)
        grazing = (view_dir @ normal).abs() > .3
        valid = inside & grazing & ~mask_t[sy, sx] & ((depth[sy, sx] - z).abs() < args.depth_tolerance)
        image = cv2.undistort(cv2.imread(str(args.data / camera["name"])), intrinsics, np.r_[distortion, 0.])[:, :, ::-1].copy()
        image = torch.as_tensor(image.astype(np.float32) / 255, device=device).permute(2, 0, 1)[None]
        grid = torch.stack([x / (full_width - 1) * 2 - 1, y / (full_height - 1) * 2 - 1], -1)[None, None]
        color = torch.nn.functional.grid_sample(image, grid, align_corners=True)[0, :, 0].T
        stacks.append(((color * 255).clamp(0, 255).byte()).cpu())
        valids.append(valid.cpu())
    colors, valid = torch.stack(stacks), torch.stack(valids)
    del stacks, valids
    counts = valid.sum(0)
    fused = torch.zeros(colors.shape[1:], dtype=torch.uint8)
    for start in range(0, colors.shape[1], 100000):
        block = colors[:, start:start + 100000].to(device).float()
        block[~valid[:, start:start + 100000].to(device)] = float("nan")
        fused[start:start + 100000] = block.nanmedian(dim=0).values.nan_to_num(0).byte().cpu()
    observed = (counts >= args.min_views).reshape(tex_h, tex_w).numpy()
    texture = fused.reshape(tex_h, tex_w, 3).numpy()
    texture[~observed] = 0
    footprint = np.zeros((tex_h, tex_w), np.uint8)
    pixels = ((torch.stack([(means[hull & ((means - center) @ normal > -.03) & ((means - center) @ normal < .03)] - center) @ axis_u,
                            (means[hull & ((means - center) @ normal > -.03) & ((means - center) @ normal < .03)] - center) @ axis_v], -1) - low) / args.texel).long().cpu().numpy()
    pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < tex_w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < tex_h)]
    footprint[pixels[:, 1], pixels[:, 0]] = 1
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    footprint = cv2.morphologyEx(footprint, cv2.MORPH_CLOSE, closing).astype(bool)
    cv2.imwrite(str(args.output / "texture_observed.png"), texture[:, :, ::-1])
    cv2.imwrite(str(args.output / "observed_mask.png"), observed.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output / "footprint.png"), footprint.astype(np.uint8) * 255)
    np.savez_compressed(args.output / "plane.npz", center=center.cpu().numpy(), normal=normal.cpu().numpy(),
                        axis_u=axis_u.cpu().numpy(), axis_v=axis_v.cpu().numpy(), low=low.cpu().numpy(), texel=args.texel,
                        hull_mask=(hull | selected).cpu().numpy())
    hole = footprint & ~observed
    (args.output / "texture_summary.json").write_text(json.dumps(
        {"size": [tex_w, tex_h], "observed_fraction": float(observed.mean()), "footprint_texels": int(footprint.sum()),
         "footprint_unobserved_texels": int(hole.sum()), "footprint_observed_fraction": float((footprint & observed).sum() / max(footprint.sum(), 1)),
         "cleared_gaussians": int((hull | selected).sum()), "texture_source": "source photos only, no inpainting"}, indent=2), encoding="utf-8")
    print((args.output / "texture_summary.json").read_text())


if __name__ == "__main__":
    main()
