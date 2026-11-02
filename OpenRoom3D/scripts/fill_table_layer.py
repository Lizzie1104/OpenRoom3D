#!/usr/bin/env python3
"""Fill the unobserved table region and build a separate, labelled Gaussian layer.

The filled texels are an estimate, not a reconstruction: low-frequency color comes
from nearby table texels, and detail is random-phase noise with the spectrum of a
real table patch. Every Gaussian records whether a photo observed its texel. The
checkpoint is unchanged; the layer is stored separately and can be switched off.
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
from select_by_masks import accumulate


def table_mask(texture, observed):
    """Tan table texels, excluding dark mats, cables and bright paper."""
    lab = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB).astype(np.float32)
    seed = lab[observed]
    center = np.median(seed[(seed[:, 0] > 120) & (seed[:, 2] > 135)], axis=0)
    distance = np.abs(lab - center) / np.array([38., 9., 12.])
    table = observed & (distance.max(axis=2) < 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(table.astype(np.uint8), kernel, iterations=1).astype(bool)


def pull_push(image, known):
    """Smooth multi-scale fill: normalized blur levels from coarse to fine."""
    image = image.astype(np.float32)
    weight = known.astype(np.float32)
    filled = None
    for sigma in (128, 64, 32, 16, 8, 4):
        blurred = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma)
        amount = cv2.GaussianBlur(weight, (0, 0), sigma)
        level = blurred / np.maximum(amount[..., None], 1e-5)
        confidence = np.clip(amount / .25, 0, 1)[..., None]
        filled = level if filled is None else confidence * level + (1 - confidence) * filled
    return known[..., None] * image + (1 - known[..., None]) * filled


def largest_clean_patch(table, size):
    integral = cv2.integral(table.astype(np.uint8))
    best, position = -1, (0, 0)
    for y in range(0, table.shape[0] - size, 8):
        for x in range(0, table.shape[1] - size, 8):
            total = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
            if total > best:
                best, position = total, (y, x)
    return position, best / size ** 2


def detail_noise(texture, table, shape, seed, base_sigma=3):
    """Random-phase texture from a fully clean real patch; edges would inflate it."""
    for size in (96, 64, 48, 32):
        (y, x), purity = largest_clean_patch(table, size)
        if purity >= .999:
            break
    else:
        return np.zeros(shape + (3,), np.float32), float(purity)
    patch = texture[y:y + size, x:x + size].astype(np.float32)
    detail = patch - cv2.GaussianBlur(patch, (0, 0), base_sigma)
    window = np.outer(np.hanning(size), np.hanning(size))[..., None]
    amplitude = np.abs(np.fft.fft2(detail * window, s=shape, axes=(0, 1)))
    phase = np.exp(2j * np.pi * np.random.default_rng(seed).random(shape))
    output = np.real(np.fft.ifft2(amplitude * phase[..., None], axes=(0, 1)))
    output *= detail.std(axis=(0, 1)) / np.maximum(output.std(axis=(0, 1)), 1e-6)
    return output, float(purity)


def fill(texture, table, hole, seed):
    known = table & ~hole
    base = pull_push(texture, known)
    noise, purity = detail_noise(texture, known, texture.shape[:2], seed)
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2)[..., None]
    result = texture.astype(np.float32).copy()
    result[hole] = (base + noise)[hole]
    result = soft * result + (1 - soft) * texture
    result[hole] = (base + noise * soft)[hole]
    return np.clip(result, 0, 255).astype(np.uint8), purity


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2) + 1e-12))


def cross_validate(texture, table, hole, seed=0, trials=2000, scale=.5):
    """Hide real table with the hole's shape; report how well each method restores it."""
    ys, xs = np.nonzero(hole)
    crop = hole[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    # The full hole is larger than any clean table region, so test a smaller copy.
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).astype(bool)
    height, width = crop.shape
    generator = np.random.default_rng(seed)
    rows = []
    for _ in range(trials):
        if len(rows) >= 12:
            break
        y = int(generator.integers(0, table.shape[0] - height))
        x = int(generator.integers(0, table.shape[1] - width))
        test = np.zeros_like(hole)
        test[y:y + height, x:x + width] = crop
        if table[test].mean() < .99:
            continue
        kept = table & ~test
        filled, _ = fill(texture, kept, test, seed + len(rows))
        mean = np.broadcast_to(texture[kept].mean(axis=0), texture.shape).astype(np.uint8)
        smooth = np.clip(pull_push(texture, kept), 0, 255).astype(np.uint8)
        rows.append({"filled": psnr(filled[test], texture[test]), "smooth_only": psnr(smooth[test], texture[test]),
                     "global_mean": psnr(mean[test], texture[test])})
    if not rows:
        return {"trials": 0, "note": "no clean region is large enough for the full hole shape"}
    return {"trials": len(rows), "hole_scale": scale, **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]}}


def rotation_to_wxyz(matrix):
    w = np.sqrt(max(1 + np.trace(matrix), 1e-12)) / 2
    return np.array([w, (matrix[2, 1] - matrix[1, 2]) / (4 * w), (matrix[0, 2] - matrix[2, 0]) / (4 * w),
                     (matrix[1, 0] - matrix[0, 1]) / (4 * w)], dtype=np.float32)


def build_layer(result, plane, region, hole, spacing_texels):
    """Thin splats on the local plane, with opacity feathered at the region edge."""
    texel = float(plane["texel"])
    edge = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.mgrid[0:region.shape[0]:spacing_texels, 0:region.shape[1]:spacing_texels]
    keep = region[v, u]
    v, u = v[keep], u[keep]
    coordinates = np.stack([plane["low"][0] + (u + .5) * texel, plane["low"][1] + (v + .5) * texel], 1)
    xyz = plane["center"] + coordinates[:, :1] * plane["axis_u"] + coordinates[:, 1:] * plane["axis_v"]
    rgb = result[v, u].astype(np.float32) / 255
    opacity = np.clip(edge[v, u] / 8., .05, 1.) * .98
    size = spacing_texels * texel * .9
    scale = np.tile([size, size, .0004], (len(xyz), 1)).astype(np.float32)
    rotation = np.stack([plane["axis_u"], plane["axis_v"], plane["normal"]], 1)
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] *= -1
    quat = np.tile(rotation_to_wxyz(rotation), (len(xyz), 1))
    return dict(xyz=xyz.astype(np.float32), rgb=(rgb * 255).round().astype(np.uint8), rgb_float=rgb,
                opacity=opacity.astype(np.float32), scale=scale, quat=quat, inferred=hole[v, u])


def residue_mask(gaussians, plane, region, ring, below=.45):
    """Leftovers of the object: everything under the opaque layer, plus dark rim splats.

    Hull voting misses displaced splats along the object's edge. The layer's feathered
    edge must keep the original table beneath it, or the gap shows through. Displaced
    splats also sit below the table, so the column reaches down without touching the floor.
    """
    device = gaussians["means"].device
    tensor = lambda name: torch.as_tensor(plane[name], device=device, dtype=torch.float32)
    offset = gaussians["means"].detach() - tensor("center")
    height = offset @ tensor("normal")
    column = ((offset @ tensor("axis_u") - float(plane["low"][0])) / float(plane["texel"])).long()
    row = ((offset @ tensor("axis_v") - float(plane["low"][1])) / float(plane["texel"])).long()
    inside = (column >= 0) & (column < region.shape[1]) & (row >= 0) & (row < region.shape[0])
    def lookup(mask):
        values = torch.zeros(len(height), dtype=torch.bool, device=device)
        values[inside] = torch.as_tensor(mask, device=device)[row[inside], column[inside]]
        return values
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring))
    rim = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
    dark = torch.sigmoid(gaussians["features_dc"]).mean(1) < .4
    opaque = cv2.erode(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))).astype(bool)
    return (lookup(opaque) & (height > -below) & (height < .3)) | (lookup(rim) & dark & (height > -.08) & (height < .05))


def to_parameters(layer, device):
    logit = lambda value: torch.logit(torch.as_tensor(value, device=device).clamp(1e-4, 1 - 1e-4))
    return {"means": torch.as_tensor(layer["xyz"], device=device), "scales": torch.log(torch.as_tensor(layer["scale"], device=device)),
            "quats": torch.as_tensor(layer["quat"], device=device), "opacities": logit(layer["opacity"])[:, None],
            "features_dc": logit(layer["rgb_float"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True, help="build_table_texture.py output")
    parser.add_argument("--spacing", type=int, default=2, help="texels between splats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--ownership", type=float, default=.3)
    parser.add_argument("--min-mass", type=float, default=.02)
    parser.add_argument("--below", type=float, default=.45, help="clear this far under the table; stay above the floor")
    parser.add_argument("--ring", type=int, default=25, help="rim width in texels for dark residue")
    args = parser.parse_args()
    texture = cv2.imread(str(args.workdir / "texture_observed.png"))[:, :, ::-1].copy()
    observed = cv2.imread(str(args.workdir / "observed_mask.png"), 0) > 127
    footprint = cv2.imread(str(args.workdir / "footprint.png"), 0) > 127
    plane = dict(np.load(args.workdir / "plane.npz"))
    # The hole is every unobserved component the footprint touches, so the footprint's
    # outline artefacts cannot leave islands, and open background stays unfilled.
    count, labels = cv2.connectedComponents((~observed).astype(np.uint8), connectivity=4)
    touched = np.unique(labels[footprint & ~observed])
    sizes = np.bincount(labels.ravel(), minlength=count)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    hole = np.isin(labels, [item for item in touched if sizes[item] > 200 and item not in border])
    if not hole.any():
        raise ValueError("no enclosed unobserved region touches the footprint")
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hole = cv2.dilate(cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE, closing), closing).astype(bool)
    table = table_mask(texture, observed)
    # Object colour bleeds into observed texels around the hole, so those are refilled too.
    region = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool) & (hole | observed)
    target = hole | (region & ~table)
    result, purity = fill(texture, table, target, args.seed)
    validation = cross_validate(texture, table, hole, args.seed)
    print("cross validation", validation, "noise patch purity", round(purity, 3), flush=True)
    layer = build_layer(result, plane, region, target, args.spacing)
    gaussians = load_gaussians(args.run)
    device = gaussians["means"].device
    cameras, intrinsics, _, (full_width, _) = load_cameras(args.data, args.run)
    cleared = torch.as_tensor(plane["hull_mask"], device=device) | residue_mask(gaussians, plane, region, args.ring, args.below)
    # Splats seen almost only through object pixels belong to its appearance, wherever
    # they sit: 3DGS grows such ghosts behind surfaces. Real scene parts appear elsewhere.
    scale_factor = 640 / full_width
    fit_K = intrinsics.copy()
    fit_K[:2] *= scale_factor
    train = [row for row in enumerate(cameras) if row[1]["split"] == "train"]
    foreground, background = accumulate(gaussians, train, fit_K, 640, 360, args.masks, 1, 2)
    ghosts = (foreground / (foreground + background).clamp_min(1e-8) >= args.ownership) & (foreground >= args.min_mass)
    # Only below the table: cables behind the object are real and share its low visibility.
    height = (gaussians["means"].detach() - torch.as_tensor(plane["center"], device=device)) @ torch.as_tensor(plane["normal"], device=device)
    ghosts &= height < -.03
    print("ghost splats", int((ghosts & ~cleared).sum()), flush=True)
    cleared |= ghosts
    print("cleared Gaussians", int(cleared.sum()), flush=True)
    keep = ~cleared
    fresh = to_parameters(layer, device)
    merged = {key: torch.cat([gaussians[key][keep].detach(), value]) for key, value in fresh.items()}
    width, height = 640, 360
    scaled = intrinsics.copy()
    scaled[:2] *= width / full_width
    rows = []
    for index in (29, 39, 69, 89):
        camera = cameras[index]
        original = render(gaussians, camera["c2w"], scaled, width, height)["rgb"].clamp(0, 1).cpu().numpy()
        removed = render({key: gaussians[key][keep] for key in fresh}, camera["c2w"], scaled, width, height)["rgb"].clamp(0, 1).cpu().numpy()
        composite = render(merged, camera["c2w"], scaled, width, height)["rgb"].clamp(0, 1).cpu().numpy()
        rows.append(np.hstack([original, removed, composite]))
    sheet = (np.vstack(rows)[:, :, ::-1] * 255).astype(np.uint8)
    bar = np.full((44, sheet.shape[1], 3), 255, np.uint8)
    for column, title in enumerate(("Original scene", "Laptop removed, no fill", "Inferred table layer ON")):
        cv2.putText(bar, title, (column * width + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.workdir / "composite_compare.jpg"), np.vstack([bar, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(args.workdir / "texture_filled.png"), result[:, :, ::-1])
    np.savez_compressed(args.workdir / "inferred_background.npz", scene_cleared_index=cleared.nonzero()[:, 0].cpu().numpy(),
                        **{key: value for key, value in layer.items() if key != "rgb_float"})
    (args.workdir / "layer_summary.json").write_text(json.dumps(
        {"status": "inferred background, not reconstructed; not observed by any photo where inferred=true",
         "gaussians": len(layer["xyz"]), "inferred_gaussians": int(layer["inferred"].sum()), "hole_texels": int(hole.sum()), "refilled_texels": int(target.sum()),
         "cross_validation_psnr": validation, "checkpoint_modified": False}, indent=2), encoding="utf-8")
    print("layer", len(layer["xyz"]), "inferred", int(layer["inferred"].sum()))


if __name__ == "__main__":
    main()
