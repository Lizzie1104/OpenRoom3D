#!/usr/bin/env python3
"""Assign original Gaussians by visible rendering contribution in view masks.

Gradients are taken only with respect to temporary dummy color channels. They
measure each Gaussian's alpha-composited contribution to foreground/background
pixels. Source means, covariance, color and opacity are NEVER optimized.
SAM-inferred masks are supervision, not ground truth. No DBSCAN pruning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.geometry import pca_pose


def contribution_signature(raw, views):
    digest = hashlib.sha256(b"visible-contribution-v1")
    for key in ("xyz", "quat", "scale", "opacity"):
        value = np.ascontiguousarray(raw[key])
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    for view in views:
        for key in ("mask", "viewmat", "intrinsics", "camera_index"):
            digest.update(np.ascontiguousarray(view[key]).tobytes())
    return digest.hexdigest()


def select_from_mass(foreground, background, seed_mask, threshold=.5, min_mass=.01):
    foreground, background = np.asarray(foreground), np.asarray(background)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    if foreground.shape != background.shape or foreground.ndim != 1 or seed_mask.shape != foreground.shape:
        raise ValueError("Mass arrays and seed mask must agree in shape [N]")
    if not np.isfinite(foreground).all() or not np.isfinite(background).all() or (foreground < 0).any() or (background < 0).any():
        raise ValueError("Contribution mass must be finite and nonnegative")
    if not 0 < threshold < 1 or not np.isfinite(min_mass) or min_mass <= 0:
        raise ValueError("Invalid ownership threshold or minimum mass")
    known = foreground + background
    score = foreground / np.maximum(known, 1e-8)
    selected = ((score >= threshold) & (foreground >= min_mass)) | (seed_mask & (known < min_mass))
    return selected, score


def contribution_mass(raw, views, device, active=None):
    n = len(raw["xyz"])
    indices = np.arange(n) if active is None else np.flatnonzero(active)
    means = torch.as_tensor(raw["xyz"][indices], device=device)
    scales = torch.as_tensor(raw["scale"][indices], device=device)
    quats = torch.as_tensor(raw["quat"][indices], device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    opacity = torch.as_tensor(raw["opacity"][indices], device=device)
    fg_mass, bg_mass, total_mass = [], [], []
    for view in views:
        mask = view["mask"].astype(bool)
        core = np.asarray(view.get("foreground_core", binary_erosion(mask, iterations=1)), dtype=bool)
        exterior = np.asarray(view.get("background_exterior", ~binary_dilation(mask, iterations=2)), dtype=bool)
        if core.shape != mask.shape or exterior.shape != mask.shape or np.any(core & exterior):
            raise ValueError("Foreground and background evidence must be disjoint masks of matching size")
        height, width = mask.shape
        colors = torch.zeros((len(indices), 3), device=device, requires_grad=True)
        result, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacity, colors=colors,
            viewmats=torch.as_tensor(view["viewmat"], device=device)[None],
            Ks=torch.as_tensor(view["intrinsics"], device=device)[None], width=width, height=height,
            packed=False, render_mode="RGB", rasterize_mode="classic")
        weights = np.stack((core, exterior, np.ones_like(core)), axis=-1).astype(np.float32)
        gradient, = torch.autograd.grad(result, colors, grad_outputs=torch.as_tensor(weights, device=device)[None])
        values = gradient.detach().cpu().numpy()
        complete = np.zeros((n, 3), dtype=np.float32)
        complete[indices] = values
        fg_mass.append(complete[:, 0])
        bg_mass.append(complete[:, 1])
        total_mass.append(complete[:, 2])
        print(f"contribution view {int(view['camera_index'])}: fg mass {values[:, 0].sum():.1f}", flush=True)
    return np.array(fg_mass), np.array(bg_mass), np.array(total_mass)


def extract(args):
    raw = dict(np.load(args.raw, allow_pickle=False))
    seed = dict(np.load(args.seed, allow_pickle=False))
    if not np.array_equal(raw["gaussian_index"], np.arange(len(raw["xyz"]))):
        raise ValueError("Full raw scene required in original checkpoint order")
    for key in ("xyz", "quat", "scale", "rgb", "opacity"):
        if not np.array_equal(raw[key][seed["gaussian_index"]], seed[key]):
            raise ValueError(f"Seed source mismatch: {key}")
    manifest = json.loads((args.masks / "manifest.json").read_text())
    if Path(manifest["seed"]).resolve() != args.seed.resolve():
        raise ValueError("Mask prompts and extraction seed must match")
    views = [dict(np.load(args.masks / entry["file"], allow_pickle=False)) for entry in manifest["views"]]
    cache = args.masks / "contribution_mass.npz"
    signature = contribution_signature(raw, views)
    if cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if not np.array_equal(data["xyz"], raw["xyz"]) or not np.array_equal(data["camera_indices"], [int(v["camera_index"]) for v in views]):
            raise ValueError("Contribution cache does not match scene/views")
        if "signature" not in data or str(data["signature"].item()) != signature:
            raise ValueError("Contribution cache source parameters/masks changed; remove stale cache explicitly before recomputing")
        fg, bg, total = data["foreground"], data["background"], data["total"]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for contribution computation")
        fg, bg, total = contribution_mass(raw, views, "cuda")
        np.savez_compressed(cache, foreground=fg, background=bg, total=total, xyz=raw["xyz"],
                            camera_indices=[int(v["camera_index"]) for v in views], signature=signature)
    fit = np.array([int(v["camera_index"]) not in args.holdout for v in views])
    if not fit.any():
        raise ValueError("No fitting views remain")
    fg_sum, bg_sum = fg[fit].sum(axis=0), bg[fit].sum(axis=0)
    seed_mask = np.zeros(len(fg_sum), dtype=bool)
    seed_mask[seed["gaussian_index"]] = True
    # Source query supplies only an unobserved fallback. Visible assignments use
    # foreground/background rendering, not center proximity or density.
    selected, score = select_from_mass(fg_sum, bg_sum, seed_mask, args.threshold, args.min_mass)
    # Very distant Gaussians may peek through an object pixel in only one view
    # while being occluded elsewhere. A generous seed-relative distance guard
    # rejects these remote scene points without forcing a dense surface.
    seed_pose = pca_pose(seed["xyz"])
    radius = args.local_radius_ratio * max(seed_pose["obb_extent"])
    distance = cKDTree(seed["xyz"]).query(raw["xyz"])[0]
    selected &= distance <= radius
    if selected.sum() < 3:
        raise ValueError("Empty contribution selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = {key: value[selected].copy() for key, value in raw.items() if value.ndim > 0 and len(value) == len(score)}
    fields.update(selected_mask=np.ones(int(selected.sum()), dtype=bool), render_ownership=score[selected].astype(np.float32),
                  foreground_mass=fg_sum[selected], background_mass=bg_sum[selected])
    # Pose from the trusted geometric subset, not dispersed rendering contributors.
    support = seed_mask[selected]
    if support.sum() < 3:
        raise ValueError("Insufficient source-seed support for pose")
    fields["pose_support_mask"] = support
    np.savez_compressed(args.output, **fields)
    contributor_pose = pca_pose(fields["xyz"], fields["xyz"][support])
    pose = pca_pose(fields["xyz"][support])
    pose.update(num_points=int(selected.sum()), pose_support_points=int(support.sum()),
                pose_geometry_scope="retained original seed; OBB bounds trusted support, not every dispersed contributor",
                contributor_obb_center=contributor_pose["obb_center"], contributor_obb_extent=contributor_pose["obb_extent"])
    pose.update(source=str(args.output.resolve()), semantic_verification="multi-view inferred masks; render review required")
    args.output.with_suffix(".pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    summaries = []
    for i, view in enumerate(views):
        fg_capture = float(fg[i, selected].sum() / max(float(fg[i].sum()), 1e-8))
        outside_share = float(bg[i, selected].sum() / max(float(total[i, selected].sum()), 1e-8))
        summaries.append({"camera_index": int(view["camera_index"]), "used_for_selection": bool(fit[i]),
                          "original_foreground_contribution_captured": fg_capture,
                          "selected_contribution_outside_inferred_mask_fraction": outside_share})
    metadata = {"object_id": args.output.stem, "label": "laptop", "method": "visible alpha-composited Gaussian contribution ownership",
                "raw_source": str(args.raw.resolve()), "seed_source": str(args.seed.resolve()), "mask_manifest": str((args.masks / "manifest.json").resolve()),
                "threshold": args.threshold, "min_foreground_mass": args.min_mass, "holdout_camera_indices": args.holdout,
                "local_radius_ratio": args.local_radius_ratio, "local_radius_reconstruction_units": radius,
                "num_gaussians": int(selected.sum()), "source_seed_count": int(seed_mask.sum()),
                "seed_retained": int((selected & seed_mask).sum()), "added_over_seed": int((selected & ~seed_mask).sum()),
                "views": summaries, "gaussian_parameters_modified": False, "checkpoint_modified": False,
                "status": "pending_original_location_and_isolation_review",
                "limitations": "Inferred masks can be wrong. Occluded unseen contributors are not reliably identified; multi-object sharing can produce tradeoffs."}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.output}: {int(selected.sum())} source Gaussians", flush=True)
    for view in summaries:
        print(view, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--min-mass", type=float, default=.01)
    p.add_argument("--holdout", type=int, nargs="*", default=[10, 40, 70])
    p.add_argument("--local-radius-ratio", type=float, default=.5, help="maximum distance to source seed as fraction of its longest OBB side")
    args = p.parse_args()
    if not 0 < args.threshold < 1 or not np.isfinite(args.min_mass) or args.min_mass <= 0 or not np.isfinite(args.local_radius_ratio) or args.local_radius_ratio <= 0:
        p.error("threshold must lie in (0,1), min-mass must be finite and positive")
    extract(args)
