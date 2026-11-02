#!/usr/bin/env python3
"""Assign original Gaussians to one object from multi-view SAM 2 masks.

Each Gaussian accumulates its visible render weight inside the mask core and
outside the dilated mask, on training frames only. Render weight alone also
picks distant Gaussians seen through the object from a few views. Centers passing
a visual-hull vote form a core; owned Gaussians near that core are kept too, because
displaced contributors to an object's appearance fail the vote. Eval frames stay held out. Masks are inferred, so these scores are
agreement with SAM 2, not ground-truth accuracy. Gaussian parameters are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.gaussian_scene import load_cameras, load_gaussians, render
from openroom3d.geometry import pca_pose


def load_mask(directory, index, width, height, device):
    mask = cv2.imread(str(Path(directory) / f"{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA) > 127
    return mask


def accumulate(gaussians, cameras, intrinsics, width, height, masks, erode, dilate):
    device = gaussians["means"].device
    count = len(gaussians["means"])
    foreground, background = torch.zeros(count, device=device), torch.zeros(count, device=device)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for index, camera in cameras:
        mask = load_mask(masks, index, width, height, device).astype(np.uint8)
        core = cv2.erode(mask, kernel, iterations=erode).astype(bool)
        exterior = ~cv2.dilate(mask, kernel, iterations=dilate).astype(bool)
        result = render(gaussians, camera["c2w"], intrinsics, width, height)
        pixel, owner, weight = result["pixel"], result["gaussian"], result["weight"]
        foreground.index_add_(0, owner, weight * torch.as_tensor(core.ravel(), device=device)[pixel])
        background.index_add_(0, owner, weight * torch.as_tensor(exterior.ravel(), device=device)[pixel])
    return foreground, background


def hull_support(means, cameras, intrinsics, width, height, masks, dilate):
    """Fraction of views containing a Gaussian center where it lands inside the mask."""
    device = means.device
    inside, seen = torch.zeros(len(means), device=device), torch.zeros(len(means), device=device)
    K = torch.as_tensor(intrinsics, device=device, dtype=torch.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for index, camera in cameras:
        mask = cv2.dilate(load_mask(masks, index, width, height, device).astype(np.uint8), kernel, iterations=dilate)
        mask = torch.as_tensor(mask.astype(bool), device=device)
        c2w = torch.as_tensor(camera["c2w"], device=device, dtype=torch.float32).clone()
        c2w[:3, 1:3] *= -1
        points = (means - c2w[:3, 3]) @ c2w[:3, :3]
        z = points[:, 2].clamp_min(1e-6)
        u = (points[:, 0] / z * K[0, 0] + K[0, 2]).floor().long()
        v = (points[:, 1] / z * K[1, 1] + K[1, 2]).floor().long()
        visible = (points[:, 2] > .01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        seen += visible.float()
        inside[visible] += mask[v[visible], u[visible]].float()
    return inside / seen.clamp_min(1), seen


def evaluate(gaussians, selected, cameras, intrinsics, width, height, masks):
    """Held-out agreement with masks, for the isolated object and the remaining scene."""
    device = gaussians["means"].device
    rows = []
    for index, camera in cameras:
        mask = torch.as_tensor(load_mask(masks, index, width, height, device), device=device)
        if not mask.any():
            continue
        full = render(gaussians, camera["c2w"], intrinsics, width, height)
        total = full["weight"][mask.ravel()[full["pixel"]]].sum()
        kept = (selected[full["gaussian"]] & mask.ravel()[full["pixel"]])
        captured = (full["weight"][kept].sum() / total).item()
        isolated = render(gaussians, camera["c2w"], intrinsics, width, height, active=selected)
        shape = isolated["alpha"] > .5
        union = (shape | mask).sum().clamp_min(1)
        outside = isolated["weight"][~mask.ravel()[isolated["pixel"]]].sum() / isolated["weight"].sum().clamp_min(1e-6)
        rows.append(dict(iou=((shape & mask).sum() / union).item(), captured=captured,
                         coverage=isolated["alpha"][mask].mean().item(),
                         outside=outside.item()))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def largest_cluster(xyz, eps=.01, min_samples=8):
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(xyz)
    if (labels >= 0).sum() == 0:
        raise ValueError("no cluster found")
    return labels == np.bincount(labels[labels >= 0]).argmax()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Feature Splatting run directory")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True, help="sam2_track_object.py masks directory")
    parser.add_argument("--scores", type=Path, required=True, help="NPZ with text score array")
    parser.add_argument("--score-key", default="laptop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="laptop")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.5, .85])
    parser.add_argument("--hull", type=float, nargs="+", default=[.95])
    parser.add_argument("--radius", type=float, nargs="+", default=[.02, .04, .08])
    parser.add_argument("--hull-dilate", type=int, default=6)
    parser.add_argument("--min-views", type=int, default=5)
    parser.add_argument("--min-mass", type=float, default=.05)
    parser.add_argument("--erode", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gaussians = load_gaussians(args.run)
    cameras, intrinsics, _, (full_width, full_height) = load_cameras(args.data, args.run)
    factor = args.width / full_width
    height = int(round(full_height * factor))
    scaled = intrinsics.copy()
    scaled[:2] *= factor
    indexed = list(enumerate(cameras))
    train = [row for row in indexed if row[1]["split"] == "train"]
    held = [row for row in indexed if row[1]["split"] == "eval"]
    foreground, background = accumulate(gaussians, train, scaled, args.width, height, args.masks, args.erode, args.dilate)
    ownership = foreground / (foreground + background).clamp_min(1e-8)
    device = gaussians["means"].device
    support, seen = hull_support(gaussians["means"].detach(), train, scaled, args.width, height, args.masks, args.hull_dilate)
    text = torch.as_tensor(np.load(args.scores)[args.score_key], device=device)
    xyz = gaussians["means"].detach().cpu().numpy()
    candidates = {}
    for threshold in (.5, .8):
        base = torch.zeros(len(xyz), dtype=torch.bool, device=device)
        chosen = (text >= threshold).nonzero()[:, 0]
        candidates[f"text>={threshold}"] = (text >= threshold)
        base[chosen[torch.as_tensor(largest_cluster(xyz[chosen.cpu().numpy()]), device=device)]] = True
        candidates[f"text>={threshold} + largest DBSCAN"] = base
    for threshold in args.thresholds:
        owned = (ownership >= threshold) & (foreground >= args.min_mass)
        candidates[f"sam2 ownership>={threshold}"] = owned
        for hull in args.hull:
            core = owned & (support >= hull) & (seen >= args.min_views)
            candidates[f"sam2 ownership>={threshold} hull>={hull}"] = core
            distance = cKDTree(xyz[core.cpu().numpy()]).query(xyz, workers=-1)[0]
            for radius in args.radius:
                near = torch.as_tensor(distance <= radius, device=device)
                candidates[f"sam2 ownership>={threshold} hull>={hull} radius<={radius}"] = core | (owned & near)
    report, validation = {}, {}
    fold = train[::9]  # choose the method on training frames; eval frames only report it
    for name, selected in candidates.items():
        report[name] = dict(count=int(selected.sum()),
                            **evaluate(gaussians, selected, held, scaled, args.width, height, args.masks))
        validation[name] = evaluate(gaussians, selected, fold, scaled, args.width, height, args.masks)["iou"]
        print(name, {key: round(value, 4) for key, value in report[name].items()},
              "train-fold iou", round(validation[name], 4), flush=True)
    sam2_names = [name for name in report if name.startswith("sam2")]
    best = max(sam2_names, key=lambda name: validation[name])
    selected = candidates[best].cpu().numpy()
    opacity = torch.sigmoid(gaussians["opacities"][:, 0]).detach().cpu().numpy().astype(np.float32)
    scale = torch.exp(gaussians["scales"]).detach().cpu().numpy().astype(np.float32)
    rgb = (torch.sigmoid(gaussians["features_dc"]).clamp(0, 1).detach().cpu().numpy() * 255).round().astype(np.uint8)
    quat = gaussians["quats"].detach().cpu().numpy().astype(np.float32)
    similarity = text.cpu().numpy().astype(np.float32)
    scene = dict(gaussian_index=np.arange(len(xyz)), xyz=xyz, similarity=similarity, rgb=rgb,
                 opacity=opacity, scale=scale, quat=quat)
    np.savez_compressed(args.output / "scene.npz", selected_mask=selected, **scene)
    fields = {key: value[selected] for key, value in scene.items()}
    fields["selected_mask"] = np.ones(int(selected.sum()), dtype=bool)
    fields["render_ownership"] = ownership.cpu().numpy()[selected].astype(np.float32)
    path = args.output / f"obj_{args.label}.npz"
    np.savez_compressed(path, **fields)
    pose = pca_pose(fields["xyz"])
    pose.update(num_points=int(selected.sum()), source=str(path.resolve()),
                semantic_verification="SAM 2 inferred masks; render review required")
    path.with_suffix(".pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(
        {"selected_method": best, "held_out_frames": [row[0] for row in held], "methods": report,
         "selection_rule": "highest IoU on every 9th training frame", "train_fold_iou": validation,
         "metric_note": "agreement with SAM 2 masks on held-out frames, not ground truth",
         "gaussian_parameters_modified": False, "checkpoint_modified": False}, indent=2), encoding="utf-8")
    print("selected", best, int(selected.sum()), path)


if __name__ == "__main__":
    main()
