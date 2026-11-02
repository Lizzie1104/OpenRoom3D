"""Compare retrained geometry with the baseline using one renderer and one split.

PSNR comes from the project's torch renderer on undistorted frames, so it matches
the diagnosis number and is not an official Nerfstudio metric. Run from the root.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path[:0] = [".", "scripts"]
from openroom3d.gaussian_scene import load_cameras, load_gaussians, render
from select_by_masks import load_mask

O = Path("outputs/retrain_2026-11-01")
DATA = Path("data/processed/desk02")
MASKS = Path("outputs/objects/laptop_sam2/tracking/masks")
BASE = Path("outputs/desk02/feature-splatting/2026-09-22_202843")
RUNS = {"baseline": BASE, "control": next(O.glob("control_sh0/splatfacto/*")), "scalereg": next(O.glob("scalereg_sh0/splatfacto/*"))}
W, H = 640, 360


def undistorted(name, K, dist):
    image = cv2.imread(str(DATA / name))[:, :, ::-1]
    return cv2.resize(cv2.undistort(image, K, np.r_[dist, 0.]), (W, H), interpolation=cv2.INTER_AREA) / 255.


def main():
    scores = dict(np.load("outputs/diagnosis_2026-10-31/scores.npz"))
    base_xyz = load_gaussians(BASE)["means"].detach().cpu().numpy()
    tree = cKDTree(base_xyz)
    out = {}
    for name, run in RUNS.items():
        g = load_gaussians(run)
        cameras, K, dist, _ = load_cameras(DATA, run)
        Ks = K.copy()
        Ks[:2] *= W / 1280
        scale = torch.exp(g["scales"]).detach()
        ratio = scale.max(-1).values / scale.min(-1).values.clamp_min(1e-12)
        psnr, heavy, total, sharp = [], 0., 0., 0.
        for index, camera in enumerate(cameras):
            if camera["split"] != "eval":
                continue
            r = render(g, camera["c2w"], Ks, W, H, background=1.)
            gt = undistorted(camera["name"], K, dist)
            psnr.append(float(-10 * np.log10(((r["rgb"].clamp(0, 1).cpu().numpy() - gt) ** 2).mean())))
            mask = torch.as_tensor(np.asarray(load_mask(MASKS, index, W, H, "cpu")).reshape(-1).astype(np.float32), device=r["pixel"].device)
            inside = mask[r["pixel"]] > 0
            weight = r["weight"][inside]
            total += float(weight.sum())
            heavy += float(weight[ratio[r["gaussian"][inside]] > 100].sum())
            sharp += float(weight[ratio[r["gaussian"][inside]] <= 10].sum())
        out[name] = dict(count=len(ratio), psnr_eval=float(np.mean(psnr)), psnr_frames=psnr,
                         ratio_gt100=float((ratio > 100).float().mean()),
                         ratio_median=float(ratio.median()),
                         laptop_weight_gt100=heavy / total, laptop_weight_le10=sharp / total,
                         transform=json.loads(Path(run, "dataparser_transforms.json").read_text()))
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()
                     if k not in ("psnr_frames", "transform")}, flush=True)
        if name != "baseline":
            xyz = g["means"].detach().cpu().numpy()
            nearest = tree.query(xyz, workers=-1)[1]
            np.savez_compressed(O / f"{name}_scores_transferred.npz", **{k: v[nearest] for k, v in scores.items()})
    (O / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


main()
