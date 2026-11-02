"""Held-out frames: photo, then object and removed scene for baseline vs retrain."""
import sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, ".")
from openroom3d.gaussian_scene import load_cameras, load_gaussians, render
O = Path("outputs/retrain_2026-11-01")
DATA = Path("data/processed/desk02")
RUNS = [("baseline", Path("outputs/desk02/feature-splatting/2026-09-22_202843"), Path("outputs/objects/laptop_sam2/selection")),
        ("scalereg", next(O.glob("scalereg_sh0/splatfacto/*")), O / "selection_scalereg")]
W, H = 640, 360
cameras, K, dist, _ = load_cameras(DATA, RUNS[0][1])
Ks = K.copy(); Ks[:2] *= W / 1280
held = [c for c in cameras if c["split"] == "eval"]
rows = []
for camera in held[1:9:3]:
    image = cv2.imread(str(DATA / camera["name"]))
    row = [cv2.resize(cv2.undistort(image, K, np.r_[dist, 0.]), (W, H), interpolation=cv2.INTER_AREA)]
    for name, run, selection in RUNS:
        g = load_gaussians(run)
        mask = torch.as_tensor(np.load(selection / "scene.npz")["selected_mask"], device=g["means"].device)
        for active, label in ((mask, "object"), (~mask, "removed")):
            rgb = render(g, camera["c2w"], Ks, W, H, active=active, background=1.)["rgb"].clamp(0, 1).cpu().numpy()
            view = (rgb[:, :, ::-1] * 255).astype(np.uint8).copy()
            cv2.putText(view, f"{name} {label}", (8, 22), 0, .6, (0, 0, 220), 2)
            row.append(view)
    rows.append(np.hstack(row))
cv2.imwrite(str(O / "compare.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
