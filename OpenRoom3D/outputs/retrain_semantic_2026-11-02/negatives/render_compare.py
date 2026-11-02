"""Render before/after panels for two held-out views.

Rows: original or joint checkpoint, scored with the Viewer default (`object`) or the
recommended pairwise-min canonical + scene negatives. Columns: photo, laptop
heatmap, look-alike absent `television` heatmap, and the isolated text >= t +
largest DBSCAN cluster at the train-fold threshold from results.json.
Run from the project root in the training environment with OPENROOM_CLIP_WEIGHTS set.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path[:0] = [".", "scripts"]
from compare_negatives import DATA, MODELS, O, SCHEMES
from export_query_gaussians import _score_in_chunks
from openroom3d.gaussian_scene import load_cameras, load_gaussians, render
from select_by_masks import largest_cluster
from trusted_scene import eval_setup

W, H = 480, 270
ROWS = [("original", "softmax: object", "Original model, negative 'object' (before)"),
        ("original", "pairwise-min: canonical + scene", "Original model, scene negatives (after)"),
        ("joint", "softmax: object", "Joint retrain, negative 'object'"),
        ("joint", "pairwise-min: canonical + scene", "Joint retrain, scene negatives")]
VIEWS = 2


def colored(gaussians, scores):
    heat = cv2.applyColorMap((np.clip(scores, 0, 1) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
    rgb = torch.as_tensor(heat / 255., device=gaussians["means"].device, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return {**gaussians, "features_dc": torch.logit(rgb)}


def image(result):
    return (result["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def label(panel, text):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (W, 22), (255, 255, 255), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    results = json.loads((O / "results.json").read_text())["models"]
    sheets = [[] for _ in range(VIEWS)]
    for name in ("original", "joint"):
        run = MODELS[name][0]
        _, pipeline, _, _ = eval_setup(run / "config.yml", test_mode="inference")
        model = pipeline.model
        encode = lambda text: model.text_encoding_func([q.strip() for q in text.split(",")]).to(model.means.device)
        gaussians = load_gaussians(run)
        xyz = gaussians["means"].detach().cpu().numpy()
        cameras, K, dist, (fw, fh) = load_cameras(DATA, run)
        Ks = K.copy()
        Ks[:2] *= W / fw
        held = [c for c in cameras if c["split"] == "eval"]
        views = [held[2], held[6]]
        for model_name, scheme, title in ROWS:
            if model_name != name:
                continue
            mode, negatives = SCHEMES[scheme]
            negative = encode(negatives)
            laptop = _score_in_chunks(model, encode("laptop"), negative, .05, 16384, mode).cpu().numpy()
            tv = _score_in_chunks(model, encode("television"), negative, .05, 16384, mode).cpu().numpy()
            t = results[name][scheme]["laptop"]["mask_agreement"]["chosen_threshold"]
            row = results[name][scheme]["laptop"]["mask_agreement"]["chosen"]
            chosen = np.flatnonzero(laptop >= t)
            keep = np.zeros(len(xyz), dtype=bool)
            keep[chosen[largest_cluster(xyz[chosen])]] = True
            active = torch.as_tensor(keep, device=gaussians["means"].device)
            for v, camera in enumerate(views):
                photo = cv2.undistort(cv2.imread(str(DATA / camera["name"]))[:, :, ::-1], K, np.r_[dist, 0.])
                photo = cv2.resize(photo, (W, H), interpolation=cv2.INTER_AREA)
                panels = [label(photo, "held-out photo"),
                          label(image(render(colored(gaussians, laptop), camera["c2w"], Ks, W, H, background=0.)),
                                f"'laptop' score  (>=0.5: {(laptop >= .5).mean():.0%})"),
                          label(image(render(colored(gaussians, tv), camera["c2w"], Ks, W, H, background=0.)),
                                f"'television' (absent) >=0.5: {int((tv >= .5).sum())} pts"),
                          label(image(render(gaussians, camera["c2w"], Ks, W, H, active=active, background=1.)),
                                f"text>={t} cluster  IoU {row['iou']:.3f}  capture {row['captured']:.2f}")]
                strip = np.concatenate(panels, axis=1)
                header = np.full((26, strip.shape[1], 3), 235, np.uint8)
                cv2.putText(header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
                sheets[v].append(np.concatenate([header, strip], axis=0))
        del pipeline, model, gaussians
        torch.cuda.empty_cache()
    for v, rows in enumerate(sheets):
        path = O / f"before_after_view{v + 1}.jpg"
        cv2.imwrite(str(path), np.concatenate(rows, axis=0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("saved", path)


main()
