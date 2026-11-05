#!/usr/bin/env python3
"""Evaluate one desk02 Feature Splatting run with every project metric.

Writes one JSON with: Nerfstudio held-out PSNR/SSIM/LPIPS; geometry statistics;
the SAM 2 ownership/hull selection agreement; and, for the default and the
legacy query scores, point counts for present/synonym/absent queries, average
precision against that run's own ownership selection, and held-out agreement
of text >= t + largest DBSCAN with t chosen on every ninth training frame.
Masks are SAM 2 propagations, so agreement is not ground-truth accuracy.

Run from the project root in the training environment with OPENROOM_CLIP_WEIGHTS
set; the run's digests must be in docs/scene_hashes.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_query_gaussians import _score_in_chunks
from openroom3d.gaussian_scene import load_cameras, load_gaussians
from query_negatives import DEFAULT_NEGATIVES, without_query
from select_by_masks import accumulate, evaluate, hull_support, largest_cluster
from trusted_scene import eval_setup

SCHEMES = {"default": ("pairwise-min", DEFAULT_NEGATIVES), "legacy": ("softmax", "object")}
QUERIES = {"laptop": "present", "notebook computer": "synonym", "banana": "absent", "television": "absent, similar"}
THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)


def text_selection(gaussians, scores, train_fold, held, K, width, height, masks):
    xyz = gaussians["means"].detach().cpu().numpy()
    device = gaussians["means"].device
    rows = {}
    for t in THRESHOLDS:
        chosen = np.flatnonzero(scores >= t)
        if len(chosen) < 8:
            continue
        try:
            keep = chosen[largest_cluster(xyz[chosen])]
            selected = torch.zeros(len(xyz), dtype=torch.bool, device=device)
            selected[torch.as_tensor(keep, device=device)] = True
            rows[t] = dict(count=len(keep), fold_iou=evaluate(gaussians, selected, train_fold, K, width, height, masks)["iou"],
                           **evaluate(gaussians, selected, held, K, width, height, masks))
        except ValueError:  # no cluster, or a cluster with nothing visible in some view
            continue
    best = max(rows, key=lambda t: rows[t]["fold_iou"])
    return {"chosen_threshold": best, "chosen": rows[best], "by_threshold": {str(t): r for t, r in rows.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/desk02"))
    parser.add_argument("--masks", type=Path, default=Path("outputs/objects/laptop_sam2/tracking/masks"))
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    run = args.config.parent
    result = {"config": str(args.config), "mask_note": "agreement with SAM 2 masks, not ground truth"}

    _, pipeline, checkpoint, _ = eval_setup(args.config, test_mode="test")
    result["checkpoint"] = str(checkpoint)
    result["nerfstudio"] = pipeline.get_average_eval_image_metrics(output_path=None, get_std=True)
    model = pipeline.model
    encode = lambda text: model.text_encoding_func([q.strip() for q in text.split(",")]).to(model.means.device)
    scores = {}
    for scheme, (mode, negatives) in SCHEMES.items():
        for query in QUERIES:
            negative = encode(without_query(negatives, query))
            scores[scheme, query] = _score_in_chunks(model, encode(query), negative, .05, 16384, mode).cpu().numpy()
    del pipeline, model
    torch.cuda.empty_cache()

    gaussians = load_gaussians(run)
    scale = torch.exp(gaussians["scales"]).detach()
    ratio = scale.max(-1).values / scale.min(-1).values.clamp_min(1e-12)
    result["geometry"] = {"gaussians": len(ratio), "aspect_gt100": float((ratio > 100).float().mean())}
    cameras, K, _, (full_w, full_h) = load_cameras(args.data, run)
    factor = args.width / full_w
    height = int(round(full_h * factor))
    K = K.copy()
    K[:2] *= factor
    indexed = list(enumerate(cameras))
    train = [row for row in indexed if row[1]["split"] == "train"]
    held = [row for row in indexed if row[1]["split"] == "eval"]
    fold = train[::9]

    # Same rule as select_by_masks.py's chosen method: ownership >= 0.85, mass >= 0.05, hull >= 0.95 in >= 5 views.
    foreground, background = accumulate(gaussians, train, K, args.width, height, args.masks, 1, 2)
    ownership = foreground / (foreground + background).clamp_min(1e-8)
    support, seen = hull_support(gaussians["means"].detach(), train, K, args.width, height, args.masks, 6)
    core = (ownership >= .85) & (foreground >= .05) & (support >= .95) & (seen >= 5)
    result["ownership_selection"] = dict(count=int(core.sum()), **evaluate(gaussians, core, held, K, args.width, height, args.masks))
    label = core.cpu().numpy()

    result["queries"] = {}
    for scheme in SCHEMES:
        rows = result["queries"][scheme] = {}
        for query, kind in QUERIES.items():
            s = scores[scheme, query]
            rows[query] = dict(kind=kind, ge_05=int((s >= .5).sum()), fraction_ge_05=float((s >= .5).mean()),
                               ge_08=int((s >= .8).sum()), average_precision=float(average_precision_score(label, s)))
        rows["laptop"]["text_selection"] = text_selection(gaussians, scores[scheme, "laptop"], fold, held, K,
                                                          args.width, height, args.masks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    d, sel = result["queries"]["default"], result["ownership_selection"]
    print(json.dumps({"psnr": round(result["nerfstudio"]["psnr"], 3), "ownership_iou": round(sel["iou"], 3),
                      "laptop_ap": round(d["laptop"]["average_precision"], 3),
                      "text_iou": round(d["laptop"]["text_selection"]["chosen"]["iou"], 3),
                      "television_ge_05": d["television"]["ge_05"]}))


if __name__ == "__main__":
    main()
