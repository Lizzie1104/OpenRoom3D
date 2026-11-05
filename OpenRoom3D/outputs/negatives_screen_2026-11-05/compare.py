#!/usr/bin/env python3
"""Paired test of screen-type negatives on every saved desk02 model.

For each model (one config.yml per eval JSON listed below), queries are scored with
pairwise-min against several negative sets. Reported per set: AP against that model's
SAM 2 ownership selection (same rule as scripts/evaluate_run.py), held-out text-selection
IoU, and point counts >= 0.5 for absent queries. "television" is close to the added
negatives, so absent queries that share no word or meaning with them (smartphone,
tablet, printer, banana) are reported separately as a held-out specificity check.
Masks are SAM 2 propagations, so these are agreements, not ground-truth accuracy.

Run from the project root in the training environment with OPENROOM_CLIP_WEIGHTS set.
Writes results.json next to this file; skips models already in it.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from evaluate_run import text_selection
from export_query_gaussians import _score_in_chunks
from openroom3d.gaussian_scene import load_cameras, load_gaussians
from query_negatives import DEFAULT_NEGATIVES, without_query
from select_by_masks import accumulate, hull_support
from trusted_scene import eval_setup

HERE = Path("outputs/negatives_screen_2026-11-05")
SETS = {"default": DEFAULT_NEGATIVES,
        "+monitor": DEFAULT_NEGATIVES + ",monitor",
        "+monitor,screen": DEFAULT_NEGATIVES + ",monitor,screen"}
PRESENT = ("laptop", "notebook computer")
ABSENT = ("television", "smartphone", "tablet", "printer", "banana")
EVALS = sorted(Path("outputs/ablation_2026-11-04/eval").glob("*.json")) + sorted(Path("outputs/feat128_2026-11-04/eval").glob("*.json"))
MASKS, DATA, WIDTH = Path("outputs/objects/laptop_sam2/tracking/masks"), Path("data/processed/desk02"), 640


def run(config):
    _, pipeline, _, _ = eval_setup(config, test_mode="inference")
    model = pipeline.model
    encode = lambda text: model.text_encoding_func([q.strip() for q in text.split(",")]).to(model.means.device)
    scores = {(name, q): _score_in_chunks(model, encode(q), encode(without_query(neg, q)), .05, 16384, "pairwise-min").cpu().numpy()
              for name, neg in SETS.items() for q in PRESENT + ABSENT}
    del pipeline, model
    torch.cuda.empty_cache()
    run_dir = config.parent
    gaussians = load_gaussians(run_dir)
    cameras, K, _, (full_w, _full_h) = load_cameras(DATA, run_dir)
    factor = WIDTH / full_w
    height = int(round(_full_h * factor))
    K = K.copy()
    K[:2] *= factor
    indexed = list(enumerate(cameras))
    train = [r for r in indexed if r[1]["split"] == "train"]
    held = [r for r in indexed if r[1]["split"] == "eval"]
    fg, bg = accumulate(gaussians, train, K, WIDTH, height, MASKS, 1, 2)
    own = fg / (fg + bg).clamp_min(1e-8)
    support, seen = hull_support(gaussians["means"].detach(), train, K, WIDTH, height, MASKS, 6)
    label = ((own >= .85) & (fg >= .05) & (support >= .95) & (seen >= 5)).cpu().numpy()
    out = {}
    for name in SETS:
        row = {f"{q} AP": float(average_precision_score(label, scores[name, q])) for q in PRESENT}
        row.update({f"{q} >= 0.5": int((scores[name, q] >= .5).sum()) for q in ABSENT})
        sel = text_selection(gaussians, scores[name, "laptop"], train[::9], held, K, WIDTH, height, MASKS)
        row["text IoU"], row["text capture"] = sel["chosen"]["iou"], sel["chosen"]["captured"]
        out[name] = row
    return out


def main():
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for e in EVALS:
        name = f"{e.parent.parent.name.split('_')[0]}/{e.stem}"
        if name in results:
            continue
        config = Path(json.loads(e.read_text())["config"])
        results[name] = run(config)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, json.dumps({k: {m: round(v, 3) for m, v in r.items()} for k, r in results[name].items()}), flush=True)


if __name__ == "__main__":
    main()
