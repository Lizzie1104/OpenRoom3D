"""Compare negative-prompt schemes on the original and the jointly retrained model.

Each model is loaded once through trusted_scene. Per scheme and query it reports
point counts, average precision against that model's own SAM 2 ownership/hull
selection (a pseudo-label, not human annotation), and held-out mask agreement of
text >= t + largest DBSCAN with t chosen on every 9th training frame.
Run from the project root in the training environment with OPENROOM_CLIP_WEIGHTS set.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path[:0] = [".", "scripts"]
from export_query_gaussians import CANONICAL_NEGATIVES, _score_in_chunks
from openroom3d.gaussian_scene import load_cameras, load_gaussians
from select_by_masks import evaluate, largest_cluster
from trusted_scene import eval_setup

O = Path("outputs/retrain_semantic_2026-11-02/negatives")
DATA = Path("data/processed/desk02")
MASKS = Path("outputs/objects/laptop_sam2/tracking/masks")
MODELS = {
    "original": (Path("outputs/desk02/feature-splatting/2026-09-22_202843"),
                 Path("outputs/objects/laptop_sam2/selection/scene.npz")),
    "joint": (next(Path("outputs/retrain_semantic_2026-11-02/feature_scalereg_sh0/feature-splatting").glob("*")),
              Path("outputs/retrain_semantic_2026-11-02/selection/scene.npz")),
}
SCENE = "table,desk,chair,floor,wall,computer mouse,water bottle,cable,paper,mat"
SCHEMES = {
    "softmax: object": ("softmax", "object"),
    "softmax: canonical": ("softmax", CANONICAL_NEGATIVES),
    "pairwise-min: canonical": ("pairwise-min", CANONICAL_NEGATIVES),
    "pairwise-min: canonical + scene": ("pairwise-min", CANONICAL_NEGATIVES + "," + SCENE),
}
QUERIES = {"laptop": "present", "notebook computer": "synonym", "banana": "absent", "television": "absent, similar"}
THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)
W = 640


def text_rows(gaussians, scores, cameras, K, height):
    xyz = gaussians["means"].detach().cpu().numpy()
    indexed = list(enumerate(cameras))
    train = [row for row in indexed if row[1]["split"] == "train"][::9]
    held = [row for row in indexed if row[1]["split"] == "eval"]
    device = gaussians["means"].device
    rows = {}
    for t in THRESHOLDS:
        chosen = np.flatnonzero(scores >= t)
        if len(chosen) < 8:
            continue
        try:
            keep = chosen[largest_cluster(xyz[chosen])]
        except ValueError:
            continue
        selected = torch.zeros(len(xyz), dtype=torch.bool, device=device)
        selected[torch.as_tensor(keep, device=device)] = True
        try:
            rows[t] = dict(count=len(keep), fold_iou=evaluate(gaussians, selected, train, K, W, height, MASKS)["iou"],
                           **evaluate(gaussians, selected, held, K, W, height, MASKS))
        except ValueError:  # the renderer has no fragments when a cluster is invisible in some view
            continue
    best = max(rows, key=lambda t: rows[t]["fold_iou"])
    return {"chosen_threshold": best, "chosen": rows[best], "by_threshold": {str(t): r for t, r in rows.items()}}


def main():
    O.mkdir(parents=True, exist_ok=True)
    results = {"schemes": {k: {"mode": m, "negatives": n} for k, (m, n) in SCHEMES.items()},
               "pseudo_label": "each model's SAM 2 ownership>=0.85 hull>=0.95 selection", "models": {}}
    for name, (run, label_path) in MODELS.items():
        _, pipeline, _, _ = eval_setup(run / "config.yml", test_mode="inference")
        model = pipeline.model
        label = np.load(label_path)["selected_mask"]
        gaussians = load_gaussians(run)
        cameras, K, _, (full_w, full_h) = load_cameras(DATA, run)
        K = K.copy()
        K[:2] *= W / full_w
        height = int(round(full_h * W / full_w))
        encode = lambda text: model.text_encoding_func([q.strip() for q in text.split(",")]).to(model.means.device)
        out = results["models"][name] = {}
        for scheme, (mode, negatives) in SCHEMES.items():
            negative = encode(negatives)
            out[scheme] = {}
            for query, kind in QUERIES.items():
                scores = _score_in_chunks(model, encode(query), negative, .05, 16384, mode).cpu().numpy()
                row = dict(kind=kind, ge_05=int((scores >= .5).sum()), ge_08=int((scores >= .8).sum()),
                           fraction_ge_05=float((scores >= .5).mean()),
                           average_precision=float(average_precision_score(label, scores)))
                if query == "laptop":
                    row["mask_agreement"] = text_rows(gaussians, scores, cameras, K, height)
                    np.save(O / f"{name}_{scheme.replace(': ', '_').replace(' + ', '+').replace(' ', '_')}_laptop.npy",
                            scores.astype(np.float32))
                out[scheme][query] = row
                chosen = row.get("mask_agreement", {}).get("chosen", {})
                print(name, scheme, query, {k: v for k, v in row.items() if k != "mask_agreement"},
                      {k: round(v, 3) for k, v in chosen.items()}, row.get("mask_agreement", {}).get("chosen_threshold"),
                      flush=True)
        del pipeline, model, gaussians
        torch.cuda.empty_cache()
    (O / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
