#!/usr/bin/env python3
"""Export one explicitly selected cluster as a scene-scoped object candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def export(args):
    raw = np.load(args.raw, allow_pickle=False)
    refined = np.load(args.refined, allow_pickle=False)
    for key in ("gaussian_index", "xyz"):
        if not np.array_equal(raw[key], refined[key]):
            raise ValueError(f"Source exports do not agree on {key}")
    mask = refined["selected_mask"] & (refined["cluster_label"] == args.cluster)
    if mask.sum() < 3:
        raise ValueError("Requested retained cluster has fewer than three points")
    fields = {key: raw[key][mask] for key in raw.files
              if raw[key].ndim > 0 and len(raw[key]) == len(mask)}
    fields.update(selected_mask=np.ones(int(mask.sum()), dtype=bool),
                  cluster_label=refined["cluster_label"][mask],
                  refined_similarity=refined["refined_similarity"][mask])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    # Scope IDs to these source files and this segmentation, not future training runs.
    digest = hashlib.sha256(fields["gaussian_index"].tobytes() + fields["xyz"].tobytes()).hexdigest()
    metadata = {"object_id": args.output.stem, "label": args.label,
                "cluster_label": args.cluster, "num_gaussians": int(mask.sum()),
                "raw_source": str(args.raw.resolve()), "refined_source": str(args.refined.resolve()),
                "selection_sha256": digest, "id_scope": "this scene export only",
                "semantic_score_is_probability_calibration": False,
                "mean_refined_similarity": float(fields["refined_similarity"].mean()),
                "status": "candidate_pending_isolated_render_review"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.output}: cluster {args.cluster}, {int(mask.sum())} Gaussians")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--cluster", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    export(p.parse_args())
