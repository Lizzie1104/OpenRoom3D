#!/usr/bin/env python3
"""Refine an exported query mask with 3D KNN smoothing and clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def write_ply(path: Path, xyz: np.ndarray, scores: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property float similarity\nend_header\n")
        for p, s in zip(xyz, scores):
            f.write(f"{p[0]:.8g} {p[1]:.8g} {p[2]:.8g} {s:.8g}\n")


def refine(args: argparse.Namespace) -> tuple[Path, Path]:
    source = np.load(args.input)
    xyz = np.asarray(source["xyz"], dtype=np.float32)
    raw = np.asarray(source["similarity"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or raw.shape != (len(xyz),):
        raise ValueError("input must contain xyz [N,3] and similarity [N]")
    if args.k < 1 or args.k >= len(xyz):
        raise ValueError("k must be between 1 and N-1")
    if not 0 <= args.alpha <= 1 or not 0 <= args.threshold <= 1:
        raise ValueError("alpha and threshold must be in [0,1]")
    if args.eps <= 0 or args.min_samples < 1 or args.min_cluster < 1:
        raise ValueError("eps, min-samples and min-cluster must be positive")
    for name in ("sigma", "color_sigma", "max_distance"):
        value = getattr(args, name, None)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(xyz).all() or not np.isfinite(raw).all():
        raise ValueError("xyz and similarity must be finite")
    tree = cKDTree(xyz)
    distances, neighbours = tree.query(xyz, k=args.k + 1, workers=-1)
    distances, neighbours = distances[:, 1:], neighbours[:, 1:]
    sigma = args.sigma if args.sigma is not None else max(float(np.median(distances[:, -1])), 1e-8)
    weights = np.exp(-(distances * distances) / (2.0 * sigma * sigma))
    if getattr(args, "max_distance", None) is not None:
        weights[distances > args.max_distance] = 0
    if getattr(args, "color_sigma", None) is not None:
        if "rgb" not in source:
            raise ValueError("color weighting requires rgb in the source export")
        colors = np.asarray(source["rgb"], dtype=np.float32) / 255.0
        color_distance_sq = ((colors[:, None, :] - colors[neighbours]) ** 2).sum(axis=2)
        weights *= np.exp(-color_distance_sq / (2.0 * args.color_sigma ** 2))
    weight_sum = weights.sum(axis=1)
    neighbour_mean = raw.astype(np.float64).copy()
    np.divide((weights * raw[neighbours]).sum(axis=1), weight_sum,
              out=neighbour_mean, where=weight_sum > 1e-12)
    refined = args.alpha * raw + (1.0 - args.alpha) * neighbour_mean
    selected = refined >= args.threshold

    labels = np.full(len(xyz), -1, dtype=np.int32)
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) >= args.min_cluster:
        labels[selected_indices] = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=-1).fit_predict(xyz[selected_indices])
        counts = np.bincount(labels[selected_indices][labels[selected_indices] >= 0])
        keep_labels = np.flatnonzero(counts >= args.min_cluster)
        selected &= np.isin(labels, keep_labels)
    else:
        selected[:] = False

    output = args.output or args.input.with_name(args.input.stem + "_refined.npz")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gaussian_index=np.asarray(source["gaussian_index"]),
        xyz=xyz,
        similarity=raw,
        refined_similarity=refined.astype(np.float32),
        selected_mask=selected,
        cluster_label=labels,
    )
    ply = output.with_suffix(".ply")
    write_ply(ply, xyz[selected], refined[selected])
    import json
    parameters = {
        "source": str(args.input.resolve()), "k": args.k, "alpha": args.alpha,
        "sigma": sigma, "threshold": args.threshold, "eps": args.eps,
        "min_samples": args.min_samples, "min_cluster": args.min_cluster,
        "color_sigma": getattr(args, "color_sigma", None),
        "max_distance": getattr(args, "max_distance", None),
        "selected_count": int(selected.sum()),
        "verification": "unverified; counts alone do not measure segmentation quality",
    }
    output.with_suffix(".parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    print(f"input={args.input} points={len(xyz)} selected={int(selected.sum())}")
    print(f"saved {output}\nsaved {ply}")
    return output, ply


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="NPZ from export_query_gaussians.py")
    p.add_argument("--output", type=Path)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.5, help="raw score weight")
    p.add_argument("--sigma", type=float, help="distance bandwidth; default is median k-distance")
    p.add_argument("--color-sigma", type=float, help="optional RGB bandwidth in [0,1] color space")
    p.add_argument("--max-distance", type=float, help="ignore smoothing neighbours beyond this reconstruction distance")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.02, help="DBSCAN radius in reconstruction units")
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--min-cluster", type=int, default=32)
    return p


if __name__ == "__main__":
    refine(parser().parse_args())
