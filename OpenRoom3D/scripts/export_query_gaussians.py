#!/usr/bin/env python3
"""Export per-Gaussian open-vocabulary scores from a Feature Splatting run.

By default the best positive query is compared with each negative in a
two-way temperature-scaled softmax and the weakest comparison is kept (LERF
relevancy). Default negatives are LERF's canonical phrases plus desk02 scene
categories; see ``query_negatives.py``. ``--score-mode softmax --negative object``
reproduces Feature Splatting's Viewer score. The output is an interchange
boundary for refinement and object extraction; it does not modify the checkpoint.

Run inside the ``openroom3d`` environment, for example::

    python scripts/export_query_gaussians.py \
      --config outputs/desk02/feature-splatting/2026-09-22_202843/config.yml \
      --query keyboard --threshold 0.5 --negative "object,things,stuff,texture,<scene categories>"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

if __package__:
    from .trusted_scene import eval_setup
else:
    from trusted_scene import eval_setup
if __package__:
    from .query_filename import query_filename
else:
    from query_filename import query_filename
if __package__:
    from .query_negatives import CANONICAL_NEGATIVES, DEFAULT_NEGATIVES, DEFAULT_SCORE_MODE, SCORE_MODES, without_query
else:
    from query_negatives import CANONICAL_NEGATIVES, DEFAULT_NEGATIVES, DEFAULT_SCORE_MODE, SCORE_MODES, without_query


def _split_queries(value: str) -> list[str]:
    queries = [part.strip() for part in value.split(",") if part.strip()]
    if not queries:
        raise ValueError("query must contain at least one non-empty text")
    return queries


def _write_ply(path: Path, xyz: np.ndarray, colors: np.ndarray, scores: np.ndarray) -> None:
    """Write a small ASCII PLY with score as an extra scalar property."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property float similarity\nend_header\n")
        for point, color, score in zip(xyz, colors, scores):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {score:.8g}\n"
            )


def _score_in_chunks(model, positive: torch.Tensor, negative: torch.Tensor, temperature: float, chunk_size: int,
                     mode: str = "softmax") -> torch.Tensor:
    """Compute scores without materializing all decoded features.

    ``softmax`` matches the Viewer: one softmax over positives and all negatives.
    ``pairwise-min`` follows LERF relevancy: the best positive is compared with
    each negative in a two-way softmax and the weakest comparison is kept, so
    adding negatives can only lower a score and never dilutes the denominator.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if mode not in SCORE_MODES:
        raise ValueError(f"mode must be one of {SCORE_MODES}")
    num_points = model.means.shape[0]
    scores = []
    text_embeddings = torch.cat((positive, negative), dim=0)
    num_positive = positive.shape[0]
    with torch.no_grad():
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            decoded = model.feature_mlp.per_gaussian_forward(model.distill_features[start:end])[model.main_feature_name]
            decoded = decoded / decoded.norm(dim=1, keepdim=True).clamp_min(1e-12)
            similarities = decoded @ text_embeddings.T
            if mode == "pairwise-min":
                best = similarities[:, :num_positive].max(dim=1, keepdim=True).values
                scores.append(((best - similarities[:, num_positive:]) / temperature).sigmoid().min(dim=1).values)
                continue
            # compute_similarity(..., "standard_softmax") from Feature Splatting.
            probabilities = (similarities / temperature).softmax(dim=1)
            scores.append(probabilities[:, :num_positive].sum(dim=1))
    return torch.cat(scores, dim=0)


def _normalized_rgb(model) -> torch.Tensor:
    # sh_degree=0 for the recorded run, so sigmoid(features_dc) is its RGB.
    colors = torch.sigmoid(model.features_dc.detach())
    return colors.clamp(0, 1)


def export(args: argparse.Namespace) -> tuple[Path, Path]:
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")

    _, pipeline, checkpoint_path, step = eval_setup(config_path, test_mode="inference")
    model = pipeline.model
    if not hasattr(model, "distill_features") or not hasattr(model, "feature_mlp"):
        raise TypeError("the config does not contain a Feature Splatting model")
    if model.text_encoding_func is None:
        raise RuntimeError("this model has no CLIP text encoder; use a CLIP/SAMCLIP run")

    positive_queries = _split_queries(args.query)
    negative_queries = _split_queries(without_query(args.negative, args.query))
    device = model.means.device
    positive = model.text_encoding_func(positive_queries).to(device)
    negative = model.text_encoding_func(negative_queries).to(device)
    scores = _score_in_chunks(model, positive, negative, args.temperature, args.chunk_size, args.score_mode)

    xyz = model.means.detach().cpu().numpy().astype(np.float32)
    similarity = scores.detach().cpu().numpy().astype(np.float32)
    indices = np.arange(len(xyz), dtype=np.int64)
    selected_mask = similarity >= args.threshold if args.threshold is not None else np.ones(len(xyz), dtype=bool)
    colors = (_normalized_rgb(model).detach().cpu().numpy() * 255).round().astype(np.uint8)
    opacity = torch.sigmoid(model.opacities.detach()).squeeze(-1).cpu().numpy().astype(np.float32)
    scales = torch.exp(model.scales.detach()).cpu().numpy().astype(np.float32)
    quats = model.quats.detach().cpu().numpy().astype(np.float32)

    output = args.output
    if output is None:
        safe_query = query_filename("_".join(positive_queries))
        output = Path("outputs/query_exports") / safe_query
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / f"{query_filename(positive_queries[0])}.npz"
    np.savez_compressed(
        npz_path,
        gaussian_index=indices,
        xyz=xyz,
        similarity=similarity,
        selected_mask=selected_mask,
        rgb=colors,
        opacity=opacity,
        scale=scales,
        quat=quats,
    )

    selected = selected_mask
    ply_path = output / f"{query_filename(positive_queries[0])}_selected.ply"
    _write_ply(ply_path, xyz[selected], colors[selected], similarity[selected])
    metadata = output / f"{query_filename(positive_queries[0])}.txt"
    metadata.write_text(
        f"query={','.join(positive_queries)}\nnegative={','.join(negative_queries)}\n"
        f"threshold={args.threshold}\ntemperature={args.temperature}\nscore_mode={args.score_mode}\n"
        f"num_gaussians={len(xyz)}\nselected={int(selected.sum())}\n"
        f"checkpoint={checkpoint_path}\nstep={step}\n",
        encoding="utf-8",
    )
    return npz_path, ply_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Feature Splatting config.yml")
    parser.add_argument("--query", required=True, help="positive query, comma-separated for synonyms")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVES,
                        help="negatives, comma-separated; the default adds desk02 categories to "
                             f"LERF's {CANONICAL_NEGATIVES}; negatives equal to the query are dropped")
    parser.add_argument("--score-mode", choices=SCORE_MODES, default=DEFAULT_SCORE_MODE,
                        help="pairwise-min is LERF relevancy; softmax with 'object' matches the Viewer")
    parser.add_argument("--threshold", type=float, default=0.5, help="selection threshold; use -1 to export all")
    parser.add_argument("--temperature", type=float, default=0.05, help="Viewer softmax temperature")
    parser.add_argument("--chunk-size", type=int, default=16384, help="decoded Gaussian batch size")
    parser.add_argument("--output", type=Path, help="output directory")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    npz, ply = export(arguments)
    print(f"saved {npz}")
    print(f"saved {ply}")
