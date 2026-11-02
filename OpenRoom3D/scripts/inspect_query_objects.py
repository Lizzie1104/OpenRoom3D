#!/usr/bin/env python3
"""Export cluster candidates and an offline interactive inspection report.

Clusters are spatial candidates, not confirmed semantic object instances.
IDs are scoped to this export and are not persistent across retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sample_indices(indices, maximum):
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def inspect(args):
    raw = np.load(args.raw, allow_pickle=False)
    refined = np.load(args.refined, allow_pickle=False)
    if not np.array_equal(raw["gaussian_index"], refined["gaussian_index"]):
        raise ValueError("Raw and refined Gaussian indices must match")
    if not np.array_equal(raw["xyz"], refined["xyz"]):
        raise ValueError("Raw and refined positions must match")
    xyz = raw["xyz"]
    selected = refined["selected_mask"].astype(bool)
    labels = refined["cluster_label"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in np.unique(labels[selected]):
        if label < 0:
            continue
        mask = selected & (labels == label)
        points = xyz[mask]
        object_id = f"candidate_cluster_{int(label):03d}"
        path = output / f"{object_id}.npz"
        fields = {key: raw[key][mask] for key in raw.files
                  if raw[key].ndim > 0 and len(raw[key]) == len(xyz)}
        fields["selected_mask"] = np.ones(len(points), dtype=bool)
        fields["refined_similarity"] = refined["refined_similarity"][mask]
        fields["cluster_label"] = labels[mask]
        np.savez_compressed(path, **fields)
        rows.append({
            "candidate_id": object_id, "cluster_label": int(label),
            "status": "unverified_spatial_candidate", "num_gaussians": len(points),
            "mean_raw_similarity": float(raw["similarity"][mask].mean()),
            "mean_refined_similarity": float(refined["refined_similarity"][mask].mean()),
            "centroid": points.mean(axis=0).tolist(),
            "aabb_extent": np.ptp(points, axis=0).tolist(), "npz": path.name,
        })
    rows.sort(key=lambda row: row["num_gaussians"], reverse=True)
    report = {
        "raw_source": str(args.raw.resolve()), "refined_source": str(args.refined.resolve()),
        "coordinate_system": "normalized reconstruction; units are not meters",
        "note": "A cluster may include background or only an object fragment. Inspect before manipulation.",
        "num_gaussians": len(xyz), "raw_selected": int(raw["selected_mask"].sum()),
        "refined_selected": int(selected.sum()), "num_candidates": len(rows), "candidates": rows,
    }
    (output / "candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=("Scene / RGB", "Raw query selection", "Refined cluster candidates"))
    scene_idx = sample_indices(np.arange(len(xyz)), args.max_points)
    rgb = raw["rgb"][scene_idx]
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    points = xyz[scene_idx]
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                             mode="markers", marker={"size": 2, "color": colors},
                             name="Scene", hoverinfo="skip"), row=1, col=1)
    for col in (2, 3):
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                 mode="markers", marker={"size": 1, "color": "#888", "opacity": 0.15},
                                 name=f"Scene context {col}", hoverinfo="skip", showlegend=False), row=1, col=col)
    idx = sample_indices(np.flatnonzero(raw["selected_mask"]), args.max_points)
    points = xyz[idx]
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                             marker={"size": 2, "color": raw["similarity"][idx], "colorscale": "Turbo", "cmin": 0, "cmax": 1},
                             name="Raw scores", customdata=raw["gaussian_index"][idx],
                             hovertemplate="Gaussian %{customdata}<br>score %{marker.color:.3f}<extra></extra>"), row=1, col=2)
    for candidate in rows:
        label = candidate["cluster_label"]
        # Sample only the display; candidate NPZs retain all original points.
        quota = max(64, int(args.max_points * candidate["num_gaussians"] / max(selected.sum(), 1)))
        idx = sample_indices(np.flatnonzero(selected & (labels == label)), quota)
        points = xyz[idx]
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                 marker={"size": 3}, name=f"Cluster {label} ({candidate['num_gaussians']})",
                                 customdata=raw["gaussian_index"][idx],
                                 hovertemplate="Gaussian %{customdata}<extra>%{fullData.name}</extra>"), row=1, col=3)
    lo, hi = np.quantile(xyz, [0.005, 0.995], axis=0)
    for name in ("scene", "scene2", "scene3"):
        fig.update_layout(**{name: {"aspectmode": "data",
                           "xaxis": {"range": [float(lo[0]), float(hi[0])]},
                           "yaxis": {"range": [float(lo[1]), float(hi[1])]},
                           "zaxis": {"range": [float(lo[2]), float(hi[2])]}}})
    scenes = ("scene", "scene2", "scene3")
    buttons = [
        {"label": "Orbit", "method": "relayout", "args": [{f"{scene}.dragmode": "orbit" for scene in scenes}]},
        {"label": "Pan", "method": "relayout", "args": [{f"{scene}.dragmode": "pan" for scene in scenes}]},
        {"label": "Hide background", "method": "restyle", "args": [{"visible": False}, [1, 2]]},
        {"label": "Show background", "method": "restyle", "args": [{"visible": True}, [1, 2]]},
        {"label": "Reset view", "method": "relayout", "args": [{f"{scene}.camera": {
            "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
            "center": {"x": 0, "y": 0, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1}} for scene in scenes}]},
    ]
    fig.update_layout(title=f"Query inspection: {len(rows)} unverified cluster candidates", height=800,
                      margin={"t": 140},
                      updatemenus=[{"type": "buttons", "direction": "right", "x": 0, "y": 1.18,
                                    "showactive": False, "buttons": buttons}])
    fig.write_html(str(output / "inspection.html"), include_plotlyjs=True,
                   config={"displayModeBar": True, "responsive": True, "scrollZoom": True})
    print(f"Exported {len(rows)} candidates and report: {output / 'inspection.html'}")
    for row in rows[:10]:
        print(f"cluster {row['cluster_label']:3d}: {row['num_gaussians']:6d} points; mean score {row['mean_refined_similarity']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=20000, help="display sampling cap; exports are complete")
    args = parser.parse_args()
    if args.max_points <= 0:
        parser.error("--max-points must be positive")
    inspect(args)
