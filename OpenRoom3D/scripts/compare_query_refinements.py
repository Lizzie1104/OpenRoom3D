#!/usr/bin/env python3
"""Compare aligned query refinements in an offline 3D HTML page."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from inspect_query_objects import sample_indices


def compare(args):
    object_label = getattr(args, "label", "Target object")
    full_selection = getattr(args, "full_selection", False)
    baseline_label = getattr(args, "baseline_label", "Baseline candidates")
    raw = np.load(args.raw, allow_pickle=False)
    baseline = np.load(args.baseline, allow_pickle=False)
    variants = []
    for item in args.variant:
        name, path = item.split("=", 1)
        variants.append((name, Path(path), np.load(path, allow_pickle=False)))
    xyz = raw["xyz"]
    for data in [baseline] + [entry[2] for entry in variants]:
        if not np.array_equal(data["gaussian_index"], raw["gaussian_index"]) or not np.array_equal(data["xyz"], xyz):
            raise ValueError("All inputs must reference the same Gaussians and positions")
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=("Scene RGB (sample)", baseline_label, "Refined candidates (select above)"))
    idx = sample_indices(np.arange(len(xyz)), args.max_points)
    p = xyz[idx]
    rgb = [f"rgb({r},{g},{b})" for r, g, b in raw["rgb"][idx]]
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                             marker={"size": 2, "color": rgb}, showlegend=False, hoverinfo="skip"), row=1, col=1)
    for col in (2, 3):
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                 marker={"size": 1, "color": "#888", "opacity": 0.12},
                                 showlegend=False, hoverinfo="skip"), row=1, col=col)

    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    def add_clusters(data, col, visible, prefix):
        labels = data["cluster_label"]
        selected = data["selected_mask"].astype(bool)
        rows = []
        groups, counts = np.unique(labels[selected], return_counts=True)
        order = np.argsort(counts)[::-1]
        start = len(fig.data)
        for rank, pos in enumerate(order):
            label, count = int(groups[pos]), int(counts[pos])
            indices = np.flatnonzero(selected & (labels == label))
            quota = max(32, int(args.max_points * count / max(selected.sum(), 1)))
            draw_idx = indices if full_selection else sample_indices(indices, quota)
            points = xyz[draw_idx]
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                     marker={"size": 2, "color": palette[rank % len(palette)]}, visible=visible,
                                     showlegend=(col == 3), name=f"{prefix} Cluster {label} ({count})",
                                     customdata=np.column_stack((raw["gaussian_index"][draw_idx], data["refined_similarity"][draw_idx])),
                                     hovertemplate="Gaussian %{customdata[0]}<br>score %{customdata[1]:.3f}<extra>%{fullData.name}</extra>"), row=1, col=col)
            rows.append({"cluster": label, "count": count,
                         "mean_score": float(data["refined_similarity"][indices].mean()),
                         "aabb_extent": np.ptp(xyz[indices], axis=0).tolist()})
        return list(range(start, len(fig.data))), rows

    add_clusters(baseline, 2, True, "Original result")
    fixed = len(fig.data)
    switches, summary = [], []
    for index, (name, path, data) in enumerate(variants):
        traces, rows = add_clusters(data, 3, index == 0, name)
        switches.append((name, traces))
        parameters_path = path.with_suffix(".parameters.json")
        parameters = json.loads(parameters_path.read_text()) if parameters_path.exists() else {}
        summary.append({"name": name, "source": str(path.resolve()), "parameters": parameters,
                        "selected": int(data["selected_mask"].sum()), "clusters": rows,
                        "status": "pending_visual_verification", "candidate_display_sampled": not full_selection})
    menus = []
    for name, traces in switches:
        visible = [i < fixed or i in traces for i in range(len(fig.data))]
        menus.append({"label": name, "method": "update", "args": [{"visible": visible}, {"title.text": f"{object_label} extraction comparison: {name} (pending visual verification)"}]})
    scenes = ("scene", "scene2", "scene3")
    controls = [
        {"label": "Orbit", "method": "relayout", "args": [{f"{s}.dragmode": "orbit" for s in scenes}]},
        {"label": "Pan", "method": "relayout", "args": [{f"{s}.dragmode": "pan" for s in scenes}]},
        {"label": "Hide background", "method": "restyle", "args": [{"visible": False}, [1, 2]]},
        {"label": "Show background", "method": "restyle", "args": [{"visible": True}, [1, 2]]},
        {"label": "Reset view", "method": "relayout", "args": [{f"{s}.camera": {
            "eye": {"x": 1.25, "y": 1.25, "z": 1.25}, "center": {"x": 0, "y": 0, "z": 0},
            "up": {"x": 0, "y": 0, "z": 1}} for s in scenes}]},
    ]
    lo, hi = np.quantile(xyz, [0.005, 0.995], axis=0)
    for scene in scenes:
        fig.update_layout(**{scene: {"aspectmode": "data", "uirevision": "preserve-view",
                                    **{f"{axis}axis": {"range": [float(lo[i]), float(hi[i])]} for i, axis in enumerate("xyz")}}})
    fig.update_layout(title=f"{object_label} extraction comparison: {switches[0][0]} (pending visual verification)", height=850,
                      margin={"t": 170}, uirevision="preserve-view",
                      updatemenus=[{"buttons": menus, "x": 0, "y": 1.28, "xanchor": "left", "yanchor": "top"},
                                   {"buttons": controls, "type": "buttons", "direction": "right", "x": 0, "y": 1.14, "showactive": False}])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True,
                   config={"displayModeBar": True, "responsive": True, "scrollZoom": True})
    reference_image = getattr(args, "reference_image", None)
    if reference_image is not None:
        reference_image = Path(reference_image).resolve()
        if not reference_image.is_file():
            raise FileNotFoundError(reference_image)
        image_url = html.escape(Path(os.path.relpath(reference_image, output.parent)).as_posix(), quote=True)
        display_note = 'All candidate centers are displayed. ' if full_selection else 'Display points are sampled. Exports retain all points. '
        notice = (
            '<section style="font-family:sans-serif;padding:12px 20px;background:#f3f6fa">'
            f'<strong>Extraction target: {html.escape(object_label)}</strong>'
            '<p>Gray points show the context. Hide the background to inspect candidates. '
            'Inspect candidates visually. Cluster IDs can change between runs.</p>'
            f'<p>{display_note}Points mark Gaussian centers. '
            'Gaps between centers alone do not establish a surface hole.</p>'
            '<details><summary>Compare the source photo (left) with the full Gaussian render (right)</summary>'
            f'<img src="{image_url}" style="max-width:100%" alt="Source image and full unfiltered Gaussian render">'
            '<p>The full-scene render shows reconstruction quality at this view. Evaluate the extracted candidate separately.</p>'
            '</details></section>'
        )
        document = output.read_text(encoding="utf-8")
        output.write_text(document.replace('<body>', '<body>' + notice, 1), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output}")
    for row in summary:
        print(row["name"], "selected", row["selected"], "clusters", len(row["clusters"]))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="LABEL=PATH; repeat for each candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--label", default="Target object", help="display label; does not change query scores")
    parser.add_argument("--baseline-label", default="Baseline candidates", help="title for middle reference panel")
    parser.add_argument("--full-selection", action="store_true", help="display every selected Gaussian center to diagnose sampling holes")
    parser.add_argument("--reference-image", type=Path, help="optional photo/render comparison image shown above the plots")
    args = parser.parse_args()
    if args.max_points <= 0:
        parser.error("--max-points must be positive")
    compare(args)
