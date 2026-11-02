#!/usr/bin/env python3
"""Render a selected Gaussian set and generate a photo/pose review page.

Selection happens only in memory and is restored after rendering. Checkpoint
files and upstream source code are never changed. Run in activated openroom3d.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image, ImageDraw, ImageOps
if __package__:
    from .trusted_scene import eval_setup
else:
    from trusted_scene import eval_setup


def validate_selection(model, data):
    indices = np.asarray(data["gaussian_index"])
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("gaussian_index must be a 1D integer array")
    if len(indices) < 3 or len(np.unique(indices)) != len(indices):
        raise ValueError("Selection must contain at least three distinct Gaussians")
    if indices.min() < 0 or indices.max() >= len(model.means):
        raise ValueError("Selection indices are outside this model")
    rows = torch.as_tensor(indices, device=model.means.device, dtype=torch.long)
    source_xyz = data["source_xyz"] if "source_xyz" in data else data["xyz"]
    if not np.allclose(model.means.detach()[rows].cpu().numpy(), source_xyz, atol=1e-7, rtol=0):
        raise ValueError("Selection positions do not agree with this checkpoint; wrong source or modified object")
    for key, tensor in (("scale", torch.exp(model.scales)), ("opacity", torch.sigmoid(model.opacities).squeeze(-1)), ("quat", model.quats), ("rgb", torch.sigmoid(model.features_dc) * 255)):
        source_key = "source_" + key if "source_" + key in data else key
        tolerance = .51 if key == "rgb" else 1e-6
        if source_key in data and not np.allclose(tensor.detach()[rows].cpu().numpy(), data[source_key], atol=tolerance, rtol=1e-6):
            raise ValueError(f"Selection {key} does not agree with checkpoint")
    return rows


def pose_figure(data, pose):
    points = data["xyz"]
    colors = [f"rgb({r},{g},{b})" for r, g, b in data["rgb"]]
    fig = go.Figure(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                              marker={"size": 2, "color": colors}, name="Object candidate (all centers)"))
    rotation = np.array(pose["rotation_matrix_columns"])
    center = np.array(pose["centroid"])
    length = float(max(pose["obb_extent"])) * .35
    for i, (label, color) in enumerate(zip("XYZ", ("red", "green", "blue"))):
        end = center + rotation[:, i] * length
        fig.add_trace(go.Scatter3d(x=[center[0], end[0]], y=[center[1], end[1]], z=[center[2], end[2]],
                                  mode="lines+text", text=["", label], line={"color": color, "width": 6}, name=f"PCA {label}"))
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    corners = np.array(pose["obb_center"]) + (signs * np.array(pose["obb_extent"]) / 2) @ rotation.T
    coordinates = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(signs[i] != signs[j]) == 1:
                coordinates.extend((corners[i].tolist(), corners[j].tolist(), [None, None, None]))
    x, y, z = zip(*coordinates)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": "orange", "width": 3}, name="OBB"))
    fig.update_layout(height=650, scene={"aspectmode": "data"}, title="Candidate local axes and oriented box (PCA pose)",
                      updatemenus=[{"type": "buttons", "direction": "right", "buttons": [
                          {"label": "Orbit", "method": "relayout", "args": [{"scene.dragmode": "orbit"}]},
                          {"label": "Pan", "method": "relayout", "args": [{"scene.dragmode": "pan"}]},
                      ]}])
    return fig


def render(args):
    data = np.load(args.object, allow_pickle=False)
    pose = json.loads(args.pose.read_text())
    if Path(pose["source"]).resolve() != args.object.resolve():
        raise ValueError("Pose source must match object selection")
    _, pipeline, checkpoint, step = eval_setup(args.config.resolve(), test_mode="inference")
    model = pipeline.model
    dataset = pipeline.datamanager.train_dataset
    rows = validate_selection(model, data)
    remaining_mask = torch.ones(len(model.means), dtype=torch.bool, device=model.means.device)
    remaining_mask[rows] = False
    args.output.mkdir(parents=True, exist_ok=True)
    originals = dict(model.gauss_params.items())
    background = model.background_color.clone()
    review_rows = []
    try:
        model.background_color = torch.ones_like(background)
        for index in args.camera_index:
            if not 0 <= index < len(dataset):
                raise ValueError(f"camera index {index} is outside dataset")
            camera = dataset.cameras[index:index + 1].to(model.means.device)
            camera.rescale_output_resolution(min(1.0, args.max_width / int(camera.width.item())))
            with torch.no_grad():
                full = model.get_outputs(camera)
                full_rgb = full["rgb"].clamp(0, 1).cpu().numpy()
                try:
                    for name, parameter in originals.items():
                        model.gauss_params[name] = torch.nn.Parameter(parameter.detach()[rows].clone(), requires_grad=False)
                    if "source_xyz" in data:
                        for name, key in (("means", "xyz"), ("quats", "quat"), ("scales", "scale")):
                            value = torch.as_tensor(data[key], device=model.means.device)
                            if not torch.isfinite(value).all() or (name == "scales" and (value <= 0).any()):
                                raise ValueError(f"Invalid derived {key}")
                            if name == "scales":
                                value = value.log()
                            model.gauss_params[name] = torch.nn.Parameter(value.clone(), requires_grad=False)
                        for name, key, divisor in (("features_dc", "rgb", 255.0), ("opacities", "opacity", 1.0)):
                            value = torch.as_tensor(data[key], device=model.means.device, dtype=torch.float32) / divisor
                            value = torch.logit(value.clamp(1e-6, 1 - 1e-6))
                            if name == "opacities":
                                value = value[:, None]
                            if "surface_repair_mask" in data:
                                changed = torch.as_tensor(data["surface_repair_mask"], device=model.means.device, dtype=torch.bool)
                            elif "source_" + key in data:
                                difference = np.asarray(data[key]) != np.asarray(data["source_" + key])
                                if difference.ndim == 2:
                                    difference = difference.any(axis=1)
                                changed = torch.as_tensor(difference, device=model.means.device, dtype=torch.bool)
                            else:
                                changed = torch.ones(len(rows), device=model.means.device, dtype=torch.bool)
                            original_value = originals[name].detach()[rows].clone()
                            original_value[changed] = value[changed]
                            value = original_value
                            model.gauss_params[name] = torch.nn.Parameter(value, requires_grad=False)
                    isolated = model.get_outputs(camera)
                    isolated_rgb = isolated["rgb"].clamp(0, 1).cpu().numpy()
                    alpha = isolated["accumulation"].squeeze(-1).cpu().numpy()
                finally:
                    for name, parameter in originals.items():
                        model.gauss_params[name] = parameter
                try:
                    for name, parameter in originals.items():
                        model.gauss_params[name] = torch.nn.Parameter(parameter.detach()[remaining_mask].clone(), requires_grad=False)
                    remainder = model.get_outputs(camera)
                    remaining_rgb = remainder["rgb"].clamp(0, 1).cpu().numpy()
                finally:
                    for name, parameter in originals.items():
                        model.gauss_params[name] = parameter
            full_image = Image.fromarray((full_rgb * 255).round().astype(np.uint8))
            np.savez_compressed(args.output / f"camera_{index:03d}_render_arrays.npz",
                                full_rgb=full_rgb, isolated_rgb=isolated_rgb, isolated_alpha=alpha,
                                remaining_rgb=remaining_rgb)
            isolated_image = Image.fromarray((isolated_rgb * 255).round().astype(np.uint8))
            ys, xs = np.where(alpha > .05)
            if len(xs) == 0:
                raise ValueError(f"Object not visible in camera {index}; choose another camera")
            box = (max(0, int(xs.min()) - 15), max(0, int(ys.min()) - 15),
                   min(isolated_image.width, int(xs.max()) + 16), min(isolated_image.height, int(ys.max()) + 16))
            crop = isolated_image.crop(box)
            crop_name = f"camera_{index:03d}_isolated.png"
            crop.save(args.output / crop_name)
            remainder_image = Image.fromarray((remaining_rgb * 255).round().astype(np.uint8))
            remainder_name = f"camera_{index:03d}_remaining.jpg"
            remainder_image.save(args.output / remainder_name)
            source_path = Path(dataset.image_filenames[index])
            source = Image.open(source_path).convert("RGB")
            source = ImageOps.fit(source, full_image.size)
            w, h = full_image.size
            canvas = Image.new("RGB", (w * 3, h + 35), "white")
            draw = ImageDraw.Draw(canvas)
            for col, (image, label) in enumerate(((source, "Source photo"), (full_image, "Full scene RGB"), (isolated_image, "Isolated candidate on white"))):
                canvas.paste(image, (col * w, 35))
                draw.text((col * w + 8, 8), label, fill="black")
            name = f"camera_{index:03d}_comparison.jpg"
            canvas.save(args.output / name)
            review_rows.append({"camera_index": index, "source_image": str(source_path), "comparison": name, "isolated_crop": crop_name,
                                "remaining_scene": remainder_name})
    finally:
        model.background_color = background
        for name, parameter in originals.items():
            model.gauss_params[name] = parameter
    assert all(model.gauss_params[name] is parameter for name, parameter in originals.items())
    metadata = {"checkpoint": str(checkpoint), "step": step, "object": str(args.object.resolve()),
                "num_gaussians": int(len(rows)), "views": review_rows, "status": "pending_isolated_render_review",
                "model_restored_in_memory": True, "checkpoint_modified": False}
    metadata["derived_geometry"] = "source_xyz" in data
    (args.output / "render_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    page = pose_figure(data, pose).to_html(include_plotlyjs=True, config={"displayModeBar": True, "scrollZoom": True})
    clusters = np.unique(data["cluster_label"]).tolist() if "cluster_label" in data else []
    intro = ('<section style="font-family:sans-serif;padding:16px"><h2>Candidate inspection in isolated renders</h2>'
             f'<p>This page contains {len(rows)} Gaussians from clusters {clusters}.'
             'Inspect the screen and base for missing parts and check for fragments of the table, cables and nearby objects.</p>'
             '<p>These RGB images render the selected Gaussians. White marks the background or uncovered regions. '
             'Inspect these renders before accepting the selection. The source checkpoint is preserved.</p>')
    if "source_xyz" in data:
        if "surface_repair_mask" in data:
            intro += '<p>This candidate includes an image-guided chassis plane repair. Some positions, scales, rotations, colors and opacities changed. Original parameters are retained for comparison.</p>'
        else:
            intro += '<p>A rigid transform was applied. The source_* fields retain the original parameters. Shape, color and opacity are preserved.</p>'
    for view in review_rows:
        intro += (f'<h3>Training view {view["camera_index"]}</h3>'
                  f'<img src="{view["comparison"]}" style="width:100%;max-width:1800px" alt="Photo, full scene, isolated candidate">'
                  f'<p>Isolated object detail:</p><img src="{view["isolated_crop"]}" style="max-width:100%;max-height:550px" alt="Isolated Gaussian render">'
                  f'<p>Scene after removing the selection. Check the original location for remnants.</p>'
                  f'<img src="{view["remaining_scene"]}" style="width:100%;max-width:1000px" alt="Remaining scene after object removal">')
    intro += '<p>The plot shows all candidate centers, PCA axes and the oriented box. Coordinates use reconstruction units. The PCA axes can differ from canonical object axes.</p></section>'
    findings_path = args.output / "visual_findings.json"
    if findings_path.is_file():
        import html
        findings = json.loads(findings_path.read_text())
        intro += '<section style="padding:16px;background:#fff4de"><strong>Inspection results</strong><ul>'
        for finding in findings["observations"]:
            intro += f'<li>{html.escape(finding)}</li>'
        intro += '</ul></section>'
    (args.output / "review.html").write_text(page.replace("<body>", "<body>" + intro, 1), encoding="utf-8")
    print(f"saved {args.output / 'review.html'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--pose", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera-index", type=int, nargs="+", default=[0, 30, 60])
    p.add_argument("--max-width", type=int, default=800)
    args = p.parse_args()
    if args.max_width <= 0:
        p.error("max-width must be positive")
    render(args)
