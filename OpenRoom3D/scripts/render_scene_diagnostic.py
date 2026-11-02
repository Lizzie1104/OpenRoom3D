#!/usr/bin/env python3
"""Save an original training image and its unfiltered Gaussian RGB render.

This distinguishes a Gaussian-center point display from actual RGB rendering.
It performs inference without changing a checkpoint or scene parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
if __package__:
    from .trusted_scene import eval_setup
else:
    from trusted_scene import eval_setup


def render(args):
    _, pipeline, checkpoint, step = eval_setup(args.config.resolve(), test_mode="inference")
    dataset = pipeline.datamanager.train_dataset
    if not 0 <= args.camera_index < len(dataset):
        raise ValueError("camera-index is outside the training dataset")
    camera = dataset.cameras[args.camera_index:args.camera_index + 1].to(pipeline.model.device)
    width = int(camera.width.item())
    camera.rescale_output_resolution(min(1.0, args.max_width / width))
    with torch.no_grad():
        rgb = pipeline.model.get_outputs(camera)["rgb"].clamp(0, 1).cpu().numpy()
    generated = Image.fromarray((rgb * 255).round().astype(np.uint8))
    image_path = Path(dataset.image_filenames[args.camera_index])
    original = Image.open(image_path).convert("RGB")
    original = ImageOps.contain(original, generated.size)
    canvas = Image.new("RGB", (generated.width * 2, generated.height + 35), "white")
    canvas.paste(original, (0, 35))
    canvas.paste(generated, (generated.width, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Source image", fill="black")
    draw.text((generated.width + 8, 8), "Full unfiltered 3DGS RGB render", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "step": step, "image": str(image_path),
        "camera_index": args.camera_index, "gaussian_mask": "none",
        "note": "Unfiltered RGB render; does not assess correctness of object extraction.",
    }, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=800)
    args = parser.parse_args()
    if args.max_width <= 0:
        parser.error("max-width must be positive")
    render(args)
