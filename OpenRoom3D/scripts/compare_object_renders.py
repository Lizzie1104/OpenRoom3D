#!/usr/bin/env python3
"""Build a same-camera before/after render gallery and a diagnostic ROI metric."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw, ImageOps


def compare(args):
    before = json.loads((args.before / "render_manifest.json").read_text())
    after = json.loads((args.after / "render_manifest.json").read_text())
    if before["checkpoint"] != after["checkpoint"]:
        raise ValueError("Compare renders derived from the same checkpoint")
    if [v["camera_index"] for v in before["views"]] != [v["camera_index"] for v in after["views"]]:
        raise ValueError("Render camera lists must match")
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    for old_view, new_view in zip(before["views"], after["views"]):
        old_image = Image.open(args.before / old_view["comparison"]).convert("RGB")
        new_image = Image.open(args.after / new_view["comparison"]).convert("RGB")
        if old_image.size != new_image.size:
            raise ValueError("Render resolutions must match")
        width = old_image.width // 3
        original = old_image.crop((0, 35, width, old_image.height))
        old = old_image.crop((2 * width, 35, 3 * width, old_image.height))
        new = new_image.crop((2 * width, 35, 3 * width, new_image.height))
        nonwhite = (np.asarray(old).mean(axis=2) < 240) | (np.asarray(new).mean(axis=2) < 240)
        ys, xs = np.where(nonwhite)
        box = (max(0, int(xs.min()) - 10), max(0, int(ys.min()) - 10),
               min(width, int(xs.max()) + 11), min(old.height, int(ys.max()) + 11))
        tiles = [ImageOps.contain(image.crop(box), (500, 420)) for image in (original, old, new)]
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (1500, height + 40), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (tile, label) in enumerate(zip(tiles, ("Source photo", "Before: L1 extracted", "After: image-guided plane repair"))):
            canvas.paste(tile, (500 * col + (500 - tile.width) // 2, 40))
            draw.text((500 * col + 8, 10), label, fill="black")
        name = f"before_after_camera_{old_view['camera_index']:03d}.jpg"
        canvas.save(args.output / name)
        images.append((old_view["camera_index"], name))

    recipe = json.loads(args.recipe.read_text())
    index = recipe["camera_index"]
    view = next(v for v in before["views"] if v["camera_index"] == index)
    image = Image.open(args.before / view["comparison"]).convert("RGB")
    old_rgb = np.asarray(image.crop(recipe["isolated_crop"])).astype(np.float64) / 255
    photo = np.asarray(image.crop(recipe["photo_crop"])).astype(np.float64) / 255
    arrays = np.load(args.after / f"camera_{index:03d}_render_arrays.npz")
    new_rgb = arrays["isolated_rgb"]
    height, width = old_rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    roi = Polygon(recipe["palm_polygon"]).contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(height, width)
    roi &= photo.mean(axis=2) < recipe["source_dark_limit"]
    counts = {"roi_pixels": int(roi.sum()), "before_white_pixels": int(((old_rgb.mean(axis=2) > .85) & roi).sum()),
              "after_white_pixels": int(((new_rgb.mean(axis=2) > .85) & roi).sum())}
    metric = {"definition": "RGB mean > 0.85 inside manually specified dark palm-rest ROI",
              "camera_index": index, "counts": counts,
              "before_white_fraction": counts["before_white_pixels"] / max(counts["roi_pixels"], 1),
              "after_white_fraction": counts["after_white_pixels"] / max(counts["roi_pixels"], 1),
              "limitations": "Reference-view fitting diagnostic; not IoU, ground-truth accuracy or unseen-view completeness. Before uses saved JPEG, after uses float RGB.",
              "recipe": str(args.recipe.resolve())}
    (args.output / "comparison_metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
    sections = ''.join(f'<h2>Camera view {index}</h2><img src="{html.escape(name)}" alt="Source photo, before, after" style="width:100%;max-width:1500px">'
                       for index, name in images)
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Laptop surface repair comparison</title>'
        '<body style="font-family:sans-serif;margin:24px;background:#f7f8fa">'
        '<h1>Laptop surface repair at matched viewpoints</h1>'
        '<p>Each row shows the source image, L1 extraction and repaired object.</p>'
        '<p>The repair filters nearby tabletop points and fits a plane to the chassis using the image. Gaussian positions and colors change. '
        'The result includes surface edits. The training checkpoint is preserved.</p>'
        f'<p>White fraction of the reference palm-rest area: {metric["before_white_fraction"]:.1%} → {metric["after_white_fraction"]:.1%}.'
        'The region was selected manually and also used for fitting. Its white fraction measures appearance.</p>'
        '<p>The main gap is smaller. Edge artifacts, blur and small gaps remain. The rear and unseen surfaces have not been validated.</p>'
        f'{sections}<p><a href="review.html">Inspect isolated renders, XYZ axes and the oriented box</a></p></body></html>'
    )
    (args.output / "before_after.html").write_text(document, encoding="utf-8")
    print(f"saved {args.output / 'before_after.html'}; ROI white fraction {metric['before_white_fraction']:.3f} -> {metric['after_white_fraction']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args())
