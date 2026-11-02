#!/usr/bin/env python3
"""Generate multi-view object masks with already cached MobileSAMV2 weights.

The existing semantic seed provides image box prompts. These are inferred masks,
not ground truth. No installation/download and no Gaussian geometry changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
if __package__:
    from .trusted_scene import eval_setup
else:
    from trusted_scene import eval_setup
from nerfstudio.models.splatfacto import get_viewmat

from render_object_diagnostic import validate_selection


def cached_predictor(cache, device):
    for filename in ("efficientvit_l2.pt", "Prompt_guided_Mask_Decoder.pt"):
        if not (cache / filename).is_file():
            raise FileNotFoundError(f"Missing cached weights: {cache / filename}; no automatic download")
    sys.path.insert(0, str(cache))
    from mobilesamv2 import sam_model_registry, SamPredictor
    from mobilesamv2.modeling import Sam
    decoder = sam_model_registry["PromptGuidedDecoder"](str(cache / "Prompt_guided_Mask_Decoder.pt"))
    encoder = sam_model_registry["efficientvit_l2"](str(cache / "efficientvit_l2.pt"))
    model = Sam(encoder, decoder["PromtEncoder"], decoder["MaskDecoder"]).to(device).eval()
    return SamPredictor(model)


def build(args):
    _, pipeline, checkpoint, step = eval_setup(args.config.resolve(), test_mode="inference")
    model, dataset = pipeline.model, pipeline.datamanager.train_dataset
    seed = np.load(args.seed, allow_pickle=False)
    validate_selection(model, seed)
    predictor = cached_predictor(args.sam_cache, model.means.device)
    args.output.mkdir(parents=True, exist_ok=True)
    entries, tiles = [], []
    for index in args.camera_index:
        if not 0 <= index < len(dataset):
            raise ValueError(f"Camera {index} outside training dataset ({len(dataset)})")
        camera = dataset.cameras[index:index + 1].to(model.means.device)
        camera.rescale_output_resolution(min(1., args.width / int(camera.width.item())))
        width, height = int(camera.width.item()), int(camera.height.item())
        view = get_viewmat(camera.camera_to_worlds)[0].cpu().numpy()
        intrinsics = camera.get_intrinsics_matrices()[0].cpu().numpy()
        photo_path = Path(dataset.image_filenames[index])
        photo = np.asarray(Image.open(photo_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
        centers = seed["xyz"] @ view[:3, :3].T + view[:3, 3]
        pixels = centers @ intrinsics.T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        visible = ((centers[:, 2] > .01) & (pixels[:, 0] > 0) & (pixels[:, 0] < width)
                   & (pixels[:, 1] > 0) & (pixels[:, 1] < height))
        if visible.sum() < 100:
            raise ValueError(f"Object seed not sufficiently visible in camera {index}")
        lo, hi = np.quantile(pixels[visible], [.005, .995], axis=0)
        margin = np.maximum((hi - lo) * .07, 5)
        box = np.concatenate((np.maximum(lo - margin, 0), np.minimum(hi + margin, [width - 1, height - 1])))
        predictor.set_image(photo)
        with torch.no_grad():
            # The cached prompt-guided decoder was trained for box prompts.
            boxes = predictor.transform.apply_boxes(box[None], predictor.original_size)
            boxes = torch.as_tensor(boxes, device=model.means.device, dtype=torch.float32)
            sparse, dense = predictor.model.prompt_encoder(points=None, boxes=boxes, masks=None)
            lowres, quality = predictor.model.mask_decoder(
                image_embeddings=predictor.features, image_pe=predictor.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False, simple_type=True)
            logits = predictor.model.postprocess_masks(lowres, predictor.input_size, predictor.original_size)
            mask = (logits[0, 0] > predictor.model.mask_threshold).cpu().numpy()
        if mask.sum() < 30 or mask.mean() > .6:
            raise ValueError(f"Suspicious inferred mask in camera {index}: {int(mask.sum())} pixels; inspect prompts")
        output_name = f"camera_{index:03d}.npz"
        np.savez_compressed(args.output / output_name, mask=mask, photo=photo, viewmat=view.astype(np.float32),
                            intrinsics=intrinsics.astype(np.float32), box_prompt=box.astype(np.float32),
                            camera_to_world=camera.camera_to_worlds[0].cpu().numpy(),
                            camera_index=index)
        overlay = photo.copy()
        overlay[mask] = (.6 * overlay[mask] + .4 * np.array([0, 255, 90])).astype(np.uint8)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(box), outline=(255, 180, 0), width=2)
        image.save(args.output / f"camera_{index:03d}_mask.jpg")
        tile = image.copy()
        tile.thumbnail((400, 225))
        label = Image.new("RGB", (400, 250), "white")
        label.paste(tile, (0, 25))
        ImageDraw.Draw(label).text((5, 5), f"Camera {index}: mask pixels {int(mask.sum())}", fill="black")
        tiles.append(label)
        entries.append({"camera_index": index, "file": output_name, "source_image": str(photo_path),
                        "mask_pixels": int(mask.sum()), "box_prompt": box.tolist(),
                        "status": "SAM-inferred mask; not ground truth"})
        print(f"camera {index}: {int(mask.sum())} mask pixels, box {box.round(1).tolist()}", flush=True)
    sheet = Image.new("RGB", (1200, 250 * ((len(tiles) + 2) // 3)), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3) * 400, (i // 3) * 250))
    sheet.save(args.output / "mask_overview.jpg")
    metadata = {"config": str(args.config.resolve()), "checkpoint": str(checkpoint), "step": step,
                "seed": str(args.seed.resolve()), "method": "cached MobileSAMV2 with projected semantic-seed box prompts",
                "width": args.width, "views": entries, "gaussian_parameters_modified": False,
                "mask_source": "inferred from source photos; not GT or user-verified annotation"}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.output / 'mask_overview.jpg'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera-index", type=int, nargs="+", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--sam-cache", type=Path, default=Path.home() / ".cache/torch/hub/RogerQi_MobileSAMV2_main")
    args = p.parse_args()
    if args.width <= 0 or len(set(args.camera_index)) != len(args.camera_index):
        p.error("width must be positive; camera indices must be unique")
    build(args)
