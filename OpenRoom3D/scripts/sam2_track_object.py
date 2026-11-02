#!/usr/bin/env python3
"""Track one object through all training frames with SAM 2.

The prompt comes from projecting semantic-seed Gaussians into a reference frame.
Frames are undistorted first so masks align with pinhole Gaussian projection.
Masks are inferred, not ground truth. Checkpoints and Gaussians are not changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.gaussian_scene import load_cameras


def prompt_from_seed(xyz, camera, intrinsics, width, height, margin=.04):
    """Box and interior point from seed Gaussians projected into one camera."""
    rotation = camera[:3, :3] @ np.diag([1., -1., -1.])
    points = (xyz - camera[:3, 3]) @ rotation
    points = points[points[:, 2] > .01]
    pixels = points @ intrinsics.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    if inside.sum() < 100:
        return None
    pixels = pixels[inside]
    lo, hi = np.quantile(pixels, [.01, .99], axis=0)
    pad = (hi - lo) * margin
    box = np.r_[np.maximum(lo - pad, 0), np.minimum(hi + pad, [width - 1, height - 1])]
    return {"box": box, "point": np.median(pixels, axis=0), "visible": int(inside.sum())}


def choose_reference(seed_xyz, cameras, intrinsics, width, height):
    """Prefer the frame where the seed projects inside the image with the largest box."""
    best, best_area = None, -1.
    for index, camera in enumerate(cameras):
        prompt = prompt_from_seed(seed_xyz, camera["c2w"], intrinsics, width, height)
        if prompt is None:
            continue
        box = prompt["box"]
        if box[0] <= 2 or box[1] <= 2 or box[2] >= width - 3 or box[3] >= height - 3:
            continue  # cropped objects give misleading prompts
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        if area > best_area:
            best, best_area = index, area
    if best is None:
        raise ValueError("no frame shows the whole seed; pass --reference-frame")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Feature Splatting run directory")
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--seed", type=Path, required=True, help="NPZ containing seed xyz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int)
    args = parser.parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    cameras, intrinsics, distortion, (width, height) = load_cameras(args.data, args.config)
    seed_xyz = np.load(args.seed, allow_pickle=False)["xyz"]
    frames_dir = args.output / "frames"
    masks_dir = args.output / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, camera in enumerate(cameras):
        target = frames_dir / f"{index:05d}.jpg"
        if not target.exists():
            image = cv2.imread(str(args.data / camera["name"]))
            cv2.imwrite(str(target), cv2.undistort(image, intrinsics, np.r_[distortion, 0.]),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
    reference = args.reference_frame
    if reference is None:
        reference = choose_reference(seed_xyz, cameras, intrinsics, width, height)
    prompt = prompt_from_seed(seed_xyz, cameras[reference]["c2w"], intrinsics, width, height)
    if prompt is None:
        raise ValueError(f"seed not visible in frame {reference}")
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device="cuda")
    masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state, frame_idx=reference, obj_id=1, box=prompt["box"])
        for reverse in (False, True):
            for frame, _, logits in predictor.propagate_in_video(state, start_frame_idx=reference, reverse=reverse):
                masks[frame] = (logits[0, 0] > 0).cpu().numpy()
    if len(masks) != len(cameras):
        raise RuntimeError(f"tracked {len(masks)} of {len(cameras)} frames")
    rows = []
    for index in range(len(cameras)):
        mask = masks[index]
        cv2.imwrite(str(masks_dir / f"{index:05d}.png"), mask.astype(np.uint8) * 255)
        rows.append({"index": index, "name": cameras[index]["name"], "split": cameras[index]["split"],
                     "mask_fraction": float(mask.mean())})
    tiles = []
    for index in np.linspace(0, len(cameras) - 1, 20).astype(int):
        image = cv2.imread(str(frames_dir / f"{index:05d}.jpg"))
        image[masks[index]] = (.5 * image[masks[index]] + .5 * np.array([60, 255, 60])).astype(np.uint8)
        tile = cv2.resize(image, (320, 180))
        cv2.putText(tile, str(index), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        tiles.append(tile)
    cv2.imwrite(str(args.output / "tracking_overview.jpg"),
                np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, 20, 5)]))
    (args.output / "manifest.json").write_text(json.dumps(
        {"reference_frame": reference, "box_prompt": prompt["box"].tolist(), "sam2_config": args.sam2_config,
         "mask_source": "SAM 2 inference; not ground truth", "frames": rows}, indent=2), encoding="utf-8")
    fractions = np.array([row["mask_fraction"] for row in rows])
    print(f"reference {reference}; mask fraction median {np.median(fractions):.3f}, "
          f"zero-mask frames {int((fractions == 0).sum())}")


if __name__ == "__main__":
    main()
