#!/usr/bin/env python3
"""Build a Feature Splatting SAMCLIP/DINOv2 cache at a chosen resolution, without training.

Writes <data>/feature_splatting_samclip_features.pt in the format the datamanager
expects. Image order is copied from a reference cache (the datamanager's train
then eval order), with each image path moved into --data. Use a new data
directory for each resolution so the existing 64-wide cache is never overwritten.

``--check N`` instead extracts only the first N images and prints their cosine
similarity to the reference cache (to confirm the local models reproduce it).

Run in the training environment with OPENROOM_CLIP_WEIGHTS, TORCH_HOME,
OPENROOM_MOBILESAM_REPO and OPENROOM_DINO_REPO set (see docs/setup.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path, PosixPath

import torch
import torch.nn.functional as F

from feature_splatting.feature_extractor import batch_extract_feature
from feature_splatting.feature_extractor_cfg import SAMCLIPArgs


def load_cache(path):
    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path, weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("data/processed/desk02/feature_splatting_samclip_features.pt"))
    parser.add_argument("--final-res", type=int, default=64)
    parser.add_argument("--obj-res", type=int, default=100)
    parser.add_argument("--check", type=int, default=0)
    args = parser.parse_args()
    reference = load_cache(args.reference)
    names = [Path(p).name for p in reference["image_fnames"]]
    data = args.data  # keep it relative, as Nerfstudio records it (train with the same --data string)
    paths = [data / "images" / n for n in names]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[0]}")
    extract_args = type("SAMCLIPArgs", (SAMCLIPArgs,), {"final_feat_res": args.final_res, "obj_feat_res": args.obj_res})
    output = data / "feature_splatting_samclip_features.pt"
    if args.check:
        features = batch_extract_feature(paths[:args.check], extract_args)
        for key, value in features.items():
            ref = reference["feature_dict"][key][:args.check]
            if value.shape[-2:] != ref.shape[-2:]:
                value = F.interpolate(value, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            valid = (ref.abs().sum(1) > 0) & (value.abs().sum(1) > 0)
            cos = F.cosine_similarity(value, ref, dim=1)[valid]
            print(f"{key}: shape {tuple(features[key].shape)} vs {tuple(ref.shape)}; "
                  f"cosine mean {cos.mean():.4f}, 5th percentile {cos.quantile(.05):.4f}")
        return
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new data directory")
    features = batch_extract_feature(paths, extract_args)
    torch.save({"args": extract_args.id_dict(), "image_fnames": [str(p) for p in paths], "feature_dict": features}, output)
    print({k: tuple(v.shape) for k, v in features.items()}, "->", output)


if __name__ == "__main__":
    main()
