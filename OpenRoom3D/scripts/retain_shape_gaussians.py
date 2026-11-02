#!/usr/bin/env python3
"""Preserve original Gaussian contributions inside an object shape prior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.shape_retention import retain_by_shape


def export(args):
    raw, seed = np.load(args.raw, allow_pickle=False), np.load(args.seed, allow_pickle=False)
    fields, pose, metadata = retain_by_shape(raw, seed, args.padding, args.interior_threshold,
                                           args.boundary_threshold, args.extent_sigma, args.max_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    metadata.update(raw_source=str(args.raw.resolve()), seed_source=str(args.seed.resolve()), label=args.label,
                    object_id=args.output.stem, id_scope="this model and selection")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pose.update(source=str(args.output.resolve()), semantic_verification="not performed by this script")
    args.output.with_suffix(".pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    print(f"saved {args.output}: {metadata['counts']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", default="object")
    p.add_argument("--padding", type=float, default=.01)
    p.add_argument("--interior-threshold", type=float, default=.35)
    p.add_argument("--boundary-threshold", type=float, default=.60)
    p.add_argument("--extent-sigma", type=float, default=1.)
    p.add_argument("--max-scale", type=float, default=.06)
    export(p.parse_args())
