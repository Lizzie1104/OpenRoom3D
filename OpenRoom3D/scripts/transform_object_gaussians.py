#!/usr/bin/env python3
"""Save a rigidly moved/rotated object, leaving source model and object intact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.manipulation import ObjectState


def save_state(state, output, source=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **state.arrays())
    pose = state.pose()
    pose.update(source=str(output.resolve()), original_object_source=str(source.resolve()) if source else None)
    output.with_suffix(".pose.json").write_text(json.dumps(pose, indent=2), encoding="utf-8")
    metadata = {"type": "rigidly transformed object", "source": str(source.resolve()) if source else None,
                "rotation_world": state.rotation.tolist(), "translation_world": state.translation.tolist(),
                "pivot_world": state.pivot.tolist(), "units": "reconstruction units, not meters", "checkpoint_modified": False,
                "gaussian_orientation_updated": True, "scale_rgb_opacity_preserved": True}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--pose", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--translation", type=float, nargs=3, default=[0., 0., 0.])
    p.add_argument("--rotation-deg", type=float, nargs=3, default=[0., 0., 0.], help="extrinsic xyz Euler angles around world axes")
    args = p.parse_args()
    if args.output.resolve() == args.object.resolve():
        p.error("output must differ from input object")
    pose = json.loads(args.pose.read_text())
    if Path(pose["source"]).resolve() != args.object.resolve():
        p.error("pose source does not match object")
    state = ObjectState(dict(np.load(args.object, allow_pickle=False)), pose)
    state.set_transform(Rotation.from_euler("xyz", args.rotation_deg, degrees=True).as_matrix(), args.translation)
    save_state(state, args.output, args.object)
