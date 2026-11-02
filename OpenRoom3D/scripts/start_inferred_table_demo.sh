#!/usr/bin/env bash
# Laptop manipulation with the inferred table layer. Set PYTHON to override the bundled environment.
set -euo pipefail
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_project_root"
exec "${PYTHON:-$HOME/.local/share/openroom3d/viewer/bin/python}" scripts/object_pose_viewer.py \
  --scene outputs/objects/laptop_sam2/selection/scene.npz \
  --inferred-background outputs/objects/laptop_sam2/inferred_table/inferred_background.npz \
  --object 'Laptop (SAM 2 multiview selection)=outputs/objects/laptop_sam2/selection/obj_laptop.npz' \
  --port 7012 "$@"
