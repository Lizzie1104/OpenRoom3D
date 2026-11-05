#!/usr/bin/env bash
# Laptop manipulation on the scale-regularized retrain, with the inferred table layer and tabletop editing. Set PYTHON to override the bundled environment.
set -euo pipefail
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_project_root"
exec "${PYTHON:-$HOME/.local/share/openroom3d/viewer/bin/python}" scripts/object_pose_viewer.py \
  --scene outputs/retrain_2026-11-01/selection_scalereg/scene.npz \
  --inferred-background outputs/retrain_2026-11-01/inferred_table/inferred_background.npz \
  --plane outputs/retrain_2026-11-01/inferred_table/plane.npz \
  --object 'Laptop (retrained geometry and SAM 2 selection)=outputs/retrain_2026-11-01/selection_scalereg/obj_laptop.npz' \
  --port 7013 "$@"
