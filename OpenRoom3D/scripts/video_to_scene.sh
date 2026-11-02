#!/usr/bin/env bash
# Build a scene from a video: bash scripts/video_to_scene.sh <name> <video> [steps]
# PHOTOS=1 treats the second argument as a photo folder.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
name=$1 source_path=$(realpath -- "$2") steps=${3:-30000}
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ && "$steps" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "Invalid scene name or training step count" >&2; exit 2; }
train_env=${OPENROOM_TRAIN_ENV:-"$HOME/.local/share/openroom3d/train"}
video_env=${OPENROOM_VIDEO_ENV:-"$HOME/.local/share/openroom3d/video"}
data=data/processed/$name output=outputs/scenes/$name
flag=--video
[ "${PHOTOS:-0}" = 1 ] && flag=--photos
"$video_env/bin/python" scripts/video_to_colmap.py "$flag" "$source_path" --output "$data"
source "$train_env/train_env.sh"
"$train_env/bin/python" - "$data" <<'PY'
import sys
from pathlib import Path
from nerfstudio.process_data.colmap_utils import colmap_to_json
root = Path(sys.argv[1])
print("registered frames in transforms.json:", colmap_to_json(root / "sparse/0", root))
PY
"$train_env/bin/python" -m nerfstudio.scripts.train splatfacto --data "$PWD/$data" --output-dir "$PWD/$output" --experiment-name "$name" \
  --max-num-iterations "$steps" --steps-per-eval-image 100000 --steps-per-eval-all-images 100000 \
  --vis tensorboard --logging.steps-per-log 500 --pipeline.datamanager.max-thread-workers 4 \
  --pipeline.model.sh-degree 0 --pipeline.model.use-scale-regularization True \
  nerfstudio-data --load-3D-points True --train-split-fraction 0.9
shopt -s nullglob
configs=("$output/$name"/splatfacto/*/config.yml)
(( ${#configs[@]} )) || { echo "No trained configuration" >&2; exit 1; }
config=${configs[-1]}
printf 'Scene ready: %s\n' "$config"
