#!/usr/bin/env bash
# Joint geometry + language retrain of desk02 with scale regularization. Run from the project root.
# Reuses data/processed/desk02/feature_splatting_samclip_features.pt (cache keys are relative paths).
# Usage: bash outputs/retrain_semantic_2026-11-02/run.sh [experiment_name] [steps]
set -euo pipefail
train_env=${OPENROOM_TRAIN_ENV:-"$HOME/.local/share/openroom3d/train"}
source "$train_env/train_env.sh"
export OPENROOM_CLIP_WEIGHTS=${OPENROOM_CLIP_WEIGHTS:-"$PWD/../training-assets/models/clip/ViT-L-14-336px.state_dict.pt"}
O=outputs/retrain_semantic_2026-11-02
name=${1:-feature_scalereg_sh0}
steps=${2:-30000}
"$train_env/bin/python" -m nerfstudio.scripts.train feature-splatting --data data/processed/desk02 \
  --output-dir "$O" --experiment-name "$name" --max-num-iterations "$steps" \
  --steps-per-eval-image 100000 --steps-per-eval-all-images 100000 \
  --vis tensorboard --logging.steps-per-log 500 \
  --pipeline.datamanager.max-thread-workers 4 \
  --pipeline.model.sh-degree 0 --pipeline.model.use-scale-regularization True \
  nerfstudio-data --load-3D-points True --train-split-fraction 0.9
