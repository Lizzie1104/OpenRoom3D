#!/usr/bin/env bash
# A/B retrain of desk02 geometry. Run from the project root.
set -euo pipefail
train_env=${OPENROOM_TRAIN_ENV:-"$HOME/.local/share/openroom3d/train"}
source "$train_env/train_env.sh"
O=outputs/retrain_2026-11-01
run() {
  name=$1; shift
  "$train_env/bin/python" -m nerfstudio.scripts.train splatfacto --data "$PWD/data/processed/desk02" \
    --output-dir "$PWD/$O" --experiment-name "$name" --max-num-iterations 30000 \
    --steps-per-eval-image 100000 --steps-per-eval-all-images 100000 \
    --vis tensorboard --logging.steps-per-log 500 \
    --pipeline.datamanager.max-thread-workers 4 "$@" \
    nerfstudio-data --load-3D-points True --train-split-fraction 0.9
}
run control_sh0 --pipeline.model.sh-degree 0
run scalereg_sh0 --pipeline.model.sh-degree 0 --pipeline.model.use-scale-regularization True
