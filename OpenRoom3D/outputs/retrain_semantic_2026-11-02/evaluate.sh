#!/usr/bin/env bash
# Evaluate the jointly retrained semantic model against the original feature checkpoint.
# Run from the project root after run.sh and after the new run's digests are in docs/scene_hashes.json.
# Both models are scored with the same local CLIP text weights, query, negative and temperature.
set -euo pipefail
train_env=${OPENROOM_TRAIN_ENV:-"$HOME/.local/share/openroom3d/train"}
analysis_env=${OPENROOM_ANALYSIS_ENV:-"$HOME/.local/share/openroom3d/analysis"}
source "$train_env/train_env.sh"
export OPENROOM_CLIP_WEIGHTS=${OPENROOM_CLIP_WEIGHTS:-"$PWD/../training-assets/models/clip/ViT-L-14-336px.state_dict.pt"}
O=outputs/retrain_semantic_2026-11-02
RUN=$(ls -d "$O"/feature_scalereg_sh0/feature-splatting/*/ | tail -1)
RUN=${RUN%/}
BASE=outputs/desk02/feature-splatting/2026-09-22_202843
MASKS=outputs/objects/laptop_sam2/tracking/masks

# Present query and an absent control, for both models.
for pair in "baseline:$BASE" "semantic:$RUN"; do
  tag=${pair%%:*}; run=${pair#*:}
  for query in laptop banana; do
    "$train_env/bin/python" scripts/export_query_gaussians.py --config "$run/config.yml" \
      --query "$query" --threshold 0.5 --output "$O/queries/$tag"
  done
done

# select_by_masks.py expects a score array keyed by label, in checkpoint order.
"$analysis_env/bin/python" -c "
import numpy as np, sys
d = np.load(sys.argv[1]); np.savez_compressed(sys.argv[2], laptop=d['similarity'])
" "$O/queries/semantic/laptop.npz" "$O/semantic_scores.npz"

"$analysis_env/bin/python" scripts/select_by_masks.py --run "$RUN" --data data/processed/desk02 \
  --masks "$MASKS" --scores "$O/semantic_scores.npz" --output "$O/selection" \
  --thresholds .5 .85 --hull .95 --radius .02 .04 .08

"$train_env/bin/python" -m nerfstudio.scripts.eval --load-config "$RUN/config.yml" \
  --output-path "$O/ns_eval.json"
