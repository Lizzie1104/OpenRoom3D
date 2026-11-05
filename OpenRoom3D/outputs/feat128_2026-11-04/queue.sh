#!/usr/bin/env bash
# 128-wide SAMCLIP features (obj_feat_res 256) on desk02; same training and evaluation as ../ablation_2026-11-04.
# Each entry: train Feature Splatting (SH0, cached SAMCLIP features), register its digests,
# then write eval/<name>.json with scripts/evaluate_run.py.
# Build the cache first with scripts/extract_features.py --data data/processed/desk02_feat128 --final-res 128 --obj-res 256.
# Training never re-extracts here: MobileSAM/DINO repositories are not set, so a mismatched cache fails instead of being overwritten.
# Usage: bash outputs/feat128_2026-11-04/queue.sh name:scalereg:feat_weight:seed [...]
set -euo pipefail
train_env=${OPENROOM_TRAIN_ENV:-"$HOME/.local/share/openroom3d/train"}
source "$train_env/train_env.sh"
export OPENROOM_CLIP_WEIGHTS=${OPENROOM_CLIP_WEIGHTS:-"$PWD/../training-assets/models/clip/ViT-L-14-336px.state_dict.pt"}
O=outputs/feat128_2026-11-04
for entry in "$@"; do
  IFS=: read -r name scalereg weight seed <<< "$entry"
  if [ -f "$O/eval/$name.json" ]; then echo "skip $name"; continue; fi
  echo "=== $name scalereg=$scalereg feat_loss_weight=$weight seed=$seed $(date +%T)"
  "$train_env/bin/python" -m nerfstudio.scripts.train feature-splatting --data data/processed/desk02_feat128 \
    --output-dir "$O/runs" --experiment-name "$name" --max-num-iterations 30000 \
    --steps-per-eval-image 100000 --steps-per-eval-all-images 100000 \
    --vis tensorboard --logging.steps-per-log 500 --machine.seed "$seed" \
    --pipeline.datamanager.max-thread-workers 4 --pipeline.datamanager.final-feat-res 128 --pipeline.datamanager.obj-feat-res 256 --pipeline.model.sh-degree 0 \
    --pipeline.model.use-scale-regularization "$scalereg" --pipeline.model.feat-loss-weight "$weight" \
    nerfstudio-data --load-3D-points True --train-split-fraction 0.9 > "$O/logs/$name.train.log" 2>&1
  config=$(ls -d "$O/runs/$name"/feature-splatting/*/config.yml | tail -1)
  "$train_env/bin/python" - "$config" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
from trusted_scene import files_for
path = "docs/scene_hashes.json"
hashes = json.load(open(path))
hashes.update(files_for(sys.argv[1]))
open(path, "w").write(json.dumps(dict(sorted(hashes.items())), indent=2) + "\n")
EOF
  "$train_env/bin/python" scripts/evaluate_run.py --config "$config" --output "$O/eval/$name.json" \
    > "$O/logs/$name.eval.log" 2>&1
  echo "$name $(tail -1 "$O/logs/$name.eval.log")"
done
