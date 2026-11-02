# Joint geometry and language retrain

`feature_scalereg_sh0` trains Feature Splatting with scale regularization and SH degree zero on desk02, reusing the cached SAMCLIP features (`final_feat_res` 64). Unlike `retrain_2026-11-01`, its language scores come from its own distilled features rather than a nearest-neighbor transfer. Reproduce with `bash outputs/retrain_semantic_2026-11-02/run.sh`, then `evaluate.sh`. The run's digests were added to `docs/scene_hashes.json`; the previous file is kept as `scene_hashes.before.json`.

The CLIP text encoder loads local weights from `OPENROOM_CLIP_WEIGHTS` (`training-assets/models/clip/ViT-L-14-336px.state_dict.pt`, converted from the bundled safetensors). Scores exported this way differ from `diagnosis_2026-10-31/scores.npz` (open_clip, fp32) by up to 0.09 on the original checkpoint: 69,340 versus 74,721 points at 0.5. Compare score counts within one exporter.

## Results

Held-out metrics, Nerfstudio evaluator through `trusted_scene.eval_setup`:

| Model | PSNR | SSIM | LPIPS | Gaussians | Aspect > 100 |
|---|---:|---:|---:|---:|---:|
| Splatfacto + scale reg (11-01) | 30.41 | 0.951 | 0.095 | 177,871 | 0% |
| Feature Splatting + scale reg (this run) | 30.33 | 0.951 | 0.094 | 177,804 | 0% |

Text scores, same exporter, negative `object`, temperature 0.05:

| Model | laptop ≥ 0.5 | laptop ≥ 0.8 | banana ≥ 0.5 |
|---|---:|---:|---:|
| Original | 69,340 (31%) | 10,967 (4.9%) | 9 |
| This run | 93,172 (52%) | 31,405 (17.7%) | 0 |

Held-out agreement with SAM 2 masks (IoU / capture / coverage / off-mask weight):

| Selection | Original | Scale reg, transferred scores | Scale reg, joint scores |
|---|---|---|---|
| Text ≥ 0.5 + largest DBSCAN | 0.734 / 0.783 / 0.892 / 0.217 | 0.727 / 0.726 / 0.871 / 0.208 | 0.681 / 0.786 / 0.882 / 0.270 |
| Text ≥ 0.8 + largest DBSCAN | 0.714 / 0.306 / 0.753 / 0.091 | 0.528 / 0.200 / 0.548 / 0.082 | **0.749 / 0.510 / 0.832** / 0.149 |
| Ownership ≥ 0.85, hull ≥ 0.95 | 0.811 / 0.844 / 0.900 / 0.132 | 0.830 / 0.833 / 0.879 / 0.082 | 0.817 / 0.824 / 0.876 / 0.098 |

The original column used the open_clip scores, so text rows across columns differ slightly in exporter as well as model.

## Interpretation

Joint training repairs the high-confidence text cluster: at 0.8 the largest cluster is the best text-only selection measured so far, capturing 51% of the laptop's mask weight against 20% with transferred scores. The scores are also broader: half the scene exceeds 0.5, so the 0.5 operating point is worse and thresholds are not transferable between models. The absent control stays silent in both models.

The final ownership/hull selection does not use text scores, so it measures geometry and SAM 2 masks only. Its small change from the 11-01 scale-regularized run reflects training variance between two geometrically similar models, not the language features. The language branch matters where text seeds the tracker and where no masks exist.
