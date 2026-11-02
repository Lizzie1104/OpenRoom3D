# Negative prompts

`compare_negatives.py` scores four queries under four negative schemes on the original checkpoint and the jointly retrained `feature_scalereg_sh0`. Temperature is 0.05 throughout. Average precision (AP) is threshold-free and uses each model's own SAM 2 ownership/hull selection as a pseudo-label. The mask columns report text ≥ t plus the largest DBSCAN cluster on the ten held-out frames, with t chosen on every ninth training frame. Full numbers, including every threshold, are in `results.json`.

| Scheme | Negatives |
|---|---|
| softmax: object | `object` (Viewer default) |
| softmax: canonical | `object, things, stuff, texture` in one softmax |
| pairwise-min: canonical | the same phrases, LERF pairwise minimum |
| pairwise-min: canonical + scene | adds table, desk, chair, floor, wall, computer mouse, water bottle, cable, paper, mat |

## Original checkpoint

| Scheme | laptop ≥ 0.5 | laptop AP | synonym AP | television ≥ 0.5 | banana ≥ 0.5 | t | IoU | capture | off-mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| softmax: object | 31.2% | 0.415 | 0.376 | 195 | 9 | 0.7 | 0.761 | 0.524 | 0.143 |
| softmax: canonical | 7.3% | 0.403 | 0.364 | 0 | 0 | 0.4 | 0.765 | 0.576 | 0.158 |
| pairwise-min: canonical | 31.0% | 0.418 | 0.377 | 185 | 2 | 0.7 | 0.761 | 0.525 | 0.142 |
| pairwise-min: canonical + scene | 13.0% | **0.491** | **0.488** | 16 | 0 | 0.6 | **0.775** | 0.549 | **0.135** |

## Jointly retrained checkpoint

| Scheme | laptop ≥ 0.5 | laptop AP | synonym AP | television ≥ 0.5 | banana ≥ 0.5 | t | IoU | capture | off-mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| softmax: object | 52.4% | 0.350 | 0.305 | 1,636 | 0 | 0.8 | 0.749 | 0.510 | 0.149 |
| softmax: canonical | 22.3% | 0.347 | 0.300 | 0 | 0 | 0.5 | 0.755 | 0.589 | 0.158 |
| pairwise-min: canonical | 51.8% | 0.357 | 0.309 | 1,570 | 0 | 0.8 | 0.749 | 0.509 | 0.148 |
| pairwise-min: canonical + scene | 28.9% | 0.328 | 0.316 | 3 | 0 | 0.7 | **0.757** | 0.542 | 0.145 |

## Findings

`things`, `stuff` and `texture` add nothing under the pairwise minimum: `object` is almost always the binding comparison, so the scores match the Viewer default. In one softmax the same phrases only dilute every score. The ranking (AP) is unchanged, absent queries go silent and the best threshold moves down. That is recalibration, not better separation.

Scene categories are the useful negatives. On the original checkpoint they raise laptop AP by 18% and synonym AP by 30%, cut the look-alike television responses from 195 to 16 and give the best held-out text selection. Held-out IoU also stays above 0.7 for every threshold from 0.4 to 0.7, against 0.5 to 0.8 before, so the choice of threshold matters less. On the retrained model they suppress the television responses almost completely (1,636 to 3) and slightly raise IoU, but AP falls. Its features rank the laptop below the original's under every scheme, consistent with the broader scores reported in `../README.md`.

The scene vocabulary was written by hand for this tabletop and must be reported as manual input. Do not reuse it for another scene without inspecting the scene. The default exporter behavior is unchanged; use `--score-mode pairwise-min --negative "object,things,stuff,texture,<scene categories>"` to apply the recommended scheme.
