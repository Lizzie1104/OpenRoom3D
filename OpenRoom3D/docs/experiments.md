# Measurements and limitations

These measurements are from the retained experiments. The overlap metric compares rendered selections to propagated SAM 2 masks, without human-labeled ground truth.

## Geometry and semantics

The baseline has 222,327 Gaussians. The project renderer produced a mean 31.24 dB on ten held-out views and 33.18 dB on 91 training views. A laptop-score threshold of 0.5 selected 74,721 points. Only 20.1% fell inside a high-confidence-cluster box, which is itself an approximate proxy. Thresholds of 0.6, 0.7 and 0.8 raised that fraction to 31.5%, 45.3% and 63.4%. An absent query still produced 11,012 responses over 0.5. Feature maps were 64 by 36 for 1280 by 720 images, limiting boundary resolution. A moving foreground occluder affected one frame.

## Segmentation

The largest cluster at score 0.8 supplied a box for frame 39. SAM 2.1 tracked masks through 101 images. Ownership used 91 training views. Threshold selection used a fold from those training views; the ten held-out views were used for reporting.

| Selection | Gaussians | IoU | Capture | Coverage | Off-mask weight |
|---|---:|---:|---:|---:|---:|
| Text ≥ 0.5 | 74,721 | 0.079 | 0.903 | 1.000 | 0.919 |
| Text + largest cluster | 13,427 | 0.734 | 0.784 | 0.892 | 0.217 |
| SAM ownership ≥ 0.85 | 14,238 | 0.428 | 0.921 | 0.985 | 0.576 |
| Ownership ≥ 0.85 and hull ≥ 0.95 | 13,200 | 0.811 | 0.844 | 0.900 | 0.132 |
| Above + radius 0.02 recovery | 13,383 | 0.782 | 0.863 | 0.926 | 0.183 |
| Above + radius 0.08 recovery | 13,789 | 0.681 | 0.893 | 0.956 | 0.305 |

The stricter selection removes much of the table contamination. Recovery fills parts of the palm rest but reintroduces background. The later rerun below uses the same ownership and hull thresholds for all three models.

| Metric | Original | Retrained control | Retrained with scale regularization |
|---|---:|---:|---:|
| Held-out PSNR, project renderer | 31.24 | 31.12 | 30.89 |
| Held-out PSNR, Nerfstudio | unavailable | 30.72 | 30.41 |
| SSIM / LPIPS, Nerfstudio | unavailable | 0.952 / 0.089 | 0.951 / 0.095 |
| Gaussians | 222,327 | 218,753 | 177,871 |
| Aspect ratio greater than 100 | 52.4% | 51.8% | 0% |
| Selection IoU | 0.811 | 0.824 | 0.830 |
| Capture / coverage | 0.844 / 0.900 | 0.853 / 0.910 | 0.833 / 0.879 |
| Off-mask weight | 0.132 | 0.129 | 0.082 |

Scale regularization eliminated the most elongated Gaussians and reduced off-mask weight. Coverage declined and small holes remain. In the geometry-only reruns the displayed language scores were transferred from the original by nearest neighbor. They do not measure newly trained language features.

## Background

Around 98.5% of the original footprint had no direct surface observation. The baseline layer contains 9,488 Gaussians, of which 7,751 are inferred. Across 12 half-sized holes, extrapolation achieved 33.07 dB, smoothing 33.23 dB and a mean-color baseline 27.43 dB. The regularized scene has 9,088 background points, of which 7,410 are inferred; the same comparison produced 35.94, 36.08 and 27.58 dB. Validation at the size of the full footprint was not possible. Planar filling cannot recover hidden cables or accurate view-dependent highlights.

## Negative prompts and joint retraining

A jointly trained Feature Splatting model with scale regularization (`outputs/retrain_semantic_2026-11-02`) reached 30.33 dB, SSIM 0.951 and LPIPS 0.094 in the Nerfstudio evaluator. Using its own scores, text ≥ 0.8 plus the largest cluster reached IoU 0.749 with capture 0.510. Transferred scores reached 0.528 and 0.200. Its scores are broader: 52% of points exceed 0.5 for the laptop.

With the LERF pairwise minimum, the canonical phrases `things, stuff, texture` behave like `object` alone. Adding hand-written scene categories as negatives raised laptop average precision on the original checkpoint from 0.415 to 0.491 against the SAM 2 pseudo-label. It reduced the look-alike `television` responses from 195 to 16 and raised held-out IoU from 0.761 to 0.775. On the retrained model the IoU gain was small and average precision fell. Details are in that directory's `negatives/README.md`.
