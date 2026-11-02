# Reconstruction diagnosis

The torch compositor measures per-pixel Gaussian contributions at 640 by 360. It is an approximation of the project renderer and differs from the official evaluator. A box derived from the strongest laptop cluster and a table-colored plane serve as approximate geometric references. Neither supplies human segmentation labels.

The 64-by-36 semantic features blur object boundaries. At threshold 0.5 the laptop response covers one third of all points, and an absent query still selects thousands of points. Mean held-out RGB quality was 31.24 dB, while a substantial fraction of visible object appearance came from elongated or spatially displaced Gaussians. One moving occluder degraded a frame. Camera ordering and feature-cache ordering were checked; zero feature maps did not account for the failure.

Use `d7.py` to generate the reference plane, then `d8.py` for scores and `d9.py` for measurements. The `rlib.py` module contains the diagnostic renderer. Consult [the retained measurements](../../docs/experiments.md) before interpreting changes in the selection.
