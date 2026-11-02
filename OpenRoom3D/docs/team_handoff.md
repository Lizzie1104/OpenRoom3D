# Continuing the project

The delivered pipeline supports open-vocabulary Gaussian scores, candidate inspection, geometry-derived pose, isolated rendering and rigid editing. The most useful next improvement is separating an object from its support surface without erasing its own detail.

1. Reproduce the saved viewer and compare the original geometry against the scale-regularized selection. See [pose controls](pose_demo.md).
2. Compare held-out images, the object-only render and the scene after removal. Preserve the original checkpoint and write each variation into a fresh output directory.
3. Start with the recorded mask agreement, off-mask weight and coverage. A smaller cluster or a cleaner point-center plot alone is insufficient evidence.
4. Evaluate a present object, a synonym and an absent object. An absent query can score above a text threshold when compared only with the generic negative prompt.
5. Record split, method, threshold, model identity, and whether masks were generated or manually annotated. Never reuse Gaussian indices after retraining.

The baseline scene is in `outputs/desk02/feature-splatting/`. The two meaningful geometry reruns are `control_sh0` and `scalereg_sh0` under the retraining results. Matching camera data live in `data/processed/desk02/`. The legacy runs without the SH0 suffix used different color representations; the project renderer assumes degree zero when decoding their DC color.

The project includes a local source packer, `python scripts/package_release.py`. It packages an explicit list of reviewed source files, checks digests, and excludes datasets and runtime dependencies. It does not upload anything. Keep the photo/model files alongside the project when the recipient needs a full local reproduction.
