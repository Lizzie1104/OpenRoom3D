# Implementation status

Pose estimation, single-object manipulation, reset, export and optional local presentation layers are implemented. The project includes original and scale-regularized geometry, object selections and held-out comparisons. All transforms apply to Gaussian means and covariance orientation.

The semantic heatmap is continuous. Instance completeness, empty-space behavior under occlusion, cross-scene parameter stability, metric scale and canonical pose accuracy still require independent evaluation. A local cleanup crop or inferred surface changes presentation; report it separately from selection quality. Start with the saved example before retraining.
