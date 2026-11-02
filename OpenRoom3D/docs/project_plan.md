# Project plan

The pipeline combines scene reconstruction, open-vocabulary features, spatial refinement, object extraction, local pose estimation and rigid manipulation in a common reconstruction coordinate system.

Start by recovering camera geometry and a Gaussian representation. Confirm registration, visually inspect the scene and measure RGB behavior in reserved views. Next export text similarity, retain raw scores and compare thresholds against geometry- and appearance-weighted neighborhood smoothing. Cluster retained points, inspect candidates separately and report failures for absent objects.

For a selected instance, estimate the translation from the center and local axes from PCA. Fit an oriented box; account for axis signs, near-equal eigenvalues and symmetries. A local 4-by-4 transform provides a practical edit frame. Choosing a canonical orientation, recovering metric scale and evaluating standardized 6DoF accuracy require extra supervision. CAD models, depth priors, FoundationPose and related estimators are potential extensions.

Prototype edits offline before connecting controls. Transform Gaussian centers and orientations together and test reset against the original state. Use object-only and removal renders to distinguish contamination from holes in the hidden background. Add multiview region constraints where contacts defeat purely spatial clustering. The principal comparison is raw semantic threshold versus geometric refinement versus measured render ownership.

The next experiments should isolate one variable per run: a competitive text vocabulary; finer feature maps; depth or normal regularization; motion masks; and a separately observed background. Preserve the numerical baseline, split and selection metrics. Only compare methods evaluated with matching rendering conventions.
