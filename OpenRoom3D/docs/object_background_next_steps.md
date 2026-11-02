# Object ownership and the hidden background

An object carrying fragments of the table and a hole appearing after it moves are different problems. The first concerns Gaussian ownership. The second concerns observations of the revealed region.

Improve ownership using multiview instance masks, inside/outside contribution ratios and held-out object-only renders. Measure leakage and missing detail together. Needles or spatially misplaced Gaussians can explain why a good full-scene render decomposes poorly.

The strongest background reference is a second capture with the object removed and the surrounding scene stationary. Register the two captures, estimate the support plane and keep the background layer separate from the object. If no such observations exist, the optional inferred layer extrapolates surrounding tabletop appearance. Its mask records which points were inferred, and it stays in the scene when the object moves. Shadows, cables, scratches and view-dependent reflections under an occluder cannot be recovered by planar extrapolation.

Validate by hiding observed regions, comparing interpolation against smoothing and mean-color baselines, and inspecting seams at several angles. Test regions comparable in size to the real occlusion whenever data permit.
