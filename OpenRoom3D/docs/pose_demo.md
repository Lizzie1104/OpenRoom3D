# Pose and manipulation

Use `bash scripts/start_retrain_demo.sh` for the scale-regularized scene and `bash scripts/start_inferred_table_demo.sh` for the original scene with the tracked selection. The first listens on 7013; the second defaults to 7012. Both accept `--port`. The compatibility launcher `start_pose_demo.sh` opens the available baseline on 7010.

Orbit with the left mouse button, pan with the right and zoom with the wheel. Select a candidate, isolate it, then display its axes and oriented bounding box. Drag an arrow or ring, enter a world-axis displacement or rotation, or use the step buttons. Reset reconstructs the original pose. Switching candidates resets the current edit.

A selection contains original Gaussian indices, centers, scales, quaternions, colors and opacities. Pose uses a centroid and a right-handed PCA frame. The box center can differ from the centroid. The 4-by-4 matrix maps object-local coordinates into reconstruction coordinates. PCA signs and nearly equal eigenvalues can make orientation ambiguous, especially for symmetric objects. Distances are in reconstruction units.

For a rigid edit, apply the same world rotation to the centers and left-compose the quaternions. This rotates covariance as R Σ Rᵀ. Scale, color and opacity stay fixed. Export writes a new NPZ and pose JSON; keep the source checkpoint as the comparison baseline. The inferred tabletop has its own switch and stays stationary when the object moves.
