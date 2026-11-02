# Local application

`bash scripts/start_app.sh` launches the Flask application on loopback port 7000 using the training environment. You can override the interpreter with `PYTHON`. For immediate inspection of the saved result, use the independent viewer on port 7013 instead.

Select a registered scene and enter a short query. Threshold decides which scores survive; the cluster radius controls how spatially close points must be to connect. Negatives compete with the query. The prefilled list names desk02 objects, so replace those categories with ones visible in a new scene. Submit, inspect the candidate count and open the viewer. The heatmap shows the scores for the full scene. Disable it to move an individual candidate.

An optional local video upload starts frame extraction, camera estimation and joint Gaussian/feature training. Use a fresh scene name. Model training requires the appropriate local dependencies and encoder weights. The application serializes compute work. Cancel a long job before submitting another. Runtime outputs are local to the current run. An interrupted job may leave intermediate assets for inspection.

The main app and its auxiliary viewer use different loopback ports. The standalone saved viewer on 7013 does not depend on exposing either auxiliary port.
