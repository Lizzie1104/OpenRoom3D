# From text features to a movable object

1. Export one record per Gaussian with `export_query_gaussians.py`. By default the best positive prompt is compared with each negative in a two-way temperature-scaled softmax, and the weakest comparison is kept (LERF relevancy, `--score-mode pairwise-min`). The default negatives are `object, things, stuff, texture` plus hand-written desk02 categories from `scripts/query_negatives.py`. Negatives sharing a word with the query are dropped. Replace the scene categories for a new scene and report them as manual input. `--score-mode softmax --negative object` reproduces the Viewer score used by the earlier experiments. Synonyms can share the positive set. The NPZ contains indices, XYZ, RGB, opacity, scales, quaternions, raw scores and the threshold mask.
2. Use `refine_query_gaussians.py` to smooth scores over a 3D nearest-neighbor graph, optionally weighting color and limiting distance. Save raw and refined scores separately. DBSCAN then removes small connected components. Its radius uses reconstruction units.
3. Run `inspect_query_objects.py` to compare scene RGB, raw selection and colored clusters. Display sampling does not remove points from the exported candidate. Cluster labels can change when parameters change. A merged multi-cluster pose is unsuitable for manipulating one instance.
4. Use `estimate_object_pose.py` on a single reviewed component. Check the isolated RGB render and the scene after removal. Axes and an OBB describe the chosen points; they do not verify the object's identity.
5. For boundary refinement, seed a 2D tracker from a high-confidence 3D cluster. Accumulate rendered contribution inside and outside training masks, then apply multiview hull constraints. Reserve held-out frames for reporting.

## Ablations

The initial comparison used 16 neighbors, a DBSCAN core minimum of 8 and a retained-cluster minimum of 32. With a radius of 0.01, thresholds of 0.50, 0.60 and 0.70 traded contamination for coverage. A second variant used alpha 0.8, normalized-color bandwidth 0.15 and maximum neighbor distance 0.02. Alternative negatives included nearby object categories. Additional negative prompts change the softmax denominator, so compare operating points instead of treating the resulting probabilities as interchangeable. The pairwise minimum avoids that dilution; a later comparison is summarized in [the experiments](experiments.md).

The largest keyboard cluster originally included neighboring surfaces. A query for the complete open laptop matched the intended editing target more closely. Changing thresholds or cropping the display could conceal leakage without fixing ownership. The masked-render approach improves separation but can remove Gaussians that contribute to an otherwise plausible palm rest. Inspect the original selection, the edited object and the newly revealed background separately.

## Criteria for a new scene

Evaluate present, synonymous, fine-grained and absent queries. Save settings and counts, inspect multiple train and held-out viewpoints and report any manual input. Pose and manipulation can be tested independently of semantic quality. Metric-scale calibration, canonical 6DoF accuracy, CAD correspondence and robotics are outside the validated pipeline.
