# Multiview laptop selection

The high-confidence seed projects into frame 39. The `tracking` directory holds propagated masks. `selection` holds Gaussian ownership, the object-only selection and held-out comparisons. The tracked masks supply a proxy reference; manual segmentation accuracy was not measured. The mask and hull thresholds are 0.85 and 0.95. The exported baseline selection has 13,200 Gaussians and held-out mask IoU 0.811.

Open `selection/compare_page.html` locally to compare the photographed scene and object removal. The `inferred_table` arrays keep estimated surface points separate. Toggle them after translating the object. The original checkpoint remains available. Detailed tradeoffs and the full parameter comparisons appear in [the experiment report](../../../docs/experiments.md).
