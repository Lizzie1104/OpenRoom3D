# Geometry ablation

Use `control_sh0` and `scalereg_sh0` for the matched comparison. The older `control` and `scalereg` trials used SH degree three, which the degree-zero RGB decoder does not reproduce correctly. They are retained for reference. The scale-regularized run supplied the default 7013 viewer. The paired `selection_*` arrays refer to their own checkpoint; transferring indices between these runs is invalid.

The regularized run scored 0.830 mask IoU and 0.082 off-mask weight, with coverage of 0.879. Language scores in these geometry reruns are nearest-neighbor transfers. Runtime logs were excluded. The saved measurements and images support the comparison without rerunning training. See [the full table](../../docs/experiments.md).
