"""64- versus 128-wide features: mean and sample std per configuration. Run from the project root.

Reads ../ablation_2026-11-04/eval (64) and eval/ here (128); writes summary.json and prints a Markdown table.
"""
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, "outputs/ablation_2026-11-04")
from summarize import METRICS

ROOTS = {"64": Path("outputs/ablation_2026-11-04/eval"), "128": Path("outputs/feat128_2026-11-04/eval")}
CONFIGS = ("scalereg_w1e-3", "control_w1e-3")


def main():
    groups = {}
    for config in CONFIGS:
        for res, root in ROOTS.items():
            runs = [json.loads(p.read_text()) for p in sorted(root.glob(f"{config}_s*.json")) if re.fullmatch(r".+_s\d+", p.stem)]
            if runs:
                groups[f"{config} @{res}"] = runs
    summary = {name: {m: {"mean": float(np.mean(v := [f(r) for r in runs])),
                          "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                          "values": [float(x) for x in v]} for m, f in METRICS.items()}
               for name, runs in groups.items()}
    Path("outputs/feat128_2026-11-04/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("| Metric | " + " | ".join(f"{n} (n={len(groups[n])})" for n in summary) + " |")
    print("|---|" + "---:|" * len(summary))
    for m in METRICS:
        digits = 0 if m in ("Gaussians", "television >= 0.5") else 3
        cells = [f"{c['mean']:.{digits}f}" + (f" ± {c['std']:.{digits}f}" if c["std"] is not None else "")
                 for c in (summary[n][m] for n in summary)]
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
