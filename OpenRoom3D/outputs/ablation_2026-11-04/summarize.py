"""Mean and sample standard deviation per configuration from eval/<config>_s<seed>.json.

Run from the project root. Writes summary.json and prints a Markdown table.
"""
import json
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

O = Path("outputs/ablation_2026-11-04")
METRICS = {
    "PSNR": lambda r: r["nerfstudio"]["psnr"],
    "LPIPS": lambda r: r["nerfstudio"]["lpips"],
    "Gaussians": lambda r: r["geometry"]["gaussians"],
    "aspect > 100": lambda r: r["geometry"]["aspect_gt100"],
    "SSIM": lambda r: r["nerfstudio"]["ssim"],
    "ownership IoU": lambda r: r["ownership_selection"]["iou"],
    "ownership coverage": lambda r: r["ownership_selection"]["coverage"],
    "ownership off-mask": lambda r: r["ownership_selection"]["outside"],
    "laptop AP": lambda r: r["queries"]["default"]["laptop"]["average_precision"],
    "synonym AP": lambda r: r["queries"]["default"]["notebook computer"]["average_precision"],
    "laptop >= 0.5": lambda r: r["queries"]["default"]["laptop"]["fraction_ge_05"],
    "text IoU": lambda r: r["queries"]["default"]["laptop"]["text_selection"]["chosen"]["iou"],
    "text capture": lambda r: r["queries"]["default"]["laptop"]["text_selection"]["chosen"]["captured"],
    "television >= 0.5": lambda r: r["queries"]["default"]["television"]["ge_05"],
    "legacy laptop AP": lambda r: r["queries"]["legacy"]["laptop"]["average_precision"],
}


def main():
    groups = defaultdict(list)
    for path in sorted((O / "eval").glob("*.json")):
        if re.fullmatch(r".+_s\d+", path.stem):
            groups[path.stem.rsplit("_s", 1)[0]].append(json.loads(path.read_text()))
    reference = O / "eval" / "original_september.json"
    if reference.exists():
        groups["original (Sept., 1 run)"] = [json.loads(reference.read_text())]
    summary = {name: {metric: {"mean": float(np.mean(values := [f(r) for r in runs])),
                               "std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n": len(values)}
                      for metric, f in METRICS.items()} for name, runs in groups.items()}
    (O / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("| Metric | " + " | ".join(f"{name} (n={len(groups[name])})" for name in summary) + " |")
    print("|---|" + "---:|" * len(summary))
    for metric in METRICS:
        cells = []
        for name in summary:
            cell = summary[name][metric]
            digits = 0 if metric in ("Gaussians", "television >= 0.5") else 3
            cells.append(f"{cell['mean']:.{digits}f}" + (f" ± {cell['std']:.{digits}f}" if cell["std"] is not None else ""))
        print(f"| {metric} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
