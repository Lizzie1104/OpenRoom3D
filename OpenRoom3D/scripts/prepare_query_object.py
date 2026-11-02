#!/usr/bin/env python3
"""Prepare scene-scoped, unverified object candidates for the pose viewer.

Run in the openroom3d environment. This reuses the existing text export,
refinement, inspection and pose scripts; it never changes a checkpoint or
fills missing geometry. A spatial cluster is not a verified object instance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__:
    from .query_filename import query_filename
    from .query_negatives import DEFAULT_NEGATIVES, DEFAULT_SCORE_MODE, SCORE_MODES, without_query
else:
    from query_filename import query_filename
    from query_negatives import DEFAULT_NEGATIVES, DEFAULT_SCORE_MODE, SCORE_MODES, without_query


def nonempty_text(value):
    value = value.strip()
    if not value or any(ord(char) < 32 for char in value):
        raise argparse.ArgumentTypeError("text must be nonempty and contain no control characters")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="this scene's Feature Splatting config.yml")
    parser.add_argument("--query", type=nonempty_text, required=True, help="positive text query; comma-separated synonyms")
    parser.add_argument("--label", type=nonempty_text, help="viewer label; defaults to first query")
    parser.add_argument("--output", type=Path, required=True, help="new/empty directory; existing results are never overwritten")
    parser.add_argument("--negative", type=nonempty_text, default=DEFAULT_NEGATIVES,
                        help="comma-separated; the default includes desk02 categories, replace them for a new scene")
    parser.add_argument("--score-mode", choices=SCORE_MODES, default=DEFAULT_SCORE_MODE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--color-sigma", type=float)
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--eps", type=float, default=0.02, help="DBSCAN radius in reconstruction coordinates")
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--min-cluster", type=int, default=32)
    parser.add_argument("--cluster", type=int, help="explicit retained cluster ID; otherwise export all candidates")
    parser.add_argument("--max-points", type=int, default=20000, help="inspection display sampling only")
    parser.add_argument("--port", type=int, default=7011, help="port in generated viewer command")
    return parser


def validate_arguments(args):
    queries = [text.strip() for text in args.query.split(",") if text.strip()]
    if not queries or not any(text.strip() for text in args.negative.split(",")):
        raise ValueError("positive and negative queries must contain text")
    without_query(args.negative, args.query)
    label = args.label or queries[0]
    if "=" in label:
        raise ValueError("viewer label must not contain '='; pass a different --label")
    for name in ("threshold", "alpha"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("--{} must be finite and in [0,1]".format(name))
    for name in ("temperature", "sigma", "color_sigma", "max_distance", "eps"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("--{} must be finite and positive".format(name.replace("_", "-")))
    for name in ("chunk_size", "k", "min_samples", "min_cluster", "max_points"):
        if getattr(args, name) <= 0:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.cluster is not None and args.cluster < 0:
        raise ValueError("--cluster must be a nonnegative retained cluster ID")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1,65535]")
    config = args.config.resolve()
    if not config.is_file():
        raise ValueError("config does not exist: {}".format(config))
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory: {}".format(output))
    return config, output, label, query_filename(queries[0])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_script(script, arguments):
    # argv, not a shell string: query/label text cannot execute shell commands.
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script)] + [str(value) for value in arguments],
        cwd=str(PROJECT_ROOT), check=True,
    )


def viewer_command(scene, candidates, port):
    if not candidates:
        return None
    argv = [sys.executable, str(PROJECT_ROOT / "scripts" / "object_pose_viewer.py"),
            "--scene", str(scene), "--port", str(port)]
    for row in candidates:
        argv.extend(["--object", "{}={}".format(row["viewer_label"], row["npz"])])
    return shlex.join(argv)


def prepare(args):
    config, output, label, filename = validate_arguments(args)
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw"
    raw = raw_dir / (filename + ".npz")
    refined = output / "refined.npz"
    candidate_dir = output / "candidates"
    # Relative dataset/checkpoint paths inside Nerfstudio configs resolve at the
    # project root, even when this wrapper was launched from another directory.
    run_script("export_query_gaussians.py", [
        "--config", config, "--query", args.query, "--negative", args.negative,
        "--score-mode", args.score_mode, "--threshold", args.threshold, "--temperature", args.temperature,
        "--chunk-size", args.chunk_size, "--output", raw_dir,
    ])
    refine_args = [raw, "--output", refined, "--k", args.k, "--alpha", args.alpha,
                   "--threshold", args.threshold, "--eps", args.eps,
                   "--min-samples", args.min_samples, "--min-cluster", args.min_cluster]
    for name in ("sigma", "color_sigma", "max_distance"):
        if getattr(args, name) is not None:
            refine_args.extend(["--" + name.replace("_", "-"), getattr(args, name)])
    run_script("refine_query_gaussians.py", refine_args)
    run_script("inspect_query_objects.py", ["--raw", raw, "--refined", refined,
               "--output", candidate_dir, "--max-points", args.max_points])
    report = json.loads((candidate_dir / "candidates.json").read_text(encoding="utf-8"))
    rows = sorted(report["candidates"], key=lambda row: (-row["num_gaussians"], row["cluster_label"]))
    if args.cluster is not None:
        rows = [row for row in rows if row["cluster_label"] == args.cluster]
        if not rows:
            raise ValueError("requested cluster {} is not retained; inspect {}".format(
                args.cluster, candidate_dir / "inspection.html"))
    scene_hash = sha256_file(raw)
    candidates = []
    for rank, original in enumerate(rows, start=1):
        row = dict(original)
        npz = candidate_dir / row["npz"]
        pose = npz.with_suffix(".pose.json")
        if row["num_gaussians"] < 3:
            row["pose"] = None
            row["viewer_label"] = None
            row["status"] = "too_few_points_for_pose"
        else:
            run_script("estimate_object_pose.py", [npz, "--output", pose])
            row["pose"] = str(pose)
            row["viewer_label"] = "{} / cluster {}".format(label, row["cluster_label"])
            row["status"] = "candidate_pending_isolated_render_review"
        row.update(rank_by_size=rank, npz=str(npz), label=label,
                   scoped_id="{}:cluster:{}".format(scene_hash[:16], row["cluster_label"]),
                   source_scene_sha256=scene_hash)
        candidates.append(row)
    # Existing exporter records the loaded checkpoint and training step. Include
    # them verbatim; candidate Gaussian indices only refer to this raw export.
    metadata = {}
    for line in (raw_dir / (filename + ".txt")).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in ("checkpoint", "step"):
            metadata[key] = value
    launchable = [row for row in candidates if row["viewer_label"] is not None]
    command = viewer_command(raw, launchable, args.port)
    manifest = {
        "schema_version": 1, "query": args.query, "label": label,
        "status": "pending_visual_verification" if launchable else "no_pose_candidates",
        "config": str(config), "config_sha256": sha256_file(config),
        "scene_npz": str(raw), "scene_npz_sha256": scene_hash,
        "checkpoint": metadata.get("checkpoint"), "training_step": metadata.get("step"),
        "refined_npz": str(refined), "inspection_html": str(candidate_dir / "inspection.html"),
        "coordinate_system": "reconstruction coordinates; scale is not established in meters",
        "id_scope": "this exported scene/checkpoint only; do not reuse IDs after retraining",
        "ranking": "Gaussian count descending; not a semantic correctness ranking",
        "instance_verification": "not performed; clusters may be fragments or include background",
        "viewer_behavior": "one selected candidate moves at a time; selector compares candidates",
        "parameters": {name: getattr(args, name) for name in (
            "negative", "score_mode", "threshold", "temperature", "k", "alpha", "sigma", "color_sigma",
            "max_distance", "eps", "min_samples", "min_cluster", "cluster")},
        "candidates": candidates, "viewer_command": command,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if command:
        (output / "viewer_command.txt").write_text(command + "\n", encoding="utf-8")
        launcher = output / "start_viewer.sh"
        launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + command + "\n", encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | 0o111)
    print("Prepared {} unverified candidate(s): {}".format(len(candidates), path))
    print("Inspect: {}".format(manifest["inspection_html"]))
    if command:
        print("Viewer:\nbash {}".format(shlex.quote(str(launcher))))
    else:
        print("No usable candidate: inspect the query/parameters; no object or pose was invented.")
    return path


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, "preparation failed: {}\n".format(error))


if __name__ == "__main__":
    main()
