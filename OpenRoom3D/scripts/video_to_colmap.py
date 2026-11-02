"""Extract frames from a video (or take a photo folder) and solve cameras with pycolmap.

Run with .venv-video. Writes <output>/images and <output>/sparse/0. COLMAP runs on the
CPU here because the pycolmap wheel has no CUDA support.
"""
from __future__ import annotations

import argparse
import json
from PIL import Image, ImageOps
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pycolmap


def extract_frames(video, images, count, long_side):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    formats = {".mp4": "mov", ".mov": "mov", ".mkv": "matroska", ".avi": "avi"}
    demux = formats.get(video.suffix.lower())
    if not demux:
        raise ValueError("Supported video types: MP4, MOV, MKV and AVI.")
    source = [ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-f", demux, "-i", str(video)]
    probe = subprocess.run(source + ["-map", "0:v:0", "-an", "-progress", "pipe:1", "-f", "null", "-"],
                           check=True, capture_output=True, text=True)
    stats = dict(line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line)
    frames, seconds = int(stats["frame"]), int(stats["out_time_us"]) / 1_000_000
    if frames < 1 or seconds <= 0:
        raise ValueError("Input must contain at least one frame and a positive duration.")
    rate = min(count / seconds, frames / seconds)
    scale = f"scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'"
    subprocess.run(source + ["-map_metadata", "-1", "-vf", f"fps={rate},{scale}",
                    str(images / "frame_%05d.png")], check=True)
    return dict(source_frames=frames, seconds=seconds, sample_fps=rate)


def copy_photos(folder, images, count):
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    step = max(1, len(photos) / count)
    chosen = [photos[int(i * step)] for i in range(min(count, len(photos)))]
    for index, photo in enumerate(chosen, 1):
        if photo.is_symlink() or not photo.is_file():
            raise ValueError("Expected a regular local image.")
        with Image.open(photo) as source:
            pixels = ImageOps.exif_transpose(source).convert("RGB")
            clean = Image.frombytes("RGB", pixels.size, pixels.tobytes())
            clean.save(images / f"frame_{index:05d}.png", format="PNG")
    return dict(source_frames=len(photos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--photos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--long-side", type=int, default=1600)
    args = parser.parse_args()
    if not 1 <= args.frames <= 5000 or not 16 <= args.long_side <= 8192:
        parser.error("Check the frame count (1..5000) and image size (16..8192).")
    if args.video and (args.video.is_symlink() or not args.video.is_file()):
        parser.error("Expected a regular video file.")
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; choose a new scene name")
    images = args.output / "images"
    images.mkdir(parents=True)
    info = extract_frames(args.video, images, args.frames, args.long_side) if args.video \
        else copy_photos(args.photos, images, args.frames)
    total = len(list(images.iterdir()))
    work = args.output / "colmap"
    work.mkdir()
    database = work / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    pycolmap.match_exhaustive(database)
    models = pycolmap.incremental_mapping(database, images, work / "sparse")
    if not models:
        raise SystemExit("COLMAP registered no model: the video likely has too little overlap or too much blur")
    best = max(models.values(), key=lambda model: model.num_reg_images())
    target = args.output / "sparse/0"
    target.mkdir(parents=True)
    best.write_binary(str(target))
    report = dict(info, extracted=total, registered=best.num_reg_images(), models=len(models),
                  points=best.num_points3D(), reprojection_error=best.compute_mean_reprojection_error())
    (args.output / "colmap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)
    if report["registered"] < .8 * total:
        print("WARNING: under 80% of frames registered; inspect the capture before trusting the scene")


if __name__ == "__main__":
    main()
