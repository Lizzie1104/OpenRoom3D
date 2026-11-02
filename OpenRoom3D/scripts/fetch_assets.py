#!/usr/bin/env python3
"""Download the files that are too large for git and verify their digests.

Checkpoints and the feature cache come from this repository's GitHub Release;
pretrained weights come from their official hosts. Paths and SHA-256 digests are
listed in docs/large_files.json. Existing files with the right digest are kept.
With --convert-clip, the CLIP safetensors file is also written as the state dict
read through OPENROOM_CLIP_WEIGHTS (needs torch and safetensors).

    python scripts/fetch_assets.py                 # everything
    python scripts/fetch_assets.py --only pretrained --convert-clip
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "large_files.json"
CLIP_STATE_DICT = ROOT.parent / "training-assets/models/clip/ViT-L-14-336px.state_dict.pt"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def origin_repository():
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"origin is not a GitHub repository: {url}; pass --repo owner/name")
    return match.group(1)


def fetch(url, row):
    target = (ROOT / row["path"]).resolve()
    if target.exists() and target.stat().st_size == row["bytes"] and digest(target) == row["sha256"]:
        print("ok      ", row["path"])
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print("download", row["path"], f"({row['bytes'] / 1e6:.0f} MB)", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while block := response.read(1 << 20):
            output.write(block)
    if partial.stat().st_size != row["bytes"] or digest(partial) != row["sha256"]:
        partial.unlink()
        raise SystemExit(f"digest mismatch for {row['path']}; nothing was replaced")
    partial.replace(target)


def convert_clip():
    import torch
    from safetensors.torch import load_file
    source = CLIP_STATE_DICT.with_name("open_clip_model.safetensors")
    torch.save(load_file(str(source)), CLIP_STATE_DICT)
    print("wrote   ", CLIP_STATE_DICT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("release", "pretrained"))
    parser.add_argument("--repo", help="owner/name; defaults to the git origin")
    parser.add_argument("--convert-clip", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.only != "pretrained":
        repo = args.repo or origin_repository()
        base = f"https://github.com/{repo}/releases/download/{manifest['release_tag']}/"
        for row in manifest["release"]:
            fetch(base + row["asset"], row)
    if args.only != "release":
        for row in manifest["pretrained"]:
            fetch(row["url"], row)
    if args.convert_clip:
        convert_clip()


if __name__ == "__main__":
    main()
