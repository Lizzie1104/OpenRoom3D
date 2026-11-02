#!/usr/bin/env python3
"""Build a local source archive from the reviewed file list."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release")
    args = parser.parse_args()
    refs = json.loads((ROOT / "docs/source_files.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "OpenRoom3D-source.tar.gz"
    with tempfile.NamedTemporaryFile(dir=args.output, delete=False) as temp:
        tmp = Path(temp.name)
    try:
        with tmp.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=raw) as gz, tarfile.open(fileobj=gz, mode="w|") as tar:
            for name, expected in sorted(refs.items()):
                path = ROOT / name
                if Path(name).is_absolute() or ".." in Path(name).parts or path.resolve() != path or not path.is_file():
                    raise ValueError("Unapproved source path")
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Review this changed file before packaging: {name}")
                entry = tarfile.TarInfo("OpenRoom3D/" + name)
                entry.mode = 0o755 if path.suffix == ".sh" else 0o644
                entry.size = len(data)
                tar.addfile(entry, io.BytesIO(data))
            entry = tarfile.TarInfo("OpenRoom3D/docs/source_files.json")
            data = (json.dumps(refs, indent=2, sort_keys=True) + "\n").encode()
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(target)


if __name__ == "__main__":
    main()
