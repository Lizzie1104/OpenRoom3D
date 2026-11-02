"""Verify the local training configuration before invoking Nerfstudio's object loader."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "scene_hashes.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1048576), b""):
        h.update(block)
    return h.hexdigest()


def files_for(config):
    config = Path(config).resolve(strict=True)
    if config.name != "config.yml" or ROOT / "outputs" not in config.parents:
        raise ValueError("Select a saved run under the project outputs directory.")
    models = sorted((config.parent / "nerfstudio_models").glob("*.ckpt"))
    if not models or any(p.is_symlink() or p.resolve().parent != config.parent / "nerfstudio_models" for p in models):
        raise ValueError("Expected regular checkpoints in the selected run.")
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in [config, *models]}


def register_training(config, job_directory):
    # Called only by the local training job after its own process exits successfully.
    record = files_for(config)
    stamp = job_directory / "accepted-model.json"
    with stamp.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)


def eval_setup(config, **kwargs):
    config = Path(config).resolve()
    actual = files_for(config)
    approved = json.loads(BASELINE.read_text(encoding="utf-8"))
    # Newly generated records live in a trusted app registry, outside imported model directories.
    registry = ROOT / "outputs" / "application"
    for stamp in registry.glob("[0-9a-f]" * 32 + "/accepted-model.json"):
        if not stamp.is_symlink():
            approved.update(json.loads(stamp.read_text(encoding="utf-8")))
    if any(approved.get(k) != v for k, v in actual.items()):
        raise ValueError("These configuration or checkpoint bytes have not been approved.")
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    import numpy as np
    import torch
    from pathlib import PosixPath
    from nerfstudio.utils.eval_utils import eval_setup as load_scene
    from contextlib import nullcontext
    scalar = np.core.multiarray.scalar
    allowed = [scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("int64")), PosixPath]
    if np.__version__.split(".")[0] != "1":
        allowed.append((scalar, "numpy.core.multiarray.scalar"))
    guard = torch.serialization.safe_globals(allowed) if hasattr(torch.serialization, "safe_globals") else nullcontext()
    with guard:
        return load_scene(config, **kwargs)
