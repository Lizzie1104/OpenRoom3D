"""Load a splatfacto checkpoint and compute per-Gaussian visible render weights.

This is a torch-only fragment compositor, so ownership works where gsplat cannot
be built. Results are close to gsplat but not identical; do not report its PSNR
as an official Nerfstudio metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _safe_globals():
    import pathlib
    from contextlib import nullcontext
    if not hasattr(torch.serialization, "safe_globals"):
        return nullcontext()
    scalar = np.core.multiarray.scalar
    scalars = [(scalar, "numpy.core.multiarray.scalar"), scalar] if int(np.__version__.split(".")[0]) >= 2 else [scalar]
    return torch.serialization.safe_globals([
        *scalars, np.dtype,
        type(np.dtype("float64")), type(np.dtype("int64")), pathlib.PosixPath])


def load_gaussians(run_dir, device="cuda"):
    checkpoints = sorted(Path(run_dir, "nerfstudio_models").glob("step-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    with _safe_globals():
        state = torch.load(checkpoints[-1], map_location=device, weights_only=True)["pipeline"]
    prefix = "_model.gauss_params."
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def load_cameras(data_dir, run_dir, train_fraction=.9):
    """Return Nerfstudio-world cameras sorted by filename.

    The saved dataparser transform is T_orient @ applied_transform, while poses in
    transforms.json already include applied_transform, so it is removed here.
    """
    meta = json.loads(Path(data_dir, "transforms.json").read_text())
    parser = json.loads(Path(run_dir, "dataparser_transforms.json").read_text())
    frames = sorted(meta["frames"], key=lambda frame: frame["file_path"])
    count = len(frames)
    train = set(np.linspace(0, count - 1, int(np.ceil(count * train_fraction)), dtype=int).tolist())
    bottom = [[0, 0, 0, 1.]]
    transform = np.vstack([parser["transform"], bottom]) @ np.linalg.inv(np.vstack([meta["applied_transform"], bottom]))
    cameras = []
    for index, frame in enumerate(frames):
        c2w = transform @ np.array(frame["transform_matrix"])
        c2w[:3, 3] *= parser["scale"]
        cameras.append({"name": frame["file_path"], "c2w": c2w, "split": "train" if index in train else "eval"})
    intrinsics = np.array([[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1.]])
    distortion = np.array([meta["k1"], meta["k2"], meta["p1"], meta["p2"]])
    return cameras, intrinsics, distortion, (meta["w"], meta["h"])


def quat_to_rotation(quats):
    quats = quats / quats.norm(dim=-1, keepdim=True)
    w, x, y, z = quats.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


@torch.no_grad()
def render(gaussians, c2w, intrinsics, width, height, active=None, background=1., sigma=3., chunk=40_000_000):
    """Composite Gaussians front to back.

    Returns rgb [H,W,3], alpha [H,W], and fragments (pixel, gaussian, weight), with
    gaussian indices referring to the full checkpoint order.
    """
    device = gaussians["means"].device
    c2w = torch.as_tensor(c2w, device=device, dtype=torch.float32).clone()
    c2w[:3, 1:3] *= -1  # OpenGL camera to OpenCV camera
    rotation = c2w[:3, :3].T
    translation = -rotation @ c2w[:3, 3]
    K = torch.as_tensor(intrinsics, device=device, dtype=torch.float32)
    source = torch.arange(len(gaussians["means"]), device=device) if active is None else active.nonzero()[:, 0]
    camera_xyz = gaussians["means"][source] @ rotation.T + translation
    front = camera_xyz[:, 2] > .01
    source, camera_xyz = source[front], camera_xyz[front]
    z = camera_xyz[:, 2]
    u = camera_xyz[:, 0] / z * K[0, 0] + K[0, 2]
    v = camera_xyz[:, 1] / z * K[1, 1] + K[1, 2]
    axes = (rotation @ quat_to_rotation(gaussians["quats"][source])) * torch.exp(gaussians["scales"][source])[:, None, :]
    covariance = axes @ axes.transpose(1, 2)
    jacobian = torch.zeros(len(z), 2, 3, device=device)
    jacobian[:, 0, 0] = K[0, 0] / z
    jacobian[:, 0, 2] = -K[0, 0] * camera_xyz[:, 0] / z ** 2
    jacobian[:, 1, 1] = K[1, 1] / z
    jacobian[:, 1, 2] = -K[1, 1] * camera_xyz[:, 1] / z ** 2
    screen = jacobian @ covariance @ jacobian.transpose(1, 2)
    a, b, d = screen[:, 0, 0] + .09, screen[:, 0, 1], screen[:, 1, 1] + .09
    determinant = (a * d - b * b).clamp_min(1e-9)
    inv_a, inv_b, inv_d = d / determinant, -b / determinant, a / determinant
    radius_u, radius_v = (sigma * a.sqrt()).clamp(max=200), (sigma * d.sqrt()).clamp(max=200)
    visible = (u + radius_u > 0) & (u - radius_u < width) & (v + radius_v > 0) & (v - radius_v < height)
    indices = visible.nonzero()[:, 0]
    opacity = torch.sigmoid(gaussians["opacities"][source][:, 0])
    x0 = (u - radius_u).floor().clamp(0, width - 1).long()
    x1 = (u + radius_u).ceil().clamp(0, width - 1).long()
    y0 = (v - radius_v).floor().clamp(0, height - 1).long()
    y1 = (v + radius_v).ceil().clamp(0, height - 1).long()
    box_w, box_h = x1 - x0 + 1, y1 - y0 + 1
    counts = (box_w * box_h)[indices]
    cumulative = counts.cumsum(0)
    pixels, alphas, depths, owners = [], [], [], []
    position = 0
    while position < len(indices):
        base = cumulative[position - 1] if position else 0
        end = min(max(int(torch.searchsorted(cumulative, base + chunk).item()), position + 1), len(indices))
        batch, batch_counts = indices[position:end], counts[position:end]
        owner = torch.repeat_interleave(batch, batch_counts)
        starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), batch_counts.cumsum(0)[:-1]])
        offset = torch.arange(int(batch_counts.sum()), device=device) - torch.repeat_interleave(starts, batch_counts)
        px = x0[owner] + offset % box_w[owner]
        py = y0[owner] + offset.div(box_w[owner], rounding_mode="floor")
        du, dv = px + .5 - u[owner], py + .5 - v[owner]
        distance = inv_a[owner] * du * du + 2 * inv_b[owner] * du * dv + inv_d[owner] * dv * dv
        alpha = opacity[owner] * torch.exp(-.5 * distance)
        keep = (distance < sigma * sigma) & (alpha > 1e-3)
        pixels.append((py * width + px)[keep])
        alphas.append(alpha[keep].float())
        depths.append(z[owner][keep])
        owners.append(owner[keep])
        position = end
    pixel, alpha, depth, owner = map(torch.cat, (pixels, alphas, depths, owners))
    order = torch.argsort(depth)
    order = order[torch.argsort(pixel[order], stable=True)]
    pixel, alpha, owner = pixel[order], alpha[order], owner[order]
    first = torch.cat([torch.ones(1, device=device, dtype=torch.bool), pixel[1:] != pixel[:-1]])
    log_transmit = torch.log1p(-alpha.clamp(max=.99999)).double()
    before = log_transmit.cumsum(0) - log_transmit
    group = first.long().cumsum(0) - 1
    weight = torch.exp(before - before[first][group]).float() * alpha
    colors = torch.sigmoid(gaussians["features_dc"][source])[owner]
    total = width * height
    accumulated = torch.zeros(total, device=device).index_add_(0, pixel, weight)
    rgb = torch.zeros(total, 3, device=device).index_add_(0, pixel, weight[:, None] * colors)
    rgb += (1 - accumulated[:, None]) * background
    return {"rgb": rgb.reshape(height, width, 3), "alpha": accumulated.reshape(height, width),
            "pixel": pixel, "gaussian": source[owner], "weight": weight}
