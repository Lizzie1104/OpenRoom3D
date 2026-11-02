"""CUDA render of the same arrays the viewer shows, unmoved and after a world X move."""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "scripts")]
import cv2, numpy as np, torch
from openroom3d.gaussian_scene import load_cameras, load_gaussians, render
RUN = next((R / "outputs/retrain_2026-11-01/scalereg_sh0/splatfacto").glob("*"))
O = R / "outputs/retrain_2026-11-01"
g = load_gaussians(RUN)
cs, K, _, (fw, fh) = load_cameras(R / "data/processed/desk02", RUN)
W, H = 960, 540
Ks = K.copy(); Ks[:2] *= W / fw
L = np.load(O / "inferred_table/inferred_background.npz")
ids = torch.as_tensor(np.load(O / "selection_scalereg/obj_laptop.npz")["gaussian_index"], device="cuda")
K_ = ("means", "scales", "quats", "opacities", "features_dc")
N = len(g["means"])
remaining = torch.ones(N, dtype=torch.bool, device="cuda"); remaining[ids] = False
cleared = torch.zeros(N, dtype=torch.bool, device="cuda"); cleared[torch.as_tensor(L["scene_cleared_index"], device="cuda")] = True
logit = lambda v: torch.logit(torch.as_tensor(v, device="cuda").clamp(1e-4, 1 - 1e-4))
table = {"means": torch.as_tensor(L["xyz"], device="cuda"), "scales": torch.log(torch.as_tensor(L["scale"], device="cuda")),
         "quats": torch.as_tensor(L["quat"], device="cuda"), "opacities": logit(L["opacity"])[:, None],
         "features_dc": logit(L["rgb"].astype(np.float32) / 255)}
def compose(shift, use_layer):
    laptop = {k: g[k][ids].detach().clone() for k in K_}
    laptop["means"] = laptop["means"] + torch.as_tensor(shift, device="cuda", dtype=torch.float32)
    keep = (remaining & ~cleared) if use_layer else remaining
    parts = [{k: g[k][keep].detach() for k in K_}, laptop] + ([table] if use_layer else [])
    return {k: torch.cat([p[k] for p in parts]) for k in K_}
# Move inside the table plane, toward the direction with the most free table.
P = np.load(O / "inferred_table/plane.npz")
tex = cv2.imread(str(O / "inferred_table/texture_observed.png"))[:, :, ::-1].astype(int)
foot = cv2.imread(str(O / "inferred_table/footprint.png"), 0) > 127
free = (tex[..., 0] > tex[..., 2] + 25) & (tex[..., 0] > 140)
best, score = None, -1
for angle in np.arange(0, 2 * np.pi, np.pi / 8):
    direction = np.array([np.cos(angle), np.sin(angle)])
    du, dv = (direction * .22 / float(P["texel"])).astype(int)
    moved = np.roll(np.roll(foot, dv, 0), du, 1)
    if abs(du) >= tex.shape[1] or abs(dv) >= tex.shape[0]:
        continue
    value = (moved & free).sum() / max(moved.sum(), 1)
    if value > score:
        best, score = direction, value
print("free-table fraction at target", round(float(score), 3))
move = (best[0] * P["axis_u"] + best[1] * P["axis_v"]) * .22
lift = float(sys.argv[1]) if len(sys.argv) > 1 else 0.
move = move + P["normal"] * lift
print("move", np.round(move, 3), "lift", lift)
states = [("Unmoved: original scene", compose([0, 0, 0], False)), ("Moved: layer OFF", compose(move, False)), (f"Moved: table ON, lift {lift}", compose(move, True))]
rows = []
for i in (39,):
    rows.append(np.hstack([render(s, cs[i]["c2w"], Ks, W, H)["rgb"].clamp(0, 1).cpu().numpy() for _, s in states]))
im = (np.vstack(rows)[:, :, ::-1] * 255).astype(np.uint8)
bar = np.full((44, im.shape[1], 3), 255, np.uint8)
for c, (t, _) in enumerate(states):
    cv2.putText(bar, t, (c * W + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2, cv2.LINE_AA)
cv2.imwrite(str(O / f"inferred_table/viewer_state_lift{lift}.jpg"), np.vstack([bar, im]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("gpu", torch.cuda.get_device_name(0), "peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
