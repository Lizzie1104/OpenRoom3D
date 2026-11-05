"""Side-by-side 'laptop' relevancy of the 2D training targets: 64-wide versus 128-wide cache, one frame.

Pairwise-min relevancy against the default negatives, computed on the cached SAMCLIP maps (no 3D model).
Run from the project root in the training environment with OPENROOM_CLIP_WEIGHTS set.
"""
from pathlib import Path, PosixPath
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

sys.path.insert(0, "scripts")
from query_negatives import DEFAULT_NEGATIVES, without_query
from feature_splatting.utils.clip_text_encoder import clip_text_encoder

FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 40
encoder = clip_text_encoder("ViT-L/14@336px", "cuda")
pos = encoder.get_text_token(["laptop"])
neg = encoder.get_text_token(without_query(DEFAULT_NEGATIVES, "laptop").split(","))
tiles = []
for data in ("data/processed/desk02", "data/processed/desk02_feat128"):
    with torch.serialization.safe_globals([PosixPath]):
        cache = torch.load(f"{data}/feature_splatting_samclip_features.pt", weights_only=True, mmap=True)
    name = cache["image_fnames"][FRAME]
    f = F.normalize(cache["feature_dict"]["samclip"][FRAME].float().cuda(), dim=0)  # C,H,W
    s_pos = torch.einsum("c,chw->hw", pos[0], f)
    s_neg = torch.einsum("nc,chw->nhw", neg, f)
    rel = ((s_pos[None] - s_neg) / .05).sigmoid().min(0).values
    rel[f.abs().sum(0) == 0] = 0
    photo = Image.open(name).convert("RGB").resize((640, 360))
    heat = cm.turbo(F.interpolate(rel[None, None], (360, 640), mode="nearest")[0, 0].cpu().numpy())[..., :3]
    tiles.append((0.45 * np.asarray(photo) / 255 + 0.55 * heat))
out = Path("outputs/feat128_2026-11-04/targets_64_vs_128.jpg")
Image.fromarray((np.concatenate(tiles, 1) * 255).astype(np.uint8)).save(out, quality=90)
print(out, name)
