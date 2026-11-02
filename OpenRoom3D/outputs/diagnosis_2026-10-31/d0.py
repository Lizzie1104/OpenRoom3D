from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import torch
R=(str(PROJECT_ROOT) + "/")
for f in ["outputs/desk02/feature-splatting/2026-09-22_202843/nerfstudio_models/step-000029999.ckpt","data/processed/desk02/feature_splatting_samclip_features.pt"]:
    print(f, torch.serialization.get_unsafe_globals_in_checkpoint(R+f))
