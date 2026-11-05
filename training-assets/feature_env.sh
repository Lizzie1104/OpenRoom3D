# Source after train_env.sh to run fresh Feature Splatting feature extraction offline.
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="$A/torch_home"
export OPENROOM_MOBILESAM_REPO="$A/hub/MobileSAMV2"
export OPENROOM_DINO_REPO="$A/hub/dinov2"
export OPENROOM_CLIP_WEIGHTS="$A/models/clip/ViT-L-14-336px.state_dict.pt"
