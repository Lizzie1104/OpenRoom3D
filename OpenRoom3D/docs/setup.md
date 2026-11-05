# Environments and reproduction

Separate the lightweight viewer, the video/COLMAP stage and CUDA training. Virtual environments, compiler caches, downloaded toolchains and machine diagnostics are local dependencies. They are excluded from the shareable source tree.

The viewer needs Python, NumPy, SciPy and a Viser build that implements Gaussian splats. The existing launchers accept `PYTHON=/path/to/python`. On a new system create a virtual environment outside the project. Check the saved viewer before building the training stack.

The historical baseline used Python 3.8, PyTorch 2.1.2 with CUDA 11.8, Nerfstudio 1.1.5 and gsplat 1.4.0. The text and Conda manifests record software versions only. A separate training configuration used Python 3.12, PyTorch 2.11 with CUDA 12.8, gsplat 1.4.0, Nerfstudio 1.1.5 and NumPy 1.26.4. Scientific analysis scripts using NumPy 2 can require a separate environment. Verify that the chosen runtime supports the intended GPU architecture.

Before a long run, check CUDA tensor arithmetic, finite gradients, and a small gsplat rasterization forward/backward pass. Install a matching compiler toolchain when the rasterizer needs to build locally. Registration of a Nerfstudio plugin alone does not demonstrate that feature extraction works. The historical 200-step geometry smoke run reduced the training loss from 0.2363 to 0.1335. The original smoke test did not verify complete feature training.

Use `OPENROOM_TRAIN_ENV` and `OPENROOM_VIDEO_ENV` to select local environments for the video pipeline. Run `bash scripts/video_to_scene.sh sample path/to/video.mp4 200` for a short geometry run, or set `PHOTOS=1` and supply a directory. Choose a new scene identifier. Review the registered-image fraction before running 30,000 steps.

Feature extraction can reuse the supplied cache. Fresh extraction needs local MobileSAMV2 and DINOv2 code and weights:

```bash
git clone https://github.com/RogerQi/MobileSAMV2 ../training-assets/hub/MobileSAMV2   # reviewed at 1033d77
git clone https://github.com/facebookresearch/dinov2 ../training-assets/hub/dinov2    # reviewed at 7764ea0
python scripts/fetch_assets.py --only pretrained       # weights go under ../training-assets/torch_home/hub
~/.local/share/openroom3d/train/bin/python -m pip install torchpack   # MobileSAMV2 imports it at load time
source ../training-assets/feature_env.sh                # sets TORCH_HOME, OPENROOM_MOBILESAM_REPO, OPENROOM_DINO_REPO, OPENROOM_CLIP_WEIGHTS
python scripts/extract_features.py --data data/processed/desk02_feat128 --final-res 128 --obj-res 256
```

`scripts/extract_features.py --check 4` re-extracts four frames and compares them with the reference cache; on this machine the local models reproduce the supplied 64-wide cache exactly (cosine similarity 1.0000). Each resolution needs its own data directory, which contains the images (or a symlink to them), `transforms.json` and `sparse_pc.ply`. Train with the same `--data` string and with `--pipeline.datamanager.final-feat-res` / `--pipeline.datamanager.obj-feat-res` matching the cache. Otherwise the cache is rejected; without the repositories set, the run then stops instead of overwriting the cache. New tracker runs use the supplied SAM 2 weight file with the corresponding SAM 2.1 large configuration. The vendored plugin and model code retain their upstream license or attribution.

The registered model list pairs each bundled Nerfstudio configuration with checkpoint digests. Newly trained runs created by the local application are recorded after training. Verify an unfamiliar training config before loading it; Nerfstudio configs contain Python object tags. The code requests restricted checkpoint loading. Refusal to load an incompatible model should be resolved by reviewing its format and provenance.
