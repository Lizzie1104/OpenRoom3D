# Environments and reproduction

Separate the lightweight viewer, the video/COLMAP stage and CUDA training. Virtual environments, compiler caches, downloaded toolchains and machine diagnostics are local dependencies. They are excluded from the shareable source tree.

The viewer needs Python, NumPy, SciPy and a Viser build that implements Gaussian splats. The existing launchers accept `PYTHON=/path/to/python`. On a new system create a virtual environment outside the project. Check the saved viewer before building the training stack.

The historical baseline used Python 3.8, PyTorch 2.1.2 with CUDA 11.8, Nerfstudio 1.1.5 and gsplat 1.4.0. The text and Conda manifests record software versions only. A separate training configuration used Python 3.12, PyTorch 2.11 with CUDA 12.8, gsplat 1.4.0, Nerfstudio 1.1.5 and NumPy 1.26.4. Scientific analysis scripts using NumPy 2 can require a separate environment. Verify that the chosen runtime supports the intended GPU architecture.

Before a long run, check CUDA tensor arithmetic, finite gradients, and a small gsplat rasterization forward/backward pass. Install a matching compiler toolchain when the rasterizer needs to build locally. Registration of a Nerfstudio plugin alone does not demonstrate that feature extraction works. The historical 200-step geometry smoke run reduced the training loss from 0.2363 to 0.1335. The original smoke test did not verify complete feature training.

Use `OPENROOM_TRAIN_ENV` and `OPENROOM_VIDEO_ENV` to select local environments for the video pipeline. Run `bash scripts/video_to_scene.sh sample path/to/video.mp4 200` for a short geometry run, or set `PHOTOS=1` and supply a directory. Choose a new scene identifier. Review the registered-image fraction before running 30,000 steps.

Feature extraction can reuse the supplied cache. For fresh MobileSAM or DINO extraction, set `OPENROOM_MOBILESAM_REPO` or `OPENROOM_DINO_REPO` to a reviewed local checkout, and provision its weights beforehand. New tracker runs use the supplied SAM 2 weight file with the corresponding SAM 2.1 large configuration. The vendored plugin and model code retain their upstream license or attribution.

The registered model list pairs each bundled Nerfstudio configuration with checkpoint digests. Newly trained runs created by the local application are recorded after training. Verify an unfamiliar training config before loading it; Nerfstudio configs contain Python object tags. The code requests restricted checkpoint loading. Refusal to load an incompatible model should be resolved by reviewing its format and provenance.
