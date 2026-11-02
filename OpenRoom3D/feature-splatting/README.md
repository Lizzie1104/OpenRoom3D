# Feature Splatting for Nerfstudio

This vendored plugin distills image features onto 3D Gaussians. Its Nerfstudio implementation uses MobileSAMV2 and a lightweight selection interface. The upstream Apache license is included in this directory. Public upstream source: https://github.com/vuer-ai/feature-splatting

Install the reviewed local package in a compatible training environment with `python -m pip install --no-deps ./feature-splatting`. Provision the pinned dependencies and local encoder weights separately. The bundled scene already contains a feature cache. For fresh extraction, see the project runtime guide.

After entering positive text queries, select the similarity output to inspect response across viewpoints. The consistent latent PCA output provides a stable low-dimensional visualization. Pause training before using editing controls. Selection and editing during optimization can race with model updates.

Open issues include instance boundaries, stable editing during training, and support-plane estimation. See the project experiments for comparisons against multiview mask ownership.
