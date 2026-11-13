# OpenRoom3D

OpenRoom3D turns a calibrated image sequence into a Gaussian scene, queries it with text, and lets the operator select, move and inspect a candidate object. The included tabletop sequence, camera estimates, original feature checkpoint, comparison runs, segmentation masks and pretrained model files are real project assets.

## Open the saved result

From this directory, run `bash scripts/start_retrain_demo.sh`. Open `http://localhost:7013` in the project browser and keep the terminal running. The retrained laptop selection and its optional inferred tabletop load directly from NPZ files. No training, model download or login is needed for this viewer. Set `PYTHON` to the viewer environment's Python executable to use a different installation.

Start with the laptop in place. Turn on isolation and inspect the palm rest and screen. Turn isolation off, translate the laptop upward and toggle the inferred table layer. Reset restores the source pose. Use Ctrl+C in the launching terminal to stop the viewer.

## Natural-language 3D Agent

Set `DEEPSEEK_API_KEY` in the launching terminal, open the viewer and expand **自然语言 3D Agent**. Enter a command such as “把笔记本向右移动一步，沿桌面旋转 90 度，最后保存场景。” and click **执行指令**.

The Agent uses `deepseek-flash` with `thinking: {"type": "disabled"}` on every request. It generates a structured plan; local geometry executes object selection, movement, rotation, relative placement, coarse collision checks, reset, display controls and whole-scene export. Manual controls work without an API key.

Objects must already be loaded with `--object LABEL=OBJECT_NPZ`; the bundled demo loads only the laptop. Switching selections preserves each object's edits. Directions use world axes and distances use reconstruction units, without metric calibration. Collision checks use Gaussian-center bounding boxes for loaded independent objects at their destination. Scene export writes new NPZ and JSON files under `outputs/object_edits/`, leaving source data intact.

See the [complete Agent guide](docs/language_agent.md) for Windows and Linux startup, multi-object setup, tools and limitations. The [repository README](../README.md) provides a Chinese quick start.

Run the related checks from this directory:

```bash
python -m pytest tests/test_language_agent.py tests/test_pose_viewer.py tests/test_object_geometry.py -q
```

## Find the implementation

| Directory | Contents |
|---|---|
| `openroom3d/` | Geometry, rigid transformations, rendering and the local application UI |
| `scripts/` | Capture, semantic scoring, mask tracking, selection, evaluation and viewer entry points |
| `feature-splatting/` | Vendored Nerfstudio plugin with its upstream license |
| `data/processed/` | Images, camera models, local reconstruction coordinates and feature cache |
| `outputs/` | Saved checkpoints, object selections, numeric measurements and comparison images |
| `../training-assets/models/` | Pretrained tracking and image/text encoder weights |
| `tests/` | Geometry, selection, presentation and application regression tests |

Run `python scripts/fetch_assets.py --convert-clip` after cloning to restore checkpoints, the feature cache and pretrained weights. [The current status and roadmap](docs/status_and_roadmap.md) summarizes results and next steps. Read [the handoff](docs/team_handoff.md) before changing the pipeline. [The experimental results](docs/experiments.md) explain the difference between agreement with a tracked mask and accuracy against human annotations. [The runtime guide](docs/setup.md) separates the viewer, reconstruction and training dependencies.
