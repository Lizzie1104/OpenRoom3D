#!/usr/bin/env bash
set -euo pipefail
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_project_root"
exec "${PYTHON:-$HOME/.local/share/openroom3d/train/bin/python}" scripts/openroom_app.py
