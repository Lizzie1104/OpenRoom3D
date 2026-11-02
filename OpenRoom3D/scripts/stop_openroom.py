#!/usr/bin/env python3
"""Stop only this workspace's local application and object viewers."""
from pathlib import Path
import os
import signal
import time


def stop():
    root = Path(__file__).resolve().parents[1]
    scripts = {root / "scripts/openroom_app.py", root / "scripts/object_pose_viewer.py"}
    targets = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [value.decode(errors="replace") for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
            if len(argv) < 2 or not Path(argv[0]).name.startswith("python"):
                continue
            script = Path(argv[1])
            if not script.is_absolute():
                script = (entry / "cwd").resolve() / script
            if script.resolve() in scripts:
                targets.append((int(entry.name), argv))
        except (OSError, PermissionError):
            continue
    if not targets:
        print("OpenRoom3D and its viewers are already stopped.")
        return
    for pid, _ in targets:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 8
    remaining = targets
    while remaining and time.monotonic() < deadline:
        still_running = []
        for pid, original_argv in remaining:
            try:
                proc = Path("/proc") / str(pid)
                argv = [value.decode(errors="replace") for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
                if argv == original_argv:
                    still_running.append((pid, original_argv))
            except OSError:
                pass
        remaining = still_running
        if remaining:
            time.sleep(.2)
    if remaining:
        print("Some processes are still stopping. Try the stop command again in a moment.")
        raise SystemExit(1)
    print("OpenRoom3D and its viewers have stopped.")


if __name__ == "__main__":
    stop()
