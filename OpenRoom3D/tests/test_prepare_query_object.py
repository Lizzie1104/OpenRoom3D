"""CPU-only checks for safe, honest object preparation orchestration."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_query_object as preparation


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "config.yml"
        self.config.write_text("immutable test config\n", encoding="utf-8")
        self.output = self.root / "cup candidates"

    def args(self, *extra):
        return preparation.build_parser().parse_args([
            "--config", str(self.config), "--query", "cup", "--output", str(self.output),
        ] + list(extra))

    def fake_runner(self, rows, calls):
        def run(script, argv):
            argv = [str(value) for value in argv]
            calls.append((script, argv))
            if script == "export_query_gaussians.py":
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir()
                first = next(text.strip() for text in argv[argv.index("--query") + 1].split(",") if text.strip())
                filename = first.replace("/", "_")
                (output / (filename + ".npz")).write_bytes(b"scene-specific-export")
                (output / (filename + ".txt")).write_text(
                    "checkpoint=/scene-only/checkpoint.ckpt\nstep=30000\n", encoding="utf-8")
            elif script == "refine_query_gaussians.py":
                Path(argv[argv.index("--output") + 1]).write_bytes(b"refined-selection")
            elif script == "inspect_query_objects.py":
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir()
                (output / "candidates.json").write_text(json.dumps({"candidates": rows}), encoding="utf-8")
                (output / "inspection.html").write_text("inspection", encoding="utf-8")
                for row in rows:
                    (output / row["npz"]).write_bytes(b"unchanged candidate parameters")
            elif script == "estimate_object_pose.py":
                Path(argv[argv.index("--output") + 1]).write_text("{}", encoding="utf-8")
            else:
                self.fail("unexpected script {}".format(script))
        return run

    @staticmethod
    def rows():
        return [
            {"cluster_label": 8, "num_gaussians": 12, "npz": "candidate_cluster_008.npz"},
            {"cluster_label": 3, "num_gaussians": 34, "npz": "candidate_cluster_003.npz"},
        ]

    def test_all_candidates_ranked_and_unverified_with_scene_scoped_command(self):
        calls = []
        with patch.object(preparation, "run_script", self.fake_runner(self.rows(), calls)):
            path = preparation.prepare(self.args("--label", "Insulated cup candidate"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([row["cluster_label"] for row in manifest["candidates"]], [3, 8])
        self.assertEqual(manifest["label"], "Insulated cup candidate")
        self.assertEqual(manifest["status"], "pending_visual_verification")
        self.assertTrue(all(row["status"] == "candidate_pending_isolated_render_review"
                            for row in manifest["candidates"]))
        self.assertEqual(manifest["checkpoint"], "/scene-only/checkpoint.ckpt")
        self.assertEqual(manifest["training_step"], "30000")
        self.assertEqual(manifest["scene_npz_sha256"], preparation.sha256_file(Path(manifest["scene_npz"])))
        command = shlex.split(manifest["viewer_command"])
        self.assertEqual(command.count("--object"), 2)
        self.assertNotIn("--presentation", command)
        self.assertIn(manifest["scene_npz"], command)
        launcher_lines = (self.output / "start_viewer.sh").read_text(encoding="utf-8").splitlines()
        self.assertEqual(launcher_lines[:2], ["#!/usr/bin/env bash", "set -euo pipefail"])
        self.assertEqual(shlex.split(launcher_lines[2]), ["exec"] + command)
        self.assertNotIn("--presentation", shlex.split(launcher_lines[2]))
        self.assertEqual(len([call for call in calls if call[0] == "estimate_object_pose.py"]), 2)

    def test_explicit_cluster_does_not_select_largest_instead(self):
        calls = []
        with patch.object(preparation, "run_script", self.fake_runner(self.rows(), calls)):
            path = preparation.prepare(self.args("--cluster", "8"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([row["cluster_label"] for row in manifest["candidates"]], [8])
        self.assertEqual(len([call for call in calls if call[0] == "estimate_object_pose.py"]), 1)

    def test_empty_selection_generates_no_pose_or_viewer(self):
        calls = []
        with patch.object(preparation, "run_script", self.fake_runner([], calls)):
            path = preparation.prepare(self.args("--label", "Absent object"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["label"], "Absent object")
        self.assertEqual(manifest["status"], "no_pose_candidates")
        self.assertEqual(manifest["candidates"], [])
        self.assertIsNone(manifest["viewer_command"])
        self.assertFalse((self.output / "viewer_command.txt").exists())
        self.assertFalse((self.output / "start_viewer.sh").exists())
        self.assertFalse(any(call[0] == "estimate_object_pose.py" for call in calls))

    def test_missing_cluster_is_error(self):
        with patch.object(preparation, "run_script", self.fake_runner(self.rows(), [])):
            with self.assertRaisesRegex(ValueError, "not retained"):
                preparation.prepare(self.args("--cluster", "0"))

    def test_shell_text_is_quoted_and_subprocess_uses_argv_project_cwd(self):
        label = "cup; touch /tmp/should-not-run $(echo x) ' quoted"
        npz = self.root / "a space and ' quote.npz"
        scene = self.root / "raw scene.npz"
        command = preparation.viewer_command(scene, [{"viewer_label": label, "npz": str(npz)}], 7011)
        argv = shlex.split(command)
        self.assertEqual(argv[argv.index("--object") + 1], label + "=" + str(npz))
        with patch.object(subprocess, "run") as runner:
            preparation.run_script("export_query_gaussians.py", ["--query", label])
        positional, keywords = runner.call_args
        self.assertEqual(positional[0][0], sys.executable)
        self.assertEqual(positional[0][-1], label)
        self.assertEqual(keywords["cwd"], str(preparation.PROJECT_ROOT))
        self.assertTrue(keywords["check"])
        self.assertNotIn("shell", keywords)

    def test_generated_launcher_preserves_shell_literal_label(self):
        label = "cup; touch /tmp/should-not-run $(echo x) ' quoted"
        with patch.object(preparation, "run_script", self.fake_runner(self.rows(), [])):
            path = preparation.prepare(self.args("--label", label))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        launcher = (self.output / "start_viewer.sh").read_text(encoding="utf-8")
        command_line = launcher.splitlines()[2]
        argv = shlex.split(command_line)
        self.assertEqual(argv, ["exec"] + shlex.split(manifest["viewer_command"]))
        first_object = argv[argv.index("--object") + 1]
        self.assertEqual(first_object.split("=", 1)[0], label + " / cluster 3")
        self.assertNotIn("--presentation", argv)

    def test_export_receives_scene_negatives_and_score_mode(self):
        calls = []
        with patch.object(preparation, "run_script", self.fake_runner(self.rows(), calls)):
            path = preparation.prepare(self.args())
        argv = dict(calls)["export_query_gaussians.py"]
        self.assertEqual(argv[argv.index("--score-mode") + 1], "pairwise-min")
        self.assertEqual(argv[argv.index("--negative") + 1], preparation.DEFAULT_NEGATIVES)
        parameters = json.loads(path.read_text(encoding="utf-8"))["parameters"]
        self.assertEqual(parameters["score_mode"], "pairwise-min")

    def test_reject_negatives_that_all_match_the_query(self):
        with patch.object(preparation, "run_script") as runner:
            with self.assertRaises(ValueError):
                preparation.prepare(self.args("--negative", "cup, paper cup"))
            runner.assert_not_called()

    def test_existing_results_are_never_overwritten(self):
        self.output.mkdir()
        trusted = self.output / "obj_001_laptop.npz"
        trusted.write_bytes(b"trusted selection")
        with patch.object(preparation, "run_script") as runner:
            with self.assertRaisesRegex(ValueError, "new or empty"):
                preparation.prepare(self.args())
        runner.assert_not_called()
        self.assertEqual(trusted.read_bytes(), b"trusted selection")

    def test_reject_invalid_parameters_before_gpu_export(self):
        for extra in (("--threshold", "nan"), ("--eps", "0"), ("--k", "0"),
                      ("--alpha", "1.1"), ("--label", "cup=unsafe"), ("--query", ",,,")):
            with self.subTest(extra=extra), patch.object(preparation, "run_script") as runner:
                with self.assertRaises(ValueError):
                    preparation.prepare(self.args(*extra))
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
