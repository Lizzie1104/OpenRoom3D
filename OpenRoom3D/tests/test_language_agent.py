"""Language plans drive real viewer geometry; the network alone is stubbed."""
from argparse import Namespace
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from object_pose_viewer import PoseViewer
from openroom3d.language_agent import AgentError, execute_actions, plan_instruction, scene_context


class LanguageAgentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.temp.name)
        # Two disjoint cubes, with a duplicate mask representing a second candidate.
        cube = np.array([[x, y, z] for x in (-.05, .05) for y in (-.05, .05) for z in (0, .1)], dtype=np.float32)
        xyz = np.concatenate([cube, cube + [.6, 0, 0]]).astype(np.float32)
        self.raw = dict(xyz=xyz, quat=np.tile([1., 0, 0, 0], (16, 1)).astype(np.float32),
                        scale=np.full((16, 3), .005, dtype=np.float32), rgb=np.full((16, 3), 80, dtype=np.uint8),
                        opacity=np.full(16, .8, dtype=np.float32), gaussian_index=np.arange(16))
        np.savez(root / "scene.npz", **self.raw)
        options = []
        for label, ids in (("Laptop", np.arange(8)), ("Cup", np.arange(8, 16)), ("Laptop alternative", np.arange(8))):
            path = root / f"{label}.npz"
            np.savez(path, **{key: value[ids] for key, value in self.raw.items()})
            options.append(f"{label}={path}")
        np.savez(root / "plane.npz", normal=np.array([0., 0, 1]), center=np.zeros(3))
        with socket.socket() as allocation:
            allocation.bind(("127.0.0.1", 0))
            port = allocation.getsockname()[1]
        self.viewer = PoseViewer(Namespace(scene=root / "scene.npz", object=options, port=port,
                                          host="127.0.0.1", export_dir=root / "exports", plane=root / "plane.npz"))

    def tearDown(self):
        self.viewer.server.stop()
        self.temp.cleanup()

    def test_multi_object_plan_display_geometry_and_save(self):
        viewer = self.viewer
        results = execute_actions(viewer, [
            {"tool": "find_object", "object": "Laptop"},
            {"tool": "move_object", "object": "Laptop", "offset": [.1, 0, .2]},
            {"tool": "rotate_object", "object": "Laptop", "axis": "table", "degrees": 90},
            {"tool": "place_near", "object": "Cup", "target": "Laptop", "gap": .03},
            {"tool": "check_collision", "object": "Cup"}, {"tool": "save_scene"}])
        laptop, cup = viewer.object_states["Laptop"], viewer.object_states["Cup"]
        np.testing.assert_allclose(laptop.translation, [.1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(laptop.rotation @ [1., 0, 0], [0, 1, 0], atol=1e-10)
        self.assertLess(cup.translation[0], 0)
        # Selecting Cup must show the edited laptop in the background splats.
        np.testing.assert_allclose(viewer.background.centers[:8], laptop.arrays()["xyz"], atol=1e-7)
        output = next((self.root / "exports").glob("scene_*.npz"))
        saved = dict(np.load(output))
        np.testing.assert_allclose(saved["xyz"][:8], laptop.arrays()["xyz"])
        np.testing.assert_allclose(saved["xyz"][8:], cup.arrays()["xyz"])
        np.testing.assert_allclose(saved["quat"][:8], laptop.arrays()["quat"])
        for key in ("scale", "rgb", "opacity"):
            np.testing.assert_array_equal(saved[key], self.raw[key])
        for key, value in self.raw.items():
            np.testing.assert_array_equal(np.load(self.root / "scene.npz")[key], value)
        self.assertIn("无", results[-2])
        self.assertEqual(set(json.loads(output.with_suffix(".json").read_text())["objects"]), {"Laptop", "Cup"})
        viewer.selector.value = "Laptop"
        viewer.select("Laptop")
        np.testing.assert_allclose(viewer.state.translation, [.1, 0, 0], atol=1e-10)
        execute_actions(viewer, [{"tool": "reset_object", "object": "Laptop"}])
        np.testing.assert_array_equal(viewer.state.arrays()["xyz"], self.raw["xyz"][:8])

    def test_invalid_plan_and_collision_leave_scene_unchanged(self):
        for bad in (
            {"tool": "move_object", "object": "Cup", "offset": [-.6, 0, 0]},
            {"tool": "find_object", "object": "Unknown"},
            {"tool": "move_object", "object": "Laptop", "offset": [float("nan"), 0, 0]},
            {"tool": "shell", "command": "anything"},
            {"tool": ["move_object"]},
        ):
            with self.subTest(bad=bad):
                before = scene_context(self.viewer)
                with self.assertRaises(AgentError):
                    execute_actions(self.viewer, [{"tool": "move_object", "object": "Laptop", "offset": [0, .01, 0]}, bad,
                                                  {"tool": "save_scene"}])
                self.assertEqual(scene_context(self.viewer), before)
                self.assertFalse((self.root / "exports").exists())

    def test_overlapping_candidate_edits_are_rejected(self):
        with self.assertRaisesRegex(AgentError, "重叠"):
            execute_actions(self.viewer, [{"tool": "move_object", "object": "Laptop", "offset": [0, .1, 0]},
                                          {"tool": "move_object", "object": "Laptop alternative", "offset": [0, -.1, 0]}])
        np.testing.assert_array_equal(self.viewer.state.translation, np.zeros(3))

    def test_turn_off_table_allows_lift_then_reenable_projects_it(self):
        execute_actions(self.viewer, [{"tool": "set_display", "setting": "on_table", "value": False},
                                      {"tool": "move_object", "object": "Laptop", "offset": [0, 0, .2]}])
        self.assertAlmostEqual(self.viewer.state.translation[2], .2)
        execute_actions(self.viewer, [{"tool": "set_display", "setting": "on_table", "value": True}])
        self.assertAlmostEqual(self.viewer.state.translation[2], 0.)


class DeepSeekRequestTest(unittest.TestCase):
    def test_flash_non_thinking_request_and_clarification(self):
        from io import BytesIO
        plan = {"message": "有两个笔记本候选，请指定。", "actions": []}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]}
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
             patch("openroom3d.language_agent.request.urlopen", return_value=BytesIO(json.dumps(response).encode())) as send:
            self.assertEqual(plan_instruction("移动电脑", {"objects": []}), plan)
        req = send.call_args.args[0]
        sent = json.loads(req.data)
        self.assertEqual(sent["model"], "deepseek-flash")
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", sent)
        self.assertEqual(req.full_url, "https://api.deepseek.com/chat/completions")

    def test_missing_key_http_failure_and_malformed_response(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaisesRegex(AgentError, "DEEPSEEK_API_KEY"):
                plan_instruction("移动", {})
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
             patch("openroom3d.language_agent.request.urlopen", side_effect=HTTPError("url", 401, "error", {}, None)):
            with self.assertRaisesRegex(AgentError, "401"):
                plan_instruction("移动", {})
        from io import BytesIO
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
             patch("openroom3d.language_agent.request.urlopen", return_value=BytesIO(b'{"choices": []}')):
            with self.assertRaises(AgentError):
                plan_instruction("移动", {})


if __name__ == "__main__":
    unittest.main()
