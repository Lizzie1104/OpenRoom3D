"""Viser server/control integration; does not replace browser visual testing."""
from argparse import Namespace
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
import urllib.request
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from object_pose_viewer import PoseViewer
from openroom3d.geometry import pca_pose, gaussian_covariances


class PoseViewerIntegrationTest(unittest.TestCase):
    def test_table_rollback_keeps_cleanup_and_object_controls(self):
        rng = np.random.default_rng(36)
        raw = {"xyz": (rng.normal(size=(50, 3)) * [.15, .12, .03] + [0, 0, .05]).astype(np.float32),
               "quat": np.tile([1., 0, 0, 0], (50, 1)).astype(np.float32),
               "scale": np.tile([.006, .003, .001], (50, 1)).astype(np.float32),
               "rgb": np.full((50, 3), 80, dtype=np.uint8), "opacity": np.full(50, .8, dtype=np.float32),
               "gaussian_index": np.arange(50), "similarity": np.full(50, .8, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene.npz"
            np.savez(scene, **raw)
            options = []
            for label, count in (("Laptop", 20), ("Other", 25)):
                path = root / f"{label}.npz"
                data = {key: value[:count].copy() for key, value in raw.items()}
                np.savez(path, **data)
                pose = pca_pose(data["xyz"])
                pose["source"] = str(path.resolve())
                path.with_suffix(".pose.json").write_text(json.dumps(pose))
                options.append(f"{label}={path}")
            config = {"scene_source": str(scene.resolve()), "compatible_object_sources": [str((root / "Laptop.npz").resolve())],
                      "normal": [0, 0, 1], "offset": 0., "origin": [0, 0, 0], "tangent_axes": [[1, 0], [0, 1], [0, 0]],
                      "footprint_min": [-.3, -.3], "footprint_max": [.3, .3], "rgb": [210, 180, 150],
                      "cleanup_scope_margin": .2, "background_min_height": -.008,
                      "column_min_height": -.008, "column_max_height": .2, "patch_height": .0003}
            presentation = root / "presentation.json"
            presentation.write_text(json.dumps(config))
            with socket.socket() as allocation:
                allocation.bind(("127.0.0.1", 0))
                port = allocation.getsockname()[1]
            args = Namespace(scene=scene, object=options, port=port, host="127.0.0.1", export_dir=root / "exports", presentation=presentation)
            with patch("viser._scene_api.SceneApi.add_mesh_simple", side_effect=AssertionError("Table replacement must not be created")):
                viewer = PoseViewer(args)
            try:
                original = viewer.state.arrays()
                self.assertFalse(viewer.background.visible)
                self.assertTrue(viewer.cleaned_background.visible)
                self.assertFalse(hasattr(viewer, "schematic_table_node"))
                self.assertFalse(hasattr(viewer, "completion_node"))
                viewer.nudge(0, 1)
                exported = viewer.export()
                snapshot = json.loads(exported.with_suffix(".presentation.json").read_text())
                self.assertFalse(snapshot["synthetic_background_in_object_export"])
                self.assertFalse(snapshot["table_replacement_enabled"])
                self.assertFalse(snapshot["background_completion_enabled"])
                for key in ("rgb", "patch_vertices_world", "patch_faces", "schematic_table_margin"):
                    self.assertNotIn(key, snapshot)
                self.assertEqual(len(np.load(exported)["xyz"]), 20)
                self.assertEqual(snapshot["selected_object_source"], str((root / "Laptop.npz").resolve()))
                viewer.apply_visibility()
                self.assertTrue(viewer.cleaned_background.visible)
                self.assertFalse(viewer.background.visible)
                viewer.nudge(1, 1)
                viewer.clean_background.value = False
                viewer.apply_visibility()
                self.assertTrue(viewer.background.visible)
                self.assertFalse(viewer.cleaned_background.visible)
                viewer.isolate.value = True
                viewer.apply_visibility()
                for node in (viewer.background, viewer.cleaned_background):
                    self.assertFalse(node.visible)
                viewer.isolate.value = False
                viewer.reset()
                self.assertFalse(hasattr(viewer, "completion_node"))
                for key in original:
                    np.testing.assert_array_equal(viewer.state.arrays()[key], original[key])
                viewer.select("Other")
                self.assertFalse(viewer.presentation_active)
                self.assertFalse(viewer.clean_background.visible)
                for key in raw:
                    np.testing.assert_array_equal(viewer.raw[key], raw[key])
            finally:
                viewer.server.stop()
            config["scene_source"] = str((root / "wrong_scene.npz").resolve())
            presentation.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                PoseViewer(args)

    def test_inferred_layer_shows_after_move_and_stays_in_place(self):
        rng = np.random.default_rng(7)
        raw = {"xyz": (rng.normal(size=(60, 3)) * [.2, .2, .05]).astype(np.float32),
               "quat": np.tile([1., 0, 0, 0], (60, 1)).astype(np.float32),
               "scale": np.tile([.006, .003, .001], (60, 1)).astype(np.float32),
               "rgb": np.full((60, 3), 80, dtype=np.uint8), "opacity": np.full(60, .8, dtype=np.float32),
               "gaussian_index": np.arange(60), "similarity": np.full(60, .4, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "scene.npz", **raw)
            options = []
            for name, start, stop in (("Laptop", 0, 20), ("Bottle", 40, 60)):
                path = root / f"{name}.npz"
                data = {key: value[start:stop] for key, value in raw.items()}
                np.savez(path, **data)
                pose = pca_pose(data["xyz"])
                pose["source"] = str(path.resolve())
                path.with_suffix(".pose.json").write_text(json.dumps(pose))
                options.append(f"{name}={path}")
            layer = root / "inferred.npz"
            table = {"xyz": raw["xyz"][:8].copy(), "rgb": raw["rgb"][:8], "opacity": raw["opacity"][:8], "scale": raw["scale"][:8],
                     "quat": raw["quat"][:8], "inferred": np.ones(8, dtype=bool), "scene_cleared_index": np.arange(25)}
            np.savez(layer, **table)
            with socket.socket() as allocation:
                allocation.bind(("127.0.0.1", 0))
                port = allocation.getsockname()[1]
            args = Namespace(scene=root / "scene.npz", object=options, port=port, host="127.0.0.1",
                             export_dir=root / "exports", inferred_background=layer)
            viewer = PoseViewer(args)
            try:
                self.assertTrue(viewer.background.visible)
                self.assertFalse(viewer.inferred_table.visible)
                viewer.nudge(0, 1)
                self.assertTrue(viewer.inferred_table.visible)
                self.assertTrue(viewer.inferred_scene.visible)
                self.assertFalse(viewer.background.visible)
                self.assertIn("estimated from the surrounding surface", viewer.mode_status.content)
                np.testing.assert_array_equal(viewer.inferred_table.position, [0, 0, 0])
                exported = viewer.export()
                self.assertEqual(len(np.load(exported)["xyz"]), 20)
                note = json.loads(exported.with_suffix(".inferred_background.json").read_text())
                self.assertTrue(note["inferred_background_shown"])
                self.assertFalse(note["synthetic_background_in_object_export"])
                viewer.show_inferred.value = False
                viewer.apply_visibility()
                self.assertFalse(viewer.inferred_table.visible)
                self.assertTrue(viewer.background.visible)
                viewer.show_inferred.value = True
                viewer.isolate.value = True
                viewer.apply_visibility()
                self.assertFalse(viewer.inferred_table.visible)
                self.assertFalse(viewer.inferred_scene.visible)
                viewer.isolate.value = False
                viewer.heatmap.value = True
                viewer.apply_visibility()
                self.assertFalse(viewer.inferred_table.visible)
                viewer.heatmap.value = False
                viewer.reset()
                self.assertFalse(viewer.inferred_table.visible)
                self.assertTrue(viewer.background.visible)
                viewer.select("Bottle")
                self.assertFalse(viewer.inferred_active)
                self.assertFalse(viewer.show_inferred.visible)
                viewer.nudge(0, 1)
                self.assertTrue(viewer.background.visible)
            finally:
                viewer.server.stop()
            table["scene_cleared_index"] = np.array([0, 60])
            np.savez(layer, **table)
            with self.assertRaises(ValueError):
                PoseViewer(args)
            table["scene_cleared_index"] = np.arange(25)
            table["xyz"] = table["xyz"][:, :2]
            np.savez(layer, **table)
            with self.assertRaises(ValueError):
                PoseViewer(args)

    def test_controls_export_scene_covariance_and_reset(self):
        rng = np.random.default_rng(16)
        raw = {"xyz": (rng.normal(size=(90, 3)) * [.25, .15, .07]).astype(np.float32),
               "quat": Rotation.random(90, random_state=rng).as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
               "scale": np.tile([.02, .006, 1e-8], (90, 1)).astype(np.float32),
               "rgb": rng.integers(0, 255, (90, 3), dtype=np.uint8), "opacity": np.full(90, .8, dtype=np.float32),
               "gaussian_index": np.arange(90), "similarity": np.full(90, .7, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "scene.npz", **raw)
            options = []
            for name, count in (("Original", 25), ("Shape", 35)):
                path = root / f"{name}.npz"
                data = {key: value[:count] for key, value in raw.items()}
                np.savez(path, **data)
                pose = pca_pose(data["xyz"])
                pose["source"] = str(path.resolve())
                path.with_suffix(".pose.json").write_text(json.dumps(pose))
                options.append(f"{name}={path}")
            with socket.socket() as allocation:
                allocation.bind(("127.0.0.1", 0))
                port = allocation.getsockname()[1]
            viewer = PoseViewer(Namespace(scene=root / "scene.npz", object=options, port=port,
                                         host="127.0.0.1", export_dir=root / "exports"))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{viewer.server.get_port()}", timeout=10) as response:
                    self.assertEqual(response.status, 200)
                original = viewer.state.arrays()
                viewer.nudge(0, 1)
                viewer.angles[2].value = 25.
                viewer.from_numbers()
                expected = Rotation.from_euler("z", 25, degrees=True).as_matrix()
                np.testing.assert_allclose(viewer.state.rotation, expected, atol=1e-10)
                np.testing.assert_allclose(viewer.state.translation, [.03, 0, 0], atol=1e-10)
                node_rotation = Rotation.from_quat(viewer.object_frame.wxyz[[1, 2, 3, 0]]).as_matrix()
                local_xyz = (original["xyz"] - viewer.state.pivot) @ viewer.axes_original
                np.testing.assert_allclose(local_xyz @ node_rotation.T + viewer.object_frame.position,
                                           viewer.state.arrays()["xyz"], atol=1e-7)
                covariance = gaussian_covariances(original["scale"], original["quat"])
                local_covariance = viewer.axes_original.T @ covariance @ viewer.axes_original
                current = viewer.state.arrays()
                np.testing.assert_allclose(node_rotation @ local_covariance @ node_rotation.T,
                                           gaussian_covariances(current["scale"], current["quat"]), atol=1e-10)
                viewer.isolate.value = True
                viewer.show_axes.value = False
                viewer.apply_visibility()
                self.assertFalse(viewer.background.visible)
                self.assertFalse(viewer.control.visible)
                self.assertTrue(viewer.object_frame.visible)
                exported = viewer.export()
                np.testing.assert_array_equal(np.load(exported)["xyz"], current["xyz"])
                self.assertTrue(exported.with_suffix(".pose.json").is_file())
                viewer.reset()
                for key in original:
                    np.testing.assert_array_equal(viewer.state.arrays()[key], original[key])
                viewer.heatmap.value = True
                viewer.apply_visibility()
                self.assertTrue(viewer.heatmap_node.visible)
                self.assertFalse(viewer.object_frame.visible)
                self.assertFalse(viewer.background.visible)
                self.assertFalse(viewer.control.visible)
                viewer.heatmap.value = False
                viewer.apply_visibility()
                self.assertFalse(viewer.heatmap_node.visible)
                self.assertTrue(viewer.object_frame.visible)
                viewer.control.position = viewer.state.pivot + [.01, .02, .03]
                viewer.control.wxyz = Rotation.from_matrix(expected @ viewer.axes_original).as_quat()[[3, 0, 1, 2]]
                viewer.from_gizmo()
                time.sleep(.05)
                np.testing.assert_allclose(viewer.state.translation, [.01, .02, .03], atol=1e-10)
                np.testing.assert_allclose(viewer.state.rotation, expected, atol=1e-10)
                viewer.select("Shape")
                self.assertEqual(len(viewer.state.original["xyz"]), 35)
                np.testing.assert_array_equal(viewer.state.rotation, np.eye(3))
            finally:
                viewer.server.stop()
