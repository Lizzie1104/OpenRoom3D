#!/usr/bin/env python3
"""Live Gaussian object pose viewer using the installed Viser splat renderer.

Uses existing original Gaussian means/covariances/RGB/opacity. Optional local
background display cleanup is reversible. An optional inferred table layer can
show after the object moves; it is an estimate, stored apart, and stays in place.
Parent-frame rigid transforms rotate both centers and covariance;
saved exports left-compose quaternions with the same world rotation.

Viser's installed experimental browser splat renderer is an interactive
preview. CUDA gsplat diagnostic renders remain the reference for appearance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openroom3d.geometry import gaussian_covariances, pca_pose, box_edges
from openroom3d.manipulation import ObjectState
from openroom3d.presentation import cleanup_mask, load_inferred_background, object_has_moved
from transform_object_gaussians import save_state


class PoseViewer:
    def __init__(self, args):
        if args.host not in ("127.0.0.1", "localhost"):
            raise ValueError("The viewer accepts loopback connections only.")
        self.args = args
        self.raw = dict(np.load(args.scene, allow_pickle=False))
        presentation_path = getattr(args, "presentation", None)
        self.presentation = json.loads(presentation_path.read_text()) if presentation_path else None
        if self.presentation and Path(self.presentation["scene_source"]).resolve() != args.scene.resolve():
            raise ValueError("Presentation source scene must match viewer scene")
        self.background_clean_mask = cleanup_mask(self.raw["xyz"], self.presentation) if self.presentation else None
        inferred_path = getattr(args, "inferred_background", None)
        self.inferred = load_inferred_background(inferred_path, len(self.raw["xyz"])) if inferred_path else None
        self.objects = {}
        for option in args.object:
            label, path = option.split("=", 1)
            if label in self.objects:
                raise ValueError("Object option labels must be unique")
            path = Path(path)
            data = dict(np.load(path, allow_pickle=False))
            ids = data["gaussian_index"]
            if not np.issubdtype(ids.dtype, np.integer) or ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(self.raw["xyz"]):
                raise ValueError(f"Invalid object indices: {path}")
            for key in ("xyz", "quat", "scale", "rgb", "opacity"):
                if not np.array_equal(data[key], self.raw[key][ids]):
                    raise ValueError(f"{path}: {key} differs from original scene; this viewer accepts unmodified selections")
            pose_path = path.with_suffix(".pose.json")
            if pose_path.is_file():
                pose = json.loads(pose_path.read_text())
                if Path(pose["source"]).resolve() != path.resolve():
                    raise ValueError("Object pose source mismatch")
            else:
                support = data.get("pose_support_mask", np.ones(len(ids), dtype=bool))
                pose = pca_pose(data["xyz"], data["xyz"][support])
            self.objects[label] = (path, data, pose)
        self.covariances = gaussian_covariances(self.raw["scale"], self.raw["quat"])
        # Viser serializes covariance through a Cholesky factor and float16.
        # Some trained Gaussians are almost flat. Stabilize only this preview's
        # encoding; exported source/rigid parameters are never regularized.
        self.preview_covariances = self.covariances + np.eye(3)[None] * (
            np.maximum(np.trace(self.covariances, axis1=1, axis2=2), 1.) * 1e-10)[:, None, None]
        if self.inferred is not None:
            covariance = gaussian_covariances(self.inferred["scale"], self.inferred["quat"])
            self.inferred_covariances = covariance + np.eye(3)[None] * (
                np.maximum(np.trace(covariance, axis1=1, axis2=2), 1.) * 1e-10)[:, None, None]
        self.lock = threading.RLock()
        self.syncing = False
        self.server = viser.ViserServer(host=args.host, port=args.port, label="OpenRoom3D Object pose and manipulation")
        if not hasattr(self.server.scene, "_add_gaussian_splats"):
            self.server.stop()
            raise RuntimeError("Installed Viser lacks Gaussian preview support; keep current environment and use CUDA render diagnostics")
        self.server.scene.set_up_direction((0., 0., 1.))
        self.server.gui.add_markdown("**Object pose and manipulation**\n\nLeft-drag to orbit, right-drag to pan and scroll to zoom. Drag the axes or rings, or enter a translation or angle. **Reset object** restores the original parameters.\n\n" + ("An optional inferred tabletop appears after movement. It is an estimate from surrounding observations." if self.inferred is not None else "The original background remains in place. ") + "Object selection can miss fine detail.")
        self.selector = self.server.gui.add_dropdown("Object selection", options=tuple(self.objects), initial_value=next(iter(self.objects)))
        self.isolate = self.server.gui.add_checkbox("Isolate the selected object", initial_value=False)
        self.show_axes = self.server.gui.add_checkbox("Show XYZ axes and handles", initial_value=True)
        self.show_box = self.server.gui.add_checkbox("Show oriented bounding box", initial_value=True)
        self.highlight = self.server.gui.add_checkbox("Highlight selected centers", initial_value=False)
        self.heatmap = self.server.gui.add_checkbox("Show query heatmap", initial_value=False)
        self.clean_background = self.server.gui.add_checkbox("Hide background artifacts in the presentation", initial_value=bool(self.presentation),
                                                             disabled=self.presentation is None, visible=self.presentation is not None)
        self.show_inferred = self.server.gui.add_checkbox("Show inferred tabletop after movement", initial_value=self.inferred is not None,
                                                          disabled=self.inferred is None, visible=self.inferred is not None)
        self.mode_status = self.server.gui.add_markdown("")
        self.step = self.server.gui.add_number("Translation step (reconstruction units)", .03, min=.001, max=.5, step=.001)
        self.positions = [self.server.gui.add_number(f"World {axis} translation", 0., step=.005) for axis in "XYZ"]
        self.angles = [self.server.gui.add_number(f"World {axis} rotation (degrees)", 0., step=5.) for axis in "XYZ"]
        for axis in range(3):
            for sign in (-1, 1):
                button = self.server.gui.add_button(f"Move {'XYZ'[axis]} {'−' if sign < 0 else '+'}")
                button.on_click(lambda _, axis=axis, sign=sign: self.nudge(axis, sign))
        self.reset_button = self.server.gui.add_button("Reset object")
        self.focus_button = self.server.gui.add_button("Focus on the object")
        self.save_button = self.server.gui.add_button("Export object and pose")
        self.details = self.server.gui.add_markdown("")
        self.server.gui.add_markdown("Coordinates use reconstruction units. PCA axes can differ from canonical object axes. Multiview selections can retain artifacts. Only original Gaussians are selected. This Gaussian preview can differ from the CUDA reference.")
        # Build the scene first: callbacks and client connects read its state.
        self.select(self.selector.value)
        self.selector.on_update(lambda _: self.select(self.selector.value))
        self.isolate.on_update(lambda _: self.apply_visibility())
        self.show_axes.on_update(lambda _: self.apply_visibility())
        self.show_box.on_update(lambda _: self.apply_visibility())
        self.highlight.on_update(lambda _: self.apply_visibility())
        self.heatmap.on_update(lambda _: self.apply_visibility())
        self.clean_background.on_update(lambda _: self.apply_visibility())
        self.show_inferred.on_update(lambda _: self.apply_visibility())
        for number in self.positions + self.angles:
            # Viser dispatches callbacks even for programmatic value updates,
            # asynchronously. Ignore these to prevent gizmo/Euler feedback.
            number.on_update(lambda event: self.from_numbers() if event.client is not None else None)
        self.reset_button.on_click(lambda _: self.reset())
        self.focus_button.on_click(lambda _: self.focus_all())
        self.save_button.on_click(lambda _: self.export())
        self.server.on_client_connect(lambda client: self.focus(client))

    def select(self, label):
        with self.lock:
            if hasattr(self, "nodes"):
                # Remove children first; visibility and names will be rebuilt.
                for node in reversed(self.nodes):
                    node.remove()
            path, data, pose = self.objects[label]
            compatible = self.presentation.get("compatible_object_sources", [self.presentation.get("object_source"), self.presentation.get("prepared_object_source")]) if self.presentation else []
            self.presentation_active = bool(self.presentation and str(path.resolve()) in compatible)
            self.object_path = path
            self.state = ObjectState(data, pose)
            pivot = np.array(pose["centroid"])
            axes = np.array(pose["rotation_matrix_columns"])
            self.axes_original = axes
            ids = data["gaussian_index"]
            remaining = np.ones(len(self.raw["xyz"]), dtype=bool)
            remaining[ids] = False
            self.nodes = []
            scores = self.raw.get("similarity", np.zeros(len(self.raw["xyz"])))
            from matplotlib import pyplot
            heat_colors = (pyplot.get_cmap("turbo")(np.clip(scores, 0, 1))[:, :3] * 255).astype(np.uint8)
            self.heatmap_node = self.server.scene._add_gaussian_splats(
                "/query_heatmap", centers=self.raw["xyz"], covariances=self.preview_covariances,
                rgbs=heat_colors, opacities=self.raw["opacity"][:, None], visible=False)
            self.nodes.append(self.heatmap_node)
            self.background = self.server.scene._add_gaussian_splats(
                "/background", centers=self.raw["xyz"][remaining], covariances=self.preview_covariances[remaining],
                rgbs=self.raw["rgb"][remaining], opacities=self.raw["opacity"][remaining, None])
            self.nodes.append(self.background)
            if self.presentation_active:
                clean_remaining = remaining & self.background_clean_mask
                self.cleaned_background = self.server.scene._add_gaussian_splats(
                    "/background_cleaned", centers=self.raw["xyz"][clean_remaining],
                    covariances=self.preview_covariances[clean_remaining], rgbs=self.raw["rgb"][clean_remaining],
                    opacities=self.raw["opacity"][clean_remaining, None], visible=False)
                self.nodes.append(self.cleaned_background)
            # The layer only pairs with objects it was built for: their Gaussians are all cleared.
            self.inferred_active = self.inferred is not None and bool(np.isin(ids, self.inferred["scene_cleared_index"]).all())
            if self.inferred_active:
                cleared_remaining = remaining.copy()
                cleared_remaining[self.inferred["scene_cleared_index"]] = False
                self.inferred_scene = self.server.scene._add_gaussian_splats(
                    "/background_inferred_scene", centers=self.raw["xyz"][cleared_remaining],
                    covariances=self.preview_covariances[cleared_remaining], rgbs=self.raw["rgb"][cleared_remaining],
                    opacities=self.raw["opacity"][cleared_remaining, None], visible=False)
                self.inferred_table = self.server.scene._add_gaussian_splats(
                    "/background_inferred_table", centers=self.inferred["xyz"], covariances=self.inferred_covariances,
                    rgbs=self.inferred["rgb"], opacities=self.inferred["opacity"][:, None], visible=False)
                self.nodes.extend((self.inferred_scene, self.inferred_table))
            self.object_frame = self.server.scene.add_frame(
                "/object", position=pivot, wxyz=Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]], show_axes=False)
            self.nodes.append(self.object_frame)
            self.control = self.server.scene.add_transform_controls(
                "/gizmo", position=pivot, wxyz=Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]],
                scale=.22, depth_test=False)
            self.nodes.append(self.control)
            local_means = (data["xyz"] - pivot) @ axes
            local_covariance = axes.T @ self.preview_covariances[ids] @ axes
            self.splats = self.server.scene._add_gaussian_splats(
                "/object/gaussians", centers=local_means, covariances=local_covariance,
                rgbs=data["rgb"], opacities=data["opacity"][:, None])
            self.nodes.append(self.splats)
            self.axes_node = self.server.scene.add_frame("/object/axes", axes_length=.12, axes_radius=.0025)
            self.nodes.append(self.axes_node)
            self.point_overlay = self.server.scene.add_point_cloud(
                "/object/highlight", points=local_means, colors=(255, 180, 40), point_size=.0015, visible=False)
            self.nodes.append(self.point_overlay)
            edges = box_edges((np.array(pose["obb_center"]) - pivot) @ axes, np.eye(3), pose["obb_extent"])
            self.box_nodes = []
            for i, edge in enumerate(edges):
                node = self.server.scene.add_spline_catmull_rom(f"/object/box_{i}", positions=edge,
                                                              color=(255, 180, 40), line_width=2)
                self.box_nodes.append(node)
                self.nodes.append(node)
            self.control.on_update(lambda _: self.from_gizmo())
            self.reset()
            self.apply_visibility()

    def inferred_visible(self):
        """Unmoved objects keep the original scene, so the layer never fills their holes."""
        return (self.inferred_active and self.show_inferred.value and not self.isolate.value and not self.heatmap.value
                and object_has_moved(self.state.rotation, self.state.translation))

    def apply_visibility(self):
        with self.lock:
            if self.presentation_active:
                cleanup_enabled = self.clean_background.value
                self.background.visible = not self.isolate.value and not cleanup_enabled
                self.cleaned_background.visible = not self.isolate.value and cleanup_enabled
                self.clean_background.visible = True
                self.mode_status.content = (f"Local background cleanup: {'enabled' if cleanup_enabled else 'disabled'}.\n\n"
                                            "Cleanup hides artifacts. It leaves unobserved holes empty and can be toggled.")
            else:
                self.background.visible = not self.isolate.value
                self.clean_background.visible = False
                self.mode_status.content = "This object has no local cleanup configuration. Showing the original scene." if self.presentation else ""
            self.show_inferred.visible = self.inferred_active
            if self.inferred_active:
                showing = self.inferred_visible()
                self.inferred_scene.visible = self.inferred_table.visible = showing
                if showing:
                    self.background.visible = False
                    if self.presentation_active:
                        self.cleaned_background.visible = False
                    self.mode_status.content = "**Inferred tabletop visible.** This region was not captured. Its appearance is estimated from the surrounding surface."
                else:
                    self.mode_status.content = "Inferred tabletop hidden. " + ("The original scene is shown until the object moves." if self.show_inferred.value else "Disabled.")
            elif self.inferred is not None:
                self.mode_status.content = "This object has no inferred tabletop. Showing the original scene."
            self.control.visible = self.show_axes.value
            self.axes_node.visible = self.show_axes.value
            self.point_overlay.visible = self.highlight.value
            for node in self.box_nodes:
                node.visible = self.show_box.value
            self.heatmap_node.visible = self.heatmap.value
            self.object_frame.visible = not self.heatmap.value
            if self.heatmap.value:
                self.background.visible = False
                if self.presentation_active:
                    self.cleaned_background.visible = False
                self.control.visible = False

    def update_scene(self):
        pose = self.state.pose()
        self.control.position = np.array(pose["centroid"])
        self.control.wxyz = Rotation.from_matrix(pose["rotation_matrix_columns"]).as_quat()[[3, 0, 1, 2]]
        self.object_frame.position = self.control.position.copy()
        self.object_frame.wxyz = self.control.wxyz.copy()
        matrix = np.array(pose["homogeneous_transform"])
        self.details.content = (f"**Selection: {self.selector.value}**\n\nGaussian count: {len(self.state.original['xyz'])}\n\n"
                                f"Center: `{np.round(pose['centroid'], 4).tolist()}`\n\n"
                                f"4x4 local-to-world pose:\n```text\n{np.array2string(matrix, precision=4)}\n```\n\n"
                                + ("Axes and box use the main support points. Some render-contributing Gaussians may lie outside." if "pose_geometry_scope" in pose else
                                 "Axes use principal components of the main object. Box extents cover the current selection."))
        self.apply_visibility()

    def from_numbers(self):
        with self.lock:
            if self.syncing:
                return
            try:
                rotation = Rotation.from_euler("xyz", [item.value for item in self.angles], degrees=True).as_matrix()
                self.state.set_transform(rotation, [item.value for item in self.positions])
                self.update_scene()
            except ValueError as error:
                self.details.content = f"Invalid input: {error}"

    def from_gizmo(self):
        with self.lock:
            if self.syncing:
                return
            rotation = Rotation.from_quat(self.control.wxyz[[1, 2, 3, 0]]).as_matrix() @ self.axes_original.T
            translation = self.control.position - self.state.pivot
            self.state.set_transform(rotation, translation)
            self.syncing = True
            try:
                for item, value in zip(self.positions, translation):
                    item.value = float(value)
                for item, value in zip(self.angles, Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)):
                    item.value = float(value)
            finally:
                self.syncing = False
            self.update_scene()

    def nudge(self, axis, sign):
        with self.lock:
            self.positions[axis].value += sign * self.step.value
            self.from_numbers()

    def reset(self):
        with self.lock:
            self.syncing = True
            try:
                self.state.reset()
                for item in self.positions + self.angles:
                    item.value = 0.
                self.update_scene()
            finally:
                self.syncing = False

    def focus(self, client):
        pose = self.state.pose()
        center = np.array(pose["centroid"])
        radius = max(float(max(pose["obb_extent"])), .15)
        client.camera.position = center + np.array([-.9, -.8, .6]) * radius * 2
        client.camera.look_at = center
        client.camera.up_direction = (0., 0., 1.)
        client.camera.fov = np.deg2rad(50)

    def focus_all(self):
        for client in self.server.get_clients().values():
            self.focus(client)

    def export(self):
        with self.lock:
            filename = f"object_{time.time_ns()}_moved.npz"
            output = self.args.export_dir / filename
            save_state(self.state, output, self.object_path)
            if self.presentation_active:
                # Export only reversible display-crop metadata. Discard legacy
                # inferred color/table mesh fields from the old preset file.
                cleanup_fields = ("normal", "offset", "origin", "tangent_axes", "footprint_min", "footprint_max",
                                  "cleanup_scope_margin", "background_min_height", "column_min_height", "column_max_height", "scene_source")
                snapshot = {key: self.presentation[key] for key in cleanup_fields}
                snapshot.update(object_export=str(output.resolve()), selected_object_source=str(self.object_path.resolve()),
                                kind="background display cleanup only; no replacement surface",
                                scene_display_mode="original scene", background_cleanup_enabled=self.clean_background.value,
                                background_completion_enabled=False, table_replacement_enabled=False,
                                patch_visible=False, schematic_table_visible=False,
                                synthetic_background_in_object_export=False)
                output.with_suffix(".presentation.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            if self.inferred_active:
                snapshot = {"object_export": str(output.resolve()), "kind": "inferred background, not reconstructed",
                            "inferred_background_source": str(self.args.inferred_background.resolve()),
                            "inferred_background_shown": bool(self.inferred_table.visible),
                            "synthetic_background_in_object_export": False}
                output.with_suffix(".inferred_background.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            self.details.content += f"\n\nExported: `{output}`"
            return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True, help="full query scene NPZ")
    p.add_argument("--object", action="append", required=True, help="LABEL=OBJECT_NPZ; repeat to compare selections")
    p.add_argument("--port", type=int, default=7010)
    p.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    p.add_argument("--export-dir", type=Path, default=Path("outputs/object_edits"))
    p.add_argument("--presentation", type=Path, help="optional scene-specific background display crop; legacy table fields ignored")
    p.add_argument("--inferred-background", type=Path, help="fill_table_layer.py NPZ; shown only after the object moves")
    args = p.parse_args()
    viewer = PoseViewer(args)
    print(f"OpenRoom3D pose viewer at http://localhost:{viewer.server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.server.stop()
