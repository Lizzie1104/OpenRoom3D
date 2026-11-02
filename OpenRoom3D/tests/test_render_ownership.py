"""Rendering attribution tests, including a real two-Gaussian CUDA check."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_render_contributors import select_from_mass, contribution_mass, contribution_signature


class OwnershipSelectionTest(unittest.TestCase):
    def test_visible_background_rejects_seed_unobserved_seed_survives(self):
        foreground = np.array([4., .1, 0., 2., .001])
        background = np.array([.1, 4., 0., 2., 0.])
        seed = np.array([False, True, True, False, False])
        selected, score = select_from_mass(foreground, background, seed, .5, .01)
        np.testing.assert_array_equal(selected, [True, False, True, True, False])
        self.assertGreater(score[0], .9)

    def test_invalid_mass_rejected(self):
        with self.assertRaises(ValueError):
            select_from_mass(np.array([-1.]), np.array([1.]), np.array([True]))

    @unittest.skipUnless(torch.cuda.is_available(), "gsplat CUDA required")
    def test_gradient_mass_measures_masked_render_and_freezes_geometry(self):
        raw = {"xyz": np.array([[-.2, 0, 2], [.2, 0, 2]], dtype=np.float32),
               "scale": np.tile([.03, .03, .03], (2, 1)).astype(np.float32),
               "quat": np.tile([1., 0., 0., 0.], (2, 1)).astype(np.float32),
               "opacity": np.ones(2, dtype=np.float32) * .9}
        original = {k: v.copy() for k, v in raw.items()}
        mask = np.zeros((64, 64), dtype=bool)
        mask[:, :32] = True
        view = {"mask": mask, "viewmat": np.eye(4, dtype=np.float32),
                "intrinsics": np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1]], dtype=np.float32),
                "camera_index": 0}
        fg, bg, total = contribution_mass(raw, [view], "cuda")
        self.assertGreater(fg[0, 0], 0)
        self.assertGreater(bg[0, 1], 0)
        self.assertLess(bg[0, 0], .001)
        self.assertLess(fg[0, 1], .001)
        selected, _ = select_from_mass(fg.sum(0), bg.sum(0), np.zeros(2, dtype=bool))
        np.testing.assert_array_equal(selected, [True, False])
        self.assertTrue((total >= fg + bg - 1e-5).all())
        for key in original:
            np.testing.assert_array_equal(raw[key], original[key])
        signature = contribution_signature(raw, [view])
        view["mask"] = ~mask
        self.assertNotEqual(signature, contribution_signature(raw, [view]))
