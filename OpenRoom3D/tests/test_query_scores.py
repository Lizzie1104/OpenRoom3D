"""CPU checks for the softmax and LERF pairwise-min query scores."""
from __future__ import annotations

import types
import unittest

import torch

from scripts.export_query_gaussians import _score_in_chunks, build_parser
from scripts.query_negatives import DEFAULT_NEGATIVES, without_query


def fake_model(features):
    mlp = types.SimpleNamespace(per_gaussian_forward=lambda x: {"clip": x})
    return types.SimpleNamespace(means=features, distill_features=features, feature_mlp=mlp, main_feature_name="clip")


class QueryScoreTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.model = fake_model(torch.randn(50, 8, generator=generator))
        unit = lambda x: x / x.norm(dim=1, keepdim=True)
        self.positive = unit(torch.randn(2, 8, generator=generator))
        self.negatives = unit(torch.randn(4, 8, generator=generator))

    def score(self, negatives, mode, positive=None):
        positive = self.positive[:1] if positive is None else positive
        return _score_in_chunks(self.model, positive, negatives, .05, 7, mode)

    def test_single_negative_modes_agree(self):
        torch.testing.assert_close(self.score(self.negatives[:1], "softmax"), self.score(self.negatives[:1], "pairwise-min"))

    def test_pairwise_min_never_rises_with_more_negatives(self):
        fewer, more = self.score(self.negatives[:2], "pairwise-min"), self.score(self.negatives, "pairwise-min")
        self.assertTrue(bool((more <= fewer + 1e-7).all()))

    def test_pairwise_min_uses_best_positive(self):
        both = self.score(self.negatives, "pairwise-min", self.positive)
        each = torch.stack([self.score(self.negatives, "pairwise-min", self.positive[i:i + 1]) for i in range(2)])
        torch.testing.assert_close(both, each.max(dim=0).values)

    def test_scores_are_probabilities(self):
        for mode in ("softmax", "pairwise-min"):
            scores = self.score(self.negatives, mode)
            self.assertEqual(scores.shape, (50,))
            self.assertTrue(bool(((scores >= 0) & (scores <= 1)).all()))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.score(self.negatives, "mean")


class NegativeDefaultTests(unittest.TestCase):
    def test_exporter_defaults_to_scene_negatives(self):
        args = build_parser().parse_args(["--config", "config.yml", "--query", "laptop"])
        self.assertEqual((args.score_mode, args.negative), ("pairwise-min", DEFAULT_NEGATIVES))

    def test_query_never_competes_with_its_own_category(self):
        kept = without_query(DEFAULT_NEGATIVES, "mouse").split(",")
        self.assertNotIn("computer mouse", kept)
        self.assertIn("table", kept)
        self.assertNotIn("water bottle", without_query(DEFAULT_NEGATIVES, "Bottle, thermos").split(","))
        self.assertEqual(without_query(DEFAULT_NEGATIVES, "laptop"), DEFAULT_NEGATIVES)

    def test_all_negatives_removed_is_an_error(self):
        with self.assertRaises(ValueError):
            without_query("table", "table")


if __name__ == "__main__":
    unittest.main()
