"""Implementation fixtures only; no synthetic values enter experiment metrics."""
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import tune_kth_spatiotemporal as task


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.api = task.cycle.imports()

    def test_grid_has_defaults_and_fixed_size(self):
        configs = task.configurations()
        self.assertEqual(len(configs), 88)
        self.assertEqual(len({r["id"] for r in configs}), 88)
        self.assertEqual(sum(task.is_default(r) for r in configs), 4)
        self.assertEqual(sum(r["scheme"] == "A_linear" for r in configs), 4)
        self.assertEqual(sum(r["scheme"].startswith("G") for r in configs), 72)

    def test_selection_ignores_test_scores_and_ideal_as_hardware(self):
        rows = []
        for c in task.configurations():
            rows.append(dict(**c, validation_macro_f1_mean=.5, validation_accuracy_mean=.5,
                validation_macro_f1_std=.01, test_accuracy=1. if not task.is_default(c) else 0.))
        ideal = next(r for r in rows if r["scheme"] == "I_ideal")
        ideal["validation_macro_f1_mean"] = 1.
        result = task.select(rows)
        self.assertEqual(result["gaussian"]["scheme"], "G0_mean")
        self.assertTrue(task.is_default(result["gaussian"]))
        self.assertTrue(task.is_default(result["per_scheme"]["A_linear"]))
        winner = next(r for r in rows if r["scheme"] == "G1_head_cycle" and r["sigma"] == .5)
        winner["validation_macro_f1_mean"] = .6
        self.assertEqual(task.select(rows)["gaussian"]["id"], winner["id"])

    def test_normalized_regularization(self):
        for anchor in task.HEAD_ALPHAS:
            self.assertAlmostEqual(task.cycle.sum_alpha(anchor, 6080)/6080, anchor/760)
            self.assertAlmostEqual(task.cycle.sum_alpha(anchor, 568)/568, anchor/760)

    def test_original_transform_matches_independent_joint_product(self):
        rng = np.random.default_rng(42)
        q = rng.uniform(0, 1, (280, 16)).astype(np.float32)
        h = rng.normal(size=(280, 256)).astype(np.float32)
        centers = q[:256].astype(float)
        cfg = task.config("G0_mean", .1, .125, .001)
        transform = task.transformer(cfg, h, q, centers, self.api)
        grouped = h.astype(float).reshape(-1, 16, 16)
        independent = np.ones((12, 256))
        for d in range(16):
            z = np.column_stack((np.ones(12),
                (grouped[:12, d]-grouped.mean(axis=0)[d])/grouped.std(axis=0)[d]))
            independent *= np.clip(z@transform.calibration_weights_[d], 1e-6, 1.)
        np.testing.assert_allclose(transform.transform(h[:12]), independent, atol=2e-7, rtol=2e-6)
        self.assertEqual(transform.transform(h[:12]).shape, (12, 256))
        self.assertEqual(transform.joint_sigma, .125)
        self.assertEqual(transform.factor_upper, 1.)
        np.testing.assert_array_equal(transform.centers_, centers)


if __name__ == "__main__":
    with threadpool_limits(limits=1): unittest.main()
