"""Synthetic math tests only. The parameter experiment uses cached real KTH features."""
import unittest

import numpy as np

from run_experiment import gaussian_sum
from run_kth import CLASSES, score
from tune_kth_gaussian import CENTER_MODES, CS, SIGMAS, centers_from_training, metrics_from_encoded


class ExpandedGaussianTests(unittest.TestCase):
    def test_candidate_count(self):
        self.assertEqual(len(CS) * (1 + len(CENTER_MODES) * len(SIGMAS)), 791)

    def test_centers_all_sixteen_monotone_and_in_range(self):
        train = np.linspace(.01, .8, 160).reshape(10, 16)
        for mode in CENTER_MODES:
            centers = centers_from_training(mode, train)
            self.assertEqual(centers.shape, (16,))
            self.assertTrue(np.all(np.diff(centers) >= 0))
            self.assertTrue(0 <= centers.min() <= centers.max() <= 1)

    def test_training_quantiles_not_evaluation(self):
        train = np.linspace(0., .2, 160).reshape(10, 16)
        centers = centers_from_training("quantile_p01_p99", train)
        np.testing.assert_array_equal(centers, np.quantile(train, np.linspace(.01, .99, 16)))
        self.assertLess(centers.max(), .2)

    def test_coincident_centers_not_removed(self):
        train = np.full((10, 16), .3)
        for mode in ["uniform_p01_p99", "uniform_p05_p95", "quantile_p01_p99", "quantile_p05_p95"]:
            centers = centers_from_training(mode, train)
            self.assertEqual(len(centers), 16)
            self.assertEqual(len(np.unique(centers)), 1)
            self.assertEqual(gaussian_sum(train, .1, centers).shape, (10, 16))

    def test_power_layout_has_defined_endpoint_and_meaning(self):
        train = np.ones((3, 16))
        for power in [2, 3]:
            np.testing.assert_array_equal(centers_from_training(f"power{power}_0_1", train), np.linspace(0, 1, 16) ** power)

    def test_every_layout_and_width_stays_invariant(self):
        train = np.linspace(.001, .9, 80).reshape(5, 16)
        for mode in CENTER_MODES:
            for sigma in SIGMAS:
                a = gaussian_sum(train, sigma, centers_from_training(mode, train))
                b = gaussian_sum(train[:, ::-1], sigma, centers_from_training(mode, train))
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)

    def test_fast_metrics_agree_with_sklearn_with_absent_predictions(self):
        actual = np.arange(60) % 6
        pred = np.arange(60) % 4
        fast = metrics_from_encoded(actual, pred)
        reference = score(np.asarray(CLASSES)[actual], np.asarray(CLASSES)[pred])
        for key in reference:
            self.assertAlmostEqual(fast[key], reference[key], places=14)


if __name__ == "__main__":
    unittest.main()
