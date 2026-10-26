"""Synthetic numerical unit tests only; reported experiment uses real KTH videos."""
import unittest

import numpy as np

from run_experiment import gaussian_sum
from run_kth import SUBJECTS, centers_from_train, sampling_plan
from tune_gaussian import scale_fit_and_eval


class KTHNumericalTests(unittest.TestCase):
    def test_subjects_disjoint_and_complete(self):
        groups = [set(s) for s in SUBJECTS.values()]
        self.assertEqual(set.union(*groups), set(range(1, 26)))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])

    def test_time_sampling_not_seventeen_adjacent_frames(self):
        times, indices = sampling_plan(101, 200, 25.)
        self.assertAlmostEqual(times[-1] - times[0], 1.)
        self.assertEqual(len(indices), 17)
        self.assertEqual(len(set(indices)), 17)
        self.assertGreater(indices[-1] - indices[0], 16)
        self.assertTrue(all(100 <= i <= 199 for i in indices))

    def test_short_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            sampling_plan(1, 24, 25.)

    def test_uint8_is_not_subtracted(self):
        frames = np.array([255, 0], dtype=np.uint8).astype(np.float32) / 255.
        self.assertEqual(float(np.abs(np.diff(frames))[0]), 1.)

    def test_scale_fit_only_training_and_clip_evaluation(self):
        train = np.full((3, 16), .2)
        evaluation = np.full((2, 16), .6)
        a, b, settings = scale_fit_and_eval(train, evaluation)
        self.assertAlmostEqual(settings["training_global_max"], .2)
        self.assertTrue(np.all(a == 1) and np.all(b == 1))
        self.assertEqual(settings["eval_clipped_ratio"], 1.)

    def test_zero_maximum_has_defined_fallback(self):
        a, b, settings = scale_fit_and_eval(np.zeros((2, 16)), np.full((1, 16), .5))
        self.assertTrue(settings["zero_max_fallback"])
        self.assertEqual(settings["divisor"], 1.)
        self.assertTrue(np.all(a == 0) and np.all(b == .5))

    def test_gaussian_shape_manual_sum_and_invariance(self):
        x = np.arange(32).reshape(2, 16) / 31.
        mu = np.linspace(0, 1, 16)
        result = gaussian_sum(x, .1, mu)
        manual = np.array([[sum(np.exp(-(float(v) - float(center)) ** 2 / .02) for v in row)
                            for center in mu] for row in x])
        self.assertEqual(result.shape, (2, 16))
        np.testing.assert_allclose(result, manual, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result, gaussian_sum(x[:, ::-1], .1, mu), atol=1e-12, rtol=1e-12)

    def test_duplicates_are_counted_and_peak_is_one(self):
        mu = np.linspace(0, 1, 16)
        x = np.full((1, 16), mu[3])
        result = gaussian_sum(x, .1, mu)
        self.assertEqual(float(result[0, 3]), 16.)

    def test_degenerate_centers_retain_sixteen_outputs(self):
        x = np.full((2, 16), .3)
        centers = centers_from_train("uniform_p01_p99", x)
        self.assertEqual(centers.shape, (16,))
        self.assertEqual(len(np.unique(centers)), 1)
        self.assertEqual(gaussian_sum(x, .1, centers).shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
