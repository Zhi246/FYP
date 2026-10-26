"""Numerical controls for the Cycle experiment, not synthetic accuracy results."""
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run_kth_cycle_robustness as experiment


class CycleRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.api = experiment.imports()

    def test_cycle_partition_is_disjoint_complete_and_reproducible(self):
        pools = experiment.cycle_partition()
        self.assertEqual([len(v) for v in pools.values()], [240, 80, 80])
        self.assertEqual(len(set(np.concatenate(list(pools.values())))), 400)
        for name, pool in pools.items():
            np.testing.assert_array_equal(pool, experiment.cycle_partition()[name])

    def actual_bank(self):
        quantizer = self.api["quantizer"].load(experiment.UPSTREAM/"dataset/csv/differential_levels.csv",
            input_features=16, expected_levels=256, raw_minimum=0, raw_maximum=100)
        bank = self.api["bank"].load(experiment.UPSTREAM/"dataset/empirical/hardware_response_samples.npz",
            expected_groups=16, expected_levels=quantizer.levels)
        q = quantizer.transform(np.tile(np.linspace(0, 100, 16), (5, 1)))
        return bank, q, quantizer

    def test_training_sampling_shares_one_cycle_across_all_dimensions_and_channels(self):
        bank, q, _ = self.actual_bank()
        pool = experiment.cycle_partition()["train"]
        responses, cycles = experiment.sampled_training_responses(bank, q, pool)
        self.assertEqual(responses.shape, (40, 256))
        self.assertTrue(set(cycles.ravel()) <= set(pool))
        indices = bank._resolve_level_indices(q)
        for n in range(5):
            self.assertEqual(len(set(cycles[n])), 8)
            for copy in range(8):
                expected = bank.response_samples[indices[n], cycles[n, copy]].reshape(256)
                np.testing.assert_array_equal(responses[n*8+copy], expected)

    def test_evaluation_pool_isolation_and_noise_conditions_share_draws(self):
        bank, q, _ = self.actual_bank()
        pools = experiment.cycle_partition()
        references = np.quantile(np.abs(bank.response_samples[:, pools["train"]]), .95, axis=(0,1))
        clean, cycles = experiment.draw_response(bank, q, pools["test"], 3042, references)
        noisy, same_cycles = experiment.draw_response(bank, q, pools["test"], 3042, references, noise=True)
        np.testing.assert_array_equal(cycles, same_cycles)
        self.assertTrue(set(cycles) <= set(pools["test"]))
        self.assertFalse(set(cycles) & set(pools["train"]))
        indices = bank._resolve_level_indices(q)
        expected = bank.response_samples[indices, cycles[:, None]].reshape(5, 256)
        np.testing.assert_array_equal(clean, expected)
        np.testing.assert_array_equal(np.sign(clean), np.sign(noisy))

    def test_pool_mean_does_not_use_other_cycle_indices(self):
        bank, q, _ = self.actual_bank()
        pool = experiment.cycle_partition()["train"]
        original = experiment.mean_response(bank, q, pool)
        samples = bank.response_samples.copy()
        other = np.setdiff1d(np.arange(400), pool)
        samples[:, other] = 1e6
        changed = type(bank)(bank.levels, samples, bank.group_magnitude_references)
        np.testing.assert_array_equal(original, experiment.mean_response(changed, q, pool))

    def test_ridge_duplicate_rows_preserve_average_objective(self):
        # Random values are for a mathematical unit test only, never dataset results.
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 7)); y = np.arange(40) % 6
        original = experiment.fit_head(x, y, .1, self.api)
        repeated = experiment.fit_head(np.repeat(x, 8, axis=0), np.repeat(y, 8), .1, self.api)
        self.assertAlmostEqual(experiment.sum_alpha(.1, 320), 8*experiment.sum_alpha(.1, 40))
        np.testing.assert_allclose(original.combined_weights_.numpy(), repeated.combined_weights_.numpy(),
                                   rtol=1e-10, atol=1e-10)

    def test_calibration_duplicate_rows_preserve_average_objective(self):
        rng = np.random.default_rng(42)
        h = rng.normal(size=(40, 6)); q = rng.uniform(size=(40, 3)); centers = q[:7]
        def fit(hh, qq):
            transformer = self.api["transform"](3, 2, 7, .25, experiment.sum_alpha(.01, len(hh)),
                                                  1e-6, 1., 1e-12, 42, 300)
            return transformer.fit(hh, qq, preset_centers=centers)
        original = fit(h, q)
        repeated = fit(np.repeat(h, 8, axis=0), np.repeat(q, 8, axis=0))
        np.testing.assert_allclose(original.calibration_weights_, repeated.calibration_weights_, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(original.transform(h), repeated.transform(h), rtol=2e-6, atol=2e-7)

    def test_input_scaling_fits_training_only_and_handles_zero(self):
        _, _, quantizer = self.actual_bank()
        train = np.full((3,16), .1)
        q1, val, scaling = experiment.prepare_inputs(train, np.full((2,16), 100.), quantizer)
        q2, _, second = experiment.prepare_inputs(train, np.zeros((2,16)), quantizer)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(scaling["maximum"], .1)
        self.assertEqual(scaling["denominator"], second["denominator"])
        self.assertEqual(scaling["other_clipped_value_fraction"], 1.)
        np.testing.assert_array_equal(val, 1.)
        q0, _, zeros = experiment.prepare_inputs(np.zeros((3,16)), np.zeros((2,16)), quantizer)
        self.assertEqual(zeros["denominator"], 1.)
        np.testing.assert_array_equal(q0, 0.)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
