"""Numerical implementation tests, not substitute classification experiments."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, "/home/ziyan/SURF2026/rbf_hardware/src")
from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer, benchmark_kmeans
from rbf_hardware.modeling.ridge_classifier import RidgeLinearClassifier
from rbf_hardware.inference.preprocessing import DifferentialLevelQuantizer
from rbf_hardware.inference.empirical_response import EmpiricalHardwareResponseBank

ROOT = Path("/home/ziyan/SURF2026/rbf_hardware")


class OriginalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(42)

    def test_quantizer_and_bank_layout(self):
        quantizer = DifferentialLevelQuantizer.load(ROOT/"dataset/csv/differential_levels.csv",
            input_features=16, expected_levels=256, raw_minimum=0, raw_maximum=100)
        bank = EmpiricalHardwareResponseBank.load(ROOT/"dataset/empirical/hardware_response_samples.npz",
            expected_groups=16, expected_levels=quantizer.levels)
        q = quantizer.transform(np.tile(np.linspace(0,100,16), (3,1)))
        self.assertEqual(q.shape, (3,16))
        actual = bank.simulate(q,sampling_mode="mean",random_generator=self.rng)
        indices = bank._resolve_level_indices(q)
        expected = bank.response_samples.mean(axis=1,dtype=np.float64)[indices].astype(np.float32).reshape(3,256)
        np.testing.assert_array_equal(actual,expected)
        # The mean mode contains neither cycle draws nor added simulator noise.
        other = bank.simulate(q,sampling_mode="mean",random_generator=np.random.default_rng(9),
                              minimum_noise_rate=.5,maximum_noise_rate=.9)
        np.testing.assert_array_equal(actual,other)

    def test_empirical_zero_noise_preserves_one_cycle_per_sample(self):
        quantizer = DifferentialLevelQuantizer.load(ROOT/"dataset/csv/differential_levels.csv",
            input_features=16, expected_levels=256, raw_minimum=0, raw_maximum=100)
        bank = EmpiricalHardwareResponseBank.load(ROOT/"dataset/empirical/hardware_response_samples.npz",
            expected_groups=16, expected_levels=quantizer.levels)
        q = np.tile(bank.levels[np.arange(16)],(3,1))
        actual=bank.simulate(q,sampling_mode="empirical",random_generator=np.random.default_rng(42),
                             minimum_noise_rate=0,maximum_noise_rate=0)
        cycles=np.random.default_rng(42).integers(0,400,size=(3,1))
        expected=bank.response_samples[np.tile(np.arange(16),(3,1)),cycles].reshape(3,256)
        np.testing.assert_array_equal(actual,expected)

    def test_transform_matches_independent_product_and_roundtrip(self):
        # Random arrays here are strictly unit-test inputs, never reported as data.
        h=self.rng.normal(size=(40,256)); q=self.rng.uniform(size=(40,16))
        transform=FullJointGaussianTransformer(16,16,7,.25,1.,1e-6,1.,1e-12,42,300)
        transform.fit(h,q,preset_centers=q[:7])
        actual=transform.transform(h)
        self.assertEqual(actual.shape,(40,7))
        z=(h.reshape(-1,16,16)-transform.channel_mean_)/transform.channel_std_
        expected=np.ones((40,7))
        for d in range(16):
            design=np.column_stack((np.ones(40),z[:,d]))
            expected*=np.clip(design@transform.calibration_weights_[d],1e-6,1.)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)
        restored=FullJointGaussianTransformer.from_state_dict(transform.state_dict())
        np.testing.assert_array_equal(actual,restored.transform(h))
        ideal=np.exp(-np.square(q[:,None,:]-q[None,:7,:]).sum(axis=2)/(2*.25**2))
        np.testing.assert_allclose(transform.ideal_pc_transform(q),ideal,rtol=2e-6,atol=2e-7)

    def test_ridge_matches_independent_unpenalized_intercept_solve(self):
        x=self.rng.normal(size=(40,8)); y=np.arange(40)%6; alpha=.1
        head=RidgeLinearClassifier(8,list(range(6)),alpha,True,False,torch.device("cpu"),torch.float64)
        head.fit(torch.as_tensor(x),torch.as_tensor(y))
        z=np.column_stack((np.ones(40),x)); penalty=np.eye(9)*alpha;penalty[0,0]=0
        independent=np.linalg.solve(z.T@z+penalty,z.T@np.eye(6)[y])
        np.testing.assert_allclose(head.combined_weights_.numpy(),independent,rtol=1e-10,atol=1e-10)

    def test_kmeans_deterministic(self):
        x=self.rng.uniform(size=(40,16))
        np.testing.assert_array_equal(benchmark_kmeans(x,7,42,30),benchmark_kmeans(x,7,42,30))


if __name__=="__main__":
    with threadpool_limits(limits=1):
        unittest.main()
