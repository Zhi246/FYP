#!/usr/bin/env python3
"""Read-only model inference audit. Does not retrain or reselect parameters."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import run_kth_joint_rbf as core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/kth_joint_rbf256_strict_v1"))
    args=parser.parse_args()
    output=args.output.resolve()
    parent=Path("outputs/kth_joint_rbf256_v1").resolve()
    signature=json.loads((parent/"signature.json").read_text())
    for path,expected in signature["source_hashes"].items():
        assert core.sha256(path)==expected,("Source changed",path)
    sys.path.insert(0,"/home/ziyan/SURF2026/rbf_hardware/src")
    from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer
    api={"transform":FullJointGaussianTransformer}
    archive=np.load(output/"input_and_bank_mean.npz",allow_pickle=False)
    q,h,y,splits=[archive[k] for k in ("quantized","bank_mean","labels","splits")]
    core.verify_saved(output,q,h,y,splits,api)
    payload=joblib.load(output/"models/Hardware_joint256_tuned.joblib")
    assert payload["transform"]["factor_lower"]==1e-6
    assert payload["transform"]["factor_upper"]==1.
    grouped=h[splits=="train"].astype(float).reshape(-1,16,16)
    np.testing.assert_array_equal(grouped.mean(axis=0),payload["transform"]["channel_mean"])
    np.testing.assert_array_equal(np.maximum(grouped.std(axis=0),1e-12),payload["transform"]["channel_std"])
    np.testing.assert_array_equal(np.load(output/"centers.npy"),payload["transform"]["centers"])
    assert json.loads((output/"model_checks.json").read_text())["passed"]
    print("PASS: source hashes unchanged; exact saved predictions; strict clip; train-only channel statistics and centers.")


if __name__=="__main__":
    with threadpool_limits(limits=1):main()
