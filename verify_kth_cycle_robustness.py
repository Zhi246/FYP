#!/usr/bin/env python3
"""Independent artifact/protocol audit; never train or choose a model."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import run_kth_cycle_robustness as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/kth_cycle_robustness_v1"))
    args=parser.parse_args(); root=args.output.resolve()
    api=experiment.imports()
    experiment.verify_output(root,api)
    data=np.load(root/"inputs.npz")
    manifest=pd.read_csv(root/"sample_manifest.csv")
    assignment=pd.read_csv(root/"cycle_split.csv")
    pools={name:assignment.loc[assignment.split==name,"bank_cycle_index_zero_based"].to_numpy()
           for name in ("train","validation","test")}
    bank_path=experiment.UPSTREAM/"dataset/empirical/hardware_response_samples.npz"
    archive=np.load(bank_path)
    quantizer=api["quantizer"].load(experiment.UPSTREAM/"dataset/csv/differential_levels.csv",
                                   input_features=16,expected_levels=256,raw_minimum=0,raw_maximum=100)
    bank=api["bank"].load(bank_path,expected_groups=16,expected_levels=quantizer.levels)
    fingerprints=[hashlib.sha256(np.ascontiguousarray(bank.response_samples[:,i]).tobytes()).hexdigest()
                  for i in range(400)]
    assert len(set(fingerprints))==400
    train=manifest.split.to_numpy()=="train"
    parent_versions=json.loads((root.parent/"kth_joint_rbf256_v1/signature.json").read_text())["versions"]
    current_versions=json.loads((root/"signature.json").read_text())["versions"]
    assert all(parent_versions[k]==v for k,v in current_versions.items())
    raw=data["raw"]; q=data["quantized"]
    models={name:joblib.load(root/"models"/(name+".joblib")) for name in experiment.SCHEMES}
    h=experiment.mean_response(bank,q[train],pools["train"])
    for name,alpha in (("A_linear",1.),("I_ideal",.1)):
        x=experiment.model_features(models[name],q[train],h,api).astype(float)
        z=np.column_stack((np.ones(len(x)),x)); target=np.eye(6)[data["labels"][train]]
        penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
        weights=np.asarray(models[name]["head"]["combined_weights"])
        residual=np.linalg.norm((z.T@z+penalty)@weights-z.T@target)/np.linalg.norm(z.T@target)
        assert residual<1e-8,(name,residual)
    folds=pd.read_csv(root/"group_cv_folds.csv")
    expected_train_ids=set(manifest.loc[train,"sample_id"])
    assert set(folds.sample_id)==expected_train_ids
    validation_folds=folds[folds.role=="validation"]
    assert validation_folds.sample_id.nunique()==760 and len(validation_folds)==760
    fold_records=[]
    for fold in range(4):
        frame=folds[folds.fold==fold]
        fit=frame[frame.role=="train"]
        val=frame[frame.role=="validation"]
        assert not set(fit.subject)&set(val.subject)
        assert set(fit.sample_id)|set(val.sample_id)==expected_train_ids
        directory=root/"group_cv"/f"fold_{fold}"
        arrays=np.load(directory/"inputs_and_assignments.npz")
        fit_index=pd.Index(data["sample_ids"]).get_indexer(arrays["fit_sample_ids"])
        val_index=pd.Index(data["sample_ids"]).get_indexer(arrays["validation_sample_ids"])
        assert (fit_index>=0).all() and (val_index>=0).all()
        q_fit,q_val,scaling=experiment.prepare_inputs(raw[fit_index],raw[val_index],quantizer)
        np.testing.assert_array_equal(q_fit,arrays["quantized_fit"])
        np.testing.assert_array_equal(q_val,arrays["quantized_validation"])
        cycles=arrays["training_cycles"]
        assert set(cycles.ravel())<=set(pools["train"])
        indices=bank._resolve_level_indices(q_fit)
        augmented=bank.response_samples[indices[:,None,:],cycles[:,:,None]].reshape(-1,256)
        mean=experiment.mean_response(bank,q_fit,pools["train"])
        for name in experiment.G_NAMES:
            payload=joblib.load(directory/(name+".joblib"))
            state=payload["transform"]
            grouped=(augmented if name==experiment.G_NAMES[2] else mean).astype(float).reshape(-1,16,16)
            np.testing.assert_array_equal(grouped.mean(axis=0),state["channel_mean"])
            np.testing.assert_array_equal(np.maximum(grouped.std(axis=0),1e-12),state["channel_std"])
            np.testing.assert_array_equal(arrays["centers"],state["centers"])
            n=len(q_fit)
            assert np.isclose(state["calibration_alpha"],experiment.sum_alpha(.01,n*8 if name==experiment.G_NAMES[2] else n),rtol=1e-13)
            assert np.isclose(payload["sum_form_head_alpha"],experiment.sum_alpha(.1,n if name==experiment.G_NAMES[0] else n*8),rtol=1e-13)
        fold_records.append({"fold":fold,"train_subjects":sorted(set(fit.subject)),
                             "validation_subjects":sorted(set(val.subject)),
                             "train_only_scaling_and_response_statistics":True,
                             "normalized_regularization_verified":True})
    feature_files=0
    for path in (root/"features").glob("*.npz"):
        values=np.load(path)
        if "sample_ids" not in values:
            assert values["features"].shape[1]==256 and np.isfinite(values["features"]).all()
        else:
            for name in experiment.SCHEMES:
                assert values[name].shape==(len(values["sample_ids"]),16 if name=="A_linear" else 256)
                assert np.isfinite(values[name]).all()
            np.testing.assert_array_equal(values["G0_mean"],values["G1_head_cycle"])
        feature_files+=1
    result={"passed":True,"complete_cycle_vectors_unique":400,
            "reused_control_environment_versions_exact":True,"reused_control_normal_equations_verified":True,
            "group_cv_fold_audit":fold_records,"saved_feature_files_checked":feature_files,
            "g0_g1_saved_gaussian_features_identical":True,
            "scope":"Processed-bank index holdout only; source workbook normalization and chronology not independently verified"}
    experiment.shared.write_json(root/"independent_protocol_verification.json",result)
    print("PASS: controls, cycle uniqueness, per-fold train-only fits, equivalent penalties and all feature shapes.")


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
