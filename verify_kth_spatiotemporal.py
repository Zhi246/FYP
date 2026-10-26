#!/usr/bin/env python3
"""Independent preprocessing/fold/normal-equation audit; never refit heads."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import run_kth_spatiotemporal as task


def audit_fit(models,q,y,bank,pool,cycles,centers,api):
    mean=task.cycle.mean_response(bank,q,pool)
    indices=bank._resolve_level_indices(q)
    augmented=bank.response_samples[indices[:,None,:],cycles[:,:,None]].reshape(-1,256)
    transformer=api["transform"].from_state_dict(models["G0_mean"]["transform"])
    grouped=mean.astype(float).reshape(-1,16,16)
    np.testing.assert_array_equal(grouped.mean(axis=0),transformer.channel_mean_)
    np.testing.assert_array_equal(np.maximum(grouped.std(axis=0),1e-12),transformer.channel_std_)
    np.testing.assert_array_equal(centers,transformer.centers_)
    assert transformer.factor_lower==1e-6 and transformer.factor_upper==1.
    assert np.isclose(transformer.calibration_alpha,task.cycle.sum_alpha(.01,len(q)),rtol=1e-13)
    for d in (0,7,15):
        z=np.column_stack((np.ones(len(q)),(grouped[:,d]-transformer.channel_mean_[d])/transformer.channel_std_[d]))
        target=np.exp(-(q[:,d,None]-centers[None,:,d])**2/(2*.25**2))
        penalty=np.eye(17)*transformer.calibration_alpha;penalty[0,0]=0
        residual=(z.T@z+penalty)@transformer.calibration_weights_[d]-z.T@target
        assert np.linalg.norm(residual)/max(np.linalg.norm(z.T@target),1e-30)<1e-8
    for name in task.SCHEMES:
        model=models[name]
        if name=="G1_head_cycle":
            features=task.cycle.model_features(model,np.repeat(q,8,axis=0),augmented,api)
            target_labels=np.repeat(y,8)
        else:
            features=task.cycle.model_features(model,q,mean,api);target_labels=y
        z=np.column_stack((np.ones(len(features)),features.astype(float)))
        alpha=task.cycle.sum_alpha(1. if name=="A_linear" else .1,len(features))
        penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
        target=np.eye(6)[target_labels]
        residual=(z.T@z+penalty)@np.asarray(model["head"]["combined_weights"])-z.T@target
        assert np.linalg.norm(residual)/max(np.linalg.norm(z.T@target),1e-30)<1e-8
    np.testing.assert_array_equal(models["A_linear"]["mean"],q.astype(float).mean(axis=0))
    np.testing.assert_array_equal(models["A_linear"]["std"],np.maximum(q.astype(float).std(axis=0),1e-12))
    for field in ("centers","calibration_weights","channel_mean","channel_std"):
        np.testing.assert_array_equal(models["G0_mean"]["transform"][field],models["G1_head_cycle"]["transform"][field])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/kth_spatiotemporal_v1"))
    args=parser.parse_args();root=args.output.resolve();api=task.cycle.imports()
    task.verify(root,api)
    sig=json.loads((root/"signature.json").read_text());source=Path(sig["old_results"])
    pre=root/"preprocessing";pre_sig=json.loads((pre/"signature.json").read_text())
    assert task.cycle.shared.sha256("extract_kth_spatiotemporal.py")==pre_sig["extractor_sha256"]
    all_checks=pd.read_csv(pre/"all_sample_reconstruction_checks.csv")
    assert len(all_checks)==2389 and all_checks.passed.all()
    old=np.load(source/"inputs.npz");new=np.load(root/"new_inputs.npz")
    np.testing.assert_array_equal(old["sample_ids"],new["sample_ids"])
    frames=np.load(pre/"training_examples.npz")
    lookup=pd.Index(new["sample_ids"])
    for index,identifier in enumerate(frames["sample_ids"]):
        value=frames["gray_frames_u8"][index].astype(float)/255.
        expected=[]
        for y0,y1,x0,x1 in [(0,32,0,32),(0,32,32,64),(32,64,0,32),(32,64,32,64)]:
            for group in range(4):
                expected.append(np.mean([np.abs(value[i+1,y0:y1,x0:x1]-value[i,y0:y1,x0:x1]).mean()
                                         for i in range(group*4,group*4+4)]))
        actual=new["raw"][lookup.get_loc(identifier)]
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
    quantizer=api["quantizer"].load(task.cycle.UPSTREAM/"dataset/csv/differential_levels.csv",input_features=16,
        expected_levels=256,raw_minimum=0,raw_maximum=100)
    bank=api["bank"].load(task.cycle.UPSTREAM/"dataset/empirical/hardware_response_samples.npz",expected_groups=16,expected_levels=quantizer.levels)
    pools=task.cycle.cycle_partition();train=new["splits"]=="train"
    _,q,scaling=task.cycle.prepare_inputs(new["raw"][train],new["raw"],quantizer)
    np.testing.assert_array_equal(q,new["quantized"])
    assert scaling["maximum"]==json.loads((root/"new_input_scaling.json").read_text())["maximum"]
    models={name:joblib.load(root/"models"/task.NEW/(name+".joblib")) for name in task.SCHEMES}
    cycles=np.load(root/"new_training_responses.npz")["cycle_indices"]
    np.testing.assert_array_equal(cycles,np.load(source/"training_responses.npz")["cycle_indices"])
    audit_fit(models,q[train],new["labels"][train],bank,pools["train"],cycles,new["centers"],api)
    fold_records=[]
    manifest=pd.read_csv(root/"sample_manifest.csv")
    for fold in range(4):
        directory=root/"group_cv"/task.NEW/f"fold_{fold}"
        arrays=np.load(directory/"inputs_and_assignments.npz")
        old_arrays=np.load(source/"group_cv"/f"fold_{fold}"/"inputs_and_assignments.npz")
        for key in ("fit_sample_ids","validation_sample_ids","training_cycles"):
            np.testing.assert_array_equal(arrays[key],old_arrays[key])
        fit=lookup.get_indexer(arrays["fit_sample_ids"]);val=lookup.get_indexer(arrays["validation_sample_ids"])
        assert not set(manifest.iloc[fit].subject)&set(manifest.iloc[val].subject)
        assert set(manifest.iloc[fit].split)=={"train"} and set(manifest.iloc[val].split)=={"train"}
        qt,qv,fold_scaling=task.cycle.prepare_inputs(new["raw"][fit],new["raw"][val],quantizer)
        np.testing.assert_array_equal(qt,arrays["quantized_fit"]);np.testing.assert_array_equal(qv,arrays["quantized_validation"])
        models={name:joblib.load(directory/(name+".joblib")) for name in task.SCHEMES}
        audit_fit(models,qt,new["labels"][fit],bank,pools["train"],arrays["training_cycles"],arrays["centers"],api)
        fold_records.append({"fold":fold,"same_subject_assignment_as_old_input":True,
                             "same_training_cycles_as_old_input":True,"fold_only_scaling_and_fit_verified":True})
    for name in task.SCHEMES:
        assert task.cycle.shared.sha256(root/"models"/task.OLD/(name+".joblib"))==task.cycle.shared.sha256(source/"models"/(name+".joblib"))
    task.cycle.shared.write_json(root/"independent_protocol_verification.json",{
        "passed":True,"old_head_files_exactly_reused":True,"all_2389_old_features_and_sampling_unchanged":True,
        "independent_new_feature_formula_training_examples":12,"same_training_cycle_rows":True,
        "new_main_fit_normal_equations_verified":True,"group_cv_folds":fold_records,
        "test_scores_not_used_for_selection":True})
    print("PASS: old fits reused exactly; new formula, train-only fits, equal folds/Cycles and normal equations verified.",flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
