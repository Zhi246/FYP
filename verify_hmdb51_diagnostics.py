#!/usr/bin/env python3
"""Read-only numerical verification of saved folds/features/models/metrics.

Only the verification report is written. No classifiers are re-fitted.
"""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from run_experiment import CLASSES
from diagnose_hmdb51 import original_results_unchanged
from tune_gaussian import write_json


def independent_centers(mode, training):
    if mode == "uniform_0_1":
        return np.linspace(0, 1, 16)
    if mode == "uniform_p01_p99":
        return np.linspace(*np.percentile(training, [1, 99]), 16)
    return np.percentile(training, np.linspace(1, 99, 16))


def response(values, centers, sigma):
    return np.exp(-np.square(values[:, :, None] - centers[None, None, :]) / (2*sigma*sigma))


def main():
    start=time.perf_counter();project=Path(__file__).resolve().parent
    output=project/"outputs/hmdb51_diagnostics_v2"
    records=pd.DataFrame(joblib.load(project/"outputs/hmdb51_six_split1/feature_cache.joblib")["records"])
    records["sample_id"]=np.arange(len(records))
    included=records[records.included].reset_index(drop=True)
    raw=np.array(included.feature.tolist()); labels=included["class"].to_numpy()
    by_id={int(row.sample_id):np.array(row.feature) for _,row in included.iterrows()}
    label_by_id=dict(zip(included.sample_id,included["class"]))
    split_by_id=dict(zip(included.sample_id,included.split))
    folds=pd.read_csv(output/"cv/fold_sample_manifest.csv")
    validation=folds[folds.role=="validation"]
    assert len(validation)==412 and validation.sample_id.nunique()==412
    assert len(folds)==3*412 and set(folds.official_split)=={"train"}
    selection=json.loads((output/"all_parameters_frozen_before_test.json").read_text())
    for fold in range(1,4):
        data=np.load(output/f"cv/fold_{fold}_AB_features.npz")
        tid,vid=data["training_sample_ids"],data["validation_sample_ids"]
        assert not set(tid)&set(vid)
        assert all(split_by_id[int(i)]=="train" for i in [*tid,*vid])
        tr=np.array([by_id[int(i)] for i in tid]);va=np.array([by_id[int(i)] for i in vid])
        assert np.array_equal(tr,data["raw_train"]) and np.array_equal(va,data["raw_validation"])
        divisor=tr.max() if tr.max()>0 else 1
        ft,fv=tr/divisor,np.clip(va/divisor,0,1)
        assert np.array_equal(ft,data["scaled_train"]) and np.array_equal(fv,data["scaled_validation"])
        for key in data.files:
            if not key.startswith("B__"):continue
            config,role=key.rsplit("_",1)
            mode=config.split("__")[1];sigma=float(config.split("__sigma_")[1])
            centers=independent_centers(mode,ft)
            values=ft if role=="train" else fv
            assert np.allclose(response(values,centers,sigma).sum(axis=1),data[key],atol=1e-12,rtol=1e-12)
        c=np.load(output/f"cv/fold_{fold}_C_features.npz")
        centers=independent_centers(selection["B"]["center_mode"],ft)
        assert np.allclose(centers,c["centers"],atol=1e-14)
        assert np.allclose(response(ft,centers,selection["B"]["sigma"]).reshape(len(ft),256),c["gaussian_flat_train"])
        assert np.allclose(response(fv,centers,selection["B"]["sigma"]).reshape(len(fv),256),c["gaussian_flat_validation"])
    detail=pd.read_csv(output/"cv/all_fold_metrics.csv");predictions=pd.read_csv(output/"cv/all_fold_predictions.csv")
    assert len(detail)==153
    assert len(predictions)==51*3*412
    for _,row in detail.iterrows():
        for role,metric in [("train","train"),("validation","validation")]:
            p=predictions[(predictions.config_id==row.config_id)&(predictions.fold==row.fold)&(predictions.role==role)]
            expected=folds[(folds.fold==row.fold)&(folds.role==role)]
            assert set(p.sample_id)==set(expected.sample_id) and p.sample_id.is_unique
            assert p.true_class.tolist()==[label_by_id[int(i)] for i in p.sample_id]
            acc=accuracy_score(p.true_class,p.predicted_class)
            f1=f1_score(p.true_class,p.predicted_class,average="macro",labels=CLASSES,zero_division=0)
            assert np.isclose(acc,row[f"{metric}_accuracy"]) and np.isclose(f1,row[f"{metric}_macro_f1"])
    summary=pd.read_csv(output/"cv/all_configuration_summary.csv")
    for _,row in summary.iterrows():
        per_fold=detail[detail.config_id==row.config_id]
        for metric in ["train_accuracy","train_macro_f1","validation_accuracy","validation_macro_f1"]:
            assert np.isclose(row[f"mean_{metric}"],per_fold[metric].mean())
            assert np.isclose(row[f"std_{metric}"],per_fold[metric].std(ddof=1))
    for family in ["A","B","C"]:
        best=summary[(summary.family==family)&summary.all_folds_converged].sort_values(
            ["mean_validation_macro_f1","mean_validation_accuracy","std_validation_macro_f1","regularization_C","config_id"],
            ascending=[False,False,True,True,True]).iloc[0]
        assert best.config_id==selection[family]["config_id"]
    final=np.load(output/"final_features.npz",allow_pickle=True)
    assert np.array_equal(final["raw"],raw) and np.array_equal(final["labels"],labels)
    train=included.split.to_numpy()=="train";test=~train
    divisor=raw[train].max();scaled=raw/divisor;scaled[test]=np.clip(scaled[test],0,1)
    all_predictions=pd.read_csv(output/"all_final_predictions.csv")
    all_metrics=pd.read_csv(output/"final_default_and_tuned_ABC_metrics.csv")
    assert len(all_predictions)==5*588 and len(all_metrics)==5
    for _,row in all_metrics.iterrows():
        bundle=joblib.load(output/f"models/{row.method}.joblib")
        model=bundle["classifier"]
        if row.family=="A":features=scaled
        else:
            expected_centers=independent_centers(row.center_mode,scaled[train])
            assert np.allclose(expected_centers,bundle["centers"])
            r=response(scaled,expected_centers,float(row.sigma))
            features=r.sum(axis=1) if row.family=="B" else r.reshape(len(raw),256)
        assert np.allclose(features,final[row.method])
        scaler=model.named_steps["standard_scaler"]
        assert scaler.n_samples_seen_==412 and np.allclose(scaler.mean_,features[train].mean(axis=0))
        assert np.allclose(scaler.var_,features[train].var(axis=0))
        assert model.named_steps["logistic_regression"].C==row.regularization_C
        for role,mask in [("train",train),("test",test)]:
            pred=model.predict(final[row.method][mask])
            saved=all_predictions[(all_predictions.method==row.method)&(all_predictions.split==role)]
            assert np.array_equal(saved.sample_id,included.loc[mask,"sample_id"])
            assert np.array_equal(saved.true_class,labels[mask]) and np.array_equal(saved.predicted_class,pred)
            cm=confusion_matrix(labels[mask],pred,labels=CLASSES)
            saved_cm=pd.read_csv(output/f"confusion_{row.method}_{role}.csv",index_col=0).to_numpy()
            assert np.array_equal(cm,saved_cm) and np.isclose(np.trace(cm)/cm.sum(),row[f"{role}_accuracy"])
            assert np.isclose(f1_score(labels[mask],pred,average="macro",labels=CLASSES,zero_division=0),row[f"{role}_macro_f1"])
        if row.method in ["A_default","B_default"]:
            previous_name="A_linear" if row.family=="A" else "B_gaussian_sum"
            old=joblib.load(project/f"outputs/hmdb51_six_split1/model_{previous_name}.joblib")
            assert np.array_equal(model.predict(final[row.method]),old.predict(final[row.method]))
    assert np.allclose(final["C_no_sum_diagnostic"].reshape(-1,16,16).sum(axis=1),final["B_tuned"])
    assert original_results_unchanged(output)
    result={"status":"PASSED","independent_training_only_fold_feature_checks":3,
        "CV_fit_metrics_recomputed":153,"saved_final_models_reproducing_predictions":5,
        "all_final_train_test_confusion_matrices_recomputed":10,
        "default_AB_predictions_identical_to_original":True,
        "C_reshaped_and_summed_equals_tuned_B":True,"original_result_hashes_unchanged":True,
        "verification_seconds":time.perf_counter()-start}
    write_json(output/"post_run_verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
