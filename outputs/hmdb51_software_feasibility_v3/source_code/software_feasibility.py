#!/usr/bin/env python3
"""Order-invariant, 16-kernel software screening; reuse matching prior CV fits."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
from threadpoolctl import threadpool_limits

from run_experiment import CLASSES, gaussian_sum, read_manifest
from diagnose_hmdb51 import (
    SOLVER, RULE, cid, centers_from_training, fit_classifier, load_records,
    rank_configurations, scores, sha256,
)
from tune_gaussian import scale_fit_and_eval, write_json

SEED = 42
DEFAULT_SIGMA = .1
SIGMAS = [.05, .1, .2]
MODES = ["uniform_0_1", "uniform_p01_p99"]
REGULARIZATIONS = [.1, 1., 10.]
FEATURE_ATOL = 1e-10
FEATURE_RTOL = 1e-10
PROBABILITY_ATOL = 1e-10
PROBABILITY_RTOL = 1e-9
COLORS = {"A": "#64748b", "G_default": "#2878a3", "G_adjusted": "#dc7a36"}


def file_manifest(roots):
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for root in roots for p in sorted(root.rglob("*")) if p.is_file()]


def function_ast(path, name):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def run_signature(project,source,repetitions):
    """Prevent reuse of a completed run after data, folds or settings change."""
    signature={"cache_sha256":sha256(project/"outputs/hmdb51_six_split1/feature_cache.joblib"),
        "seed":SEED,"sigmas":SIGMAS,"center_modes":MODES,"regularization_C":REGULARIZATIONS,
        "solver":SOLVER,"selection_rule":RULE,"class_order":CLASSES,"permutations":repetitions,
        "feature_tolerance":[FEATURE_ATOL,FEATURE_RTOL],"probability_tolerance":[PROBABILITY_ATOL,PROBABILITY_RTOL],
        "numpy":np.__version__,"sklearn":sklearn.__version__}
    sources=[source/"experiment_config.json",source/"cv/fold_sample_manifest.csv",source/"cv/all_fold_metrics.csv",
        source/"cv/all_fold_predictions.csv",source/"cv/all_configuration_summary.csv"]
    sources += sorted((source/"cv").glob("fold_*_AB_features.npz"))
    sources += sorted((source/"models").glob("*.joblib"))
    sources += [next((project/"data/splits").rglob(f"{cls}_test_split1.txt")) for cls in CLASSES]
    signature["source_file_hashes"]={str(p.resolve()):sha256(p) for p in sources}
    signature["computation_AST"]={f"{file}:{name}":hashlib.sha256(function_ast(project/file,name).encode()).hexdigest()
        for file,name in [("run_experiment.py","gaussian_sum"),("tune_gaussian.py","scale_fit_and_eval"),
                          ("diagnose_hmdb51.py","centers_from_training"),("diagnose_hmdb51.py","fit_classifier")]}
    return signature


def audit_and_reuse(project, source, output):
    start = time.perf_counter()
    cache = project / "outputs/hmdb51_six_split1/feature_cache.joblib"
    old_config = json.loads((source / "experiment_config.json").read_text())
    checks = []
    def require(name, condition, detail):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        write_json(output / "checks.json", {"status": "PASSED" if all(c["passed"] for c in checks) else "FAILED", "checks": checks})
        if not condition:
            raise RuntimeError(f"Check failed: {name}: {detail}")

    require("identical_feature_cache", sha256(cache) == old_config["cache_sha256"], "SHA-256 agrees with the previously audited CV run")
    records, included, raw = load_records(cache)
    train = included.split.to_numpy() == "train"
    require("feature_shape_and_finite_values", raw.shape == (588,16) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
            f"train={train.sum()}, test={(~train).sum()}, all-zero samples={int(np.all(raw == 0,axis=1).sum())}")
    official = read_manifest(project / "data/videos", project / "data/splits")
    matched = records.merge(official[official.split.isin(["train","test"])], on="filename", suffixes=("_cache","_official"), validate="one_to_one")
    require("official_labels_and_splits", len(matched) == len(records) == 600 and
            np.array_equal(matched.class_cache,matched.class_official) and
            np.array_equal(matched.split_cache,matched.split_official) and
            np.array_equal(matched.official_flag_cache,matched.official_flag_official), "All candidate filenames checked against the local official split-1 files")
    old_manifest = pd.read_csv(source / "sample_manifest.csv")
    require("sample_row_identity", np.array_equal(old_manifest.sample_id,included.sample_id) and
            np.array_equal(old_manifest.filename,included.filename) and np.array_equal(old_manifest["class"],included["class"]), "Sample IDs, filenames, labels and row order agree")
    digests = [sha256(p) for p in included.path]
    require("video_content_unchanged", digests == old_manifest.video_sha256.tolist(), "All 588 local video SHA-256 values agree with the prior audit; no frames reprocessed")
    require("no_training_test_overlap", not set(included.loc[train,"filename"]) & set(included.loc[~train,"filename"]) and
            not {str(Path(p).resolve()) for p in included.loc[train,"path"]} & {str(Path(p).resolve()) for p in included.loc[~train,"path"]} and
            not set(np.asarray(digests)[train]) & set(np.asarray(digests)[~train]), "No shared filename, resolved path or identical file content across official train/test")
    old_runtime = json.loads((source / "runtime.json").read_text())
    require("same_classifier_and_environment", old_config["solver"] == SOLVER and old_config["seed"] == SEED and
            old_config["folds"] == 3 and old_runtime["numpy"] == np.__version__ and old_runtime["sklearn"] == sklearn.__version__,
            "Same lbfgs/L2/logistic regression, tolerance, max_iter, seed, NumPy and scikit-learn versions")
    comparisons = [("diagnose_hmdb51.py",n) for n in ["gaussian_response","features","centers_from_training","fit_classifier"]]
    comparisons += [("run_experiment.py","gaussian_sum"),("tune_gaussian.py","scale_fit_and_eval")]
    require("same_preprocessing_and_classifier_code", all(function_ast(project / f,n) == function_ast(source / "source_code" / f,n) for f,n in comparisons),
            "AST identities checked for Gaussian response/sum, center construction, training scaling and classifier fitting")
    old_checks = json.loads((source / "implementation_checks.json").read_text())
    require("previous_audit_applicable", old_checks["status"] == "PASSED" and old_checks["cache_sha256"] == sha256(cache),
            "Existing independent video/float-difference checks remain applicable to unchanged videos, cache and code")

    cvdir = output / "cv"; cvdir.mkdir(exist_ok=True)
    folds = pd.read_csv(source / "cv/fold_sample_manifest.csv")
    validation = folds[folds.role == "validation"]
    require("identical_three_fold_partition", len(folds) == 1236 and len(validation) == 412 and
            validation.sample_id.nunique() == 412 and set(folds.official_split) == {"train"},
            "Original seed-42 stratified 3-fold manifest reused verbatim; verified source IDs are unavailable, so no guessed groups")
    shutil.copy2(source / "cv/fold_sample_manifest.csv", cvdir / "fold_sample_manifest.csv")
    shutil.copy2(source / "cv/fold_class_counts.csv", cvdir / "fold_class_counts.csv")
    by_id = {int(row.sample_id):np.asarray(row.feature,dtype=np.float64) for _,row in included.iterrows()}
    label_by_id = dict(zip(included.sample_id,included["class"]))
    centers_log, scaling_log = [], []
    for fold in range(1,4):
        data = np.load(source / f"cv/fold_{fold}_AB_features.npz")
        tid,vid = data["training_sample_ids"],data["validation_sample_ids"]
        expected_t = folds[(folds.fold==fold)&(folds.role=="train")].sample_id.to_numpy()
        expected_v = folds[(folds.fold==fold)&(folds.role=="validation")].sample_id.to_numpy()
        require(f"fold_{fold}_train_validation_alignment", np.array_equal(tid,expected_t) and np.array_equal(vid,expected_v) and
                not set(tid)&set(vid) and set(tid)|set(vid) == set(included.loc[train,"sample_id"]), "Saved arrays and manifests use the same training-only sample IDs")
        rt,rv = np.array([by_id[int(i)] for i in tid]),np.array([by_id[int(i)] for i in vid])
        ft,fv,scaling = scale_fit_and_eval(rt,rv)
        require(f"fold_{fold}_training_only_scaling", np.array_equal(rt,data["raw_train"]) and np.array_equal(rv,data["raw_validation"]) and
                np.array_equal(ft,data["scaled_train"]) and np.array_equal(fv,data["scaled_validation"]), "Global training maximum recomputed; validation clipping reproduced")
        saved = {"raw_train":rt,"raw_validation":rv,"scaled_train":ft,"scaled_validation":fv,
                 "training_sample_ids":tid,"validation_sample_ids":vid}
        for mode in MODES:
            centers = centers_from_training(mode,ft)
            centers_log.append({"fold":fold,"center_mode":mode,"centers":centers.tolist(),"unique_centers":len(np.unique(centers)),
                                "duplicate_center_policy":"retain every center; no jitter or dimensionality changes"})
            for sigma in SIGMAS:
                for role,x in [("train",ft),("validation",fv)]:
                    key = f"B__{mode}__sigma_{sigma:g}_{role}"
                    independent = np.exp(-np.square(x[:,:,None]-centers[None,None,:])/(2*sigma*sigma)).sum(axis=1)
                    require(f"fold_{fold}_{mode}_{sigma:g}_{role}_Gaussian", np.allclose(independent,data[key],rtol=1e-12,atol=1e-12),
                            "Reused summed features agree with the independent formula and training-only centers")
                    saved[key] = data[key]
        scaling_log.append({"fold":fold,**scaling})
        np.savez_compressed(cvdir / f"fold_{fold}_reused_features.npz",**saved)
    write_json(cvdir / "fold_scaling.json",scaling_log)
    write_json(cvdir / "fold_centers.json",centers_log)
    configs = {cid("A","none",0.,c) for c in REGULARIZATIONS}
    configs |= {cid("B",m,s,c) for m in MODES for s in SIGMAS for c in REGULARIZATIONS}
    detail = pd.read_csv(source / "cv/all_fold_metrics.csv")
    detail = detail[detail.config_id.isin(configs)].copy()
    predictions = pd.read_csv(source / "cv/all_fold_predictions.csv")
    predictions = predictions[predictions.config_id.isin(configs)].copy()
    require("complete_matching_CV_cache", len(detail) == 63 and detail.config_id.nunique() == 21 and detail.converged.all(),
            "All 21 requested classifier/Gaussian configurations have all 3 completed, converged folds; no CV retraining")
    for _,row in detail.iterrows():
        for role,metric in [("train","train"),("validation","validation")]:
            pred = predictions[(predictions.config_id==row.config_id)&(predictions.fold==row.fold)&(predictions.role==role)]
            expected_ids = folds[(folds.fold==row.fold)&(folds.role==role)].sample_id
            sc = scores(pred.true_class,pred.predicted_class)
            require(f"CV_{row.config_id}_{row.fold}_{role}", len(pred)==len(expected_ids) and set(pred.sample_id)==set(expected_ids) and
                    pred.true_class.tolist()==[label_by_id[int(i)] for i in pred.sample_id] and
                    np.isclose(sc["accuracy"],row[f"{metric}_accuracy"]) and np.isclose(sc["macro_f1"],row[f"{metric}_macro_f1"]),
                    "Saved predictions/labels and recalculated metrics match; prior fit times are historical")
    summary = pd.read_csv(source / "cv/all_configuration_summary.csv")
    summary = summary[summary.config_id.isin(configs)].copy()
    for _,row in summary.iterrows():
        matching = detail[detail.config_id==row.config_id]
        for metric in ["train_accuracy","train_macro_f1","validation_accuracy","validation_macro_f1"]:
            require(f"summary_{row.config_id}_{metric}",np.isclose(row[f"mean_{metric}"],matching[metric].mean()) and
                    np.isclose(row[f"std_{metric}"],matching[metric].std(ddof=1)),"Mean and sample SD recomputed from identical folds")
    detail["reused_from"] = str(source / "cv/all_fold_metrics.csv")
    detail["fit_seconds_this_run"] = 0.
    detail.to_csv(cvdir / "fold_metrics.csv",index=False)
    predictions.to_csv(cvdir / "fold_predictions.csv",index=False)
    summary.to_csv(cvdir / "configuration_summary.csv",index=False)
    included.drop(columns=["feature"]).to_csv(output / "sample_manifest.csv",index=False)
    records[~records.included].to_csv(output / "exclusions.csv",index=False)
    np.savez_compressed(output / "raw_features.npz",raw=raw,labels=included["class"].to_numpy(),
                        sample_ids=included.sample_id.to_numpy(),official_splits=included.split.to_numpy())
    check_record = {"status":"PASSED","checks":checks,"cache_sha256":sha256(cache),"CV_fits_reused":63,"CV_fits_new":0,
        "reliable_source_IDs_available":False,"limitation":"Stratified folds can share source/scene content; filename heuristics are not treated as verified groups.",
        "standard_scaler_CV_evidence":"Identical authenticated fitting code asserts fit-sample count/mean/variance against fold training features. Fold models were not saved, so scaler objects are not claimed to have been re-read.",
        "runtime_seconds":time.perf_counter()-start}
    write_json(output / "checks.json",check_record)
    return included,raw,summary,check_record


def lock_selection(summary, output):
    best_a = rank_configurations(summary,"A").iloc[0]
    default_options = summary[(summary.family=="B")&(summary.center_mode=="uniform_0_1")&(summary.sigma==DEFAULT_SIGMA)]
    best_default = rank_configurations(default_options,"B").iloc[0]
    best_g = rank_configurations(summary,"B").iloc[0]
    chosen = {}
    for method,row in [("A",best_a),("G_default",best_default),("G_adjusted",best_g)]:
        chosen[method] = {"config_id":row.config_id,"center_mode":row.center_mode,"sigma":float(row.sigma),
                          "regularization_C":float(row.regularization_C)}
    selection = {"models":chosen,"selection_rule":RULE,"selected_for_confusion_display":"G_adjusted",
        "selection_uses_test":False,"prior_test_already_observed":True,
        "freeze_time_local":time.strftime("%Y-%m-%d %H:%M:%S %z")}
    write_json(output / "parameters_frozen_before_test.json",selection)
    return selection


def final_models(source, included, raw, selection, output):
    train = included.split.to_numpy()=="train"; labels=included["class"].to_numpy()
    ft,fe,scaling = scale_fit_and_eval(raw[train],raw[~train])
    x = np.empty_like(raw); x[train],x[~train] = ft,fe
    write_json(output / "final_training_only_scale.json",scaling)
    destination=output/"models";destination.mkdir(exist_ok=True)
    bundles,logs = {},[]
    for method,config in selection["models"].items():
        centers=None if method=="A" else centers_from_training(config["center_mode"],ft)
        feature=x if method=="A" else gaussian_sum(x,config["sigma"],centers)
        assert feature.shape==(len(raw),16) and np.isfinite(feature).all()
        model=None;reused_from=None;fit_seconds=0.;warnings_saved=[]
        for path in [destination/f"{method}.joblib", *sorted((source/"models").glob("*.joblib"))]:
            if not path.exists():continue
            candidate=joblib.load(path)
            if candidate["family"] != ("A" if method=="A" else "B"):continue
            if candidate["regularization_C"]!=config["regularization_C"] or candidate["sigma"]!=config["sigma"]:continue
            if centers is not None and not np.array_equal(centers,candidate["centers"]):continue
            if candidate["input_scale"]!=scaling:continue
            if candidate["classifier"].named_steps["logistic_regression"].get_params()["tol"]!=SOLVER["tol"]:continue
            model=candidate["classifier"];reused_from=str(path);break
        if model is None:
            model,_,_,stats=fit_classifier(feature[train],labels[train],feature[~train],labels[~train],config["regularization_C"])
            if not stats["converged"]:raise RuntimeError(f"Final model did not converge: {method}")
            fit_seconds=stats["classifier_training_seconds"];warnings_saved=json.loads(stats["warnings"])
        scaler=model.named_steps["standard_scaler"]
        assert scaler.n_samples_seen_==int(train.sum())
        assert np.allclose(scaler.mean_,feature[train].mean(axis=0)) and np.allclose(scaler.var_,feature[train].var(axis=0))
        classifier=model.named_steps["logistic_regression"]
        assert classifier.n_iter_.max()<classifier.max_iter
        bundle={"classifier":model,"input_scale":scaling,"family":"A" if method=="A" else "B",
            "centers":centers,"sigma":config["sigma"],"regularization_C":config["regularization_C"],
            "center_mode":config["center_mode"],"class_order":CLASSES,"output_dimensions":16,
            "input_coordinate":"training-global-max normalized scalar, dimensionless",
            "peak":1.,"model_definition":"y[n,j]=sum_i exp(-(x[n,i]-mu[j])**2/(2*sigma**2))"}
        joblib.dump(bundle,destination/f"{method}.joblib");bundles[method]=bundle
        logs.append({"method":method,"reused":reused_from is not None,"reused_from":reused_from,
            "classifier_training_seconds_this_run":fit_seconds,"converged":True,"n_iter":int(classifier.n_iter_.max()),"warnings":warnings_saved})
    write_json(output / "final_model_fit_provenance.json",logs)
    return x,bundles,logs


def check_permutations(x, included, bundles, repetitions, output, source):
    rng=np.random.default_rng(SEED)
    permutations=np.argsort(rng.random((repetitions,len(x),16)),axis=2).astype(np.uint8)
    np.savez_compressed(output/"permutation_indices.npz",indices=permutations,sample_ids=included.sample_id.to_numpy())
    reports={};per_trial=[];examples={}
    for method,bundle in bundles.items():
        if method=="A":continue
        centers,sigma=bundle["centers"],bundle["sigma"]
        base=gaussian_sum(x,sigma,centers);model=bundle["classifier"]
        probabilities=model.predict_proba(base);prediction=model.predict(base)
        feature_diffs=[];probability_diffs=[];changed=[]
        feature_pass=True;probability_pass=True
        for trial,perm in enumerate(permutations):
            permuted=np.take_along_axis(x,perm,axis=1)
            mapped=gaussian_sum(permuted,sigma,centers)
            p=model.predict_proba(mapped);yp=model.predict(mapped)
            fd=np.max(np.abs(mapped-base),axis=1);pdiff=np.max(np.abs(p-probabilities),axis=1);diff=yp!=prediction
            feature_diffs.append(fd);probability_diffs.append(pdiff);changed.append(diff)
            feature_pass=feature_pass and bool(np.allclose(base,mapped,atol=FEATURE_ATOL,rtol=FEATURE_RTOL))
            probability_pass=probability_pass and bool(np.allclose(probabilities,p,atol=PROBABILITY_ATOL,rtol=PROBABILITY_RTOL))
            per_trial.append({"method":method,"trial":trial+1,"sample_count":len(x),"max_absolute_feature_difference":float(fd.max()),
                "max_absolute_probability_difference":float(pdiff.max()),"changed_classifications":int(diff.sum())})
            if trial==0:examples[method]={"base":base,"permuted":mapped,"base_probabilities":probabilities,"permuted_probabilities":p,"base_predictions":prediction,"permuted_predictions":yp}
        # Exact multiplicity tests, separate from the real-video evaluation data.
        value=float(x[included.split.to_numpy()=="train"][0,0])
        repeat=np.full((1,16),value)
        expected=np.array([16*math.exp(-((value-float(mu))**2)/(2*sigma*sigma)) for mu in centers])
        repeat_result=gaussian_sum(repeat,sigma,centers)[0]
        repeated_pass=bool(np.allclose(expected,repeat_result,atol=1e-12,rtol=1e-12))
        literal=np.array([[sum(math.exp(-((float(v)-float(mu))**2)/(2*sigma*sigma)) for v in sample) for mu in centers] for sample in x[:8]])
        literal_error=float(np.max(np.abs(literal-gaussian_sum(x[:8],sigma,centers))))
        result={"status":"PASSED" if feature_pass and probability_pass and np.sum(changed)==0 and repeated_pass and literal_error<1e-12 else "FAILED",
            "permutations_per_sample":repetitions,"sample_count":len(x),"total_sample_permutation_checks":repetitions*len(x),
            "feature_atol":FEATURE_ATOL,"feature_rtol":FEATURE_RTOL,"probability_atol":PROBABILITY_ATOL,"probability_rtol":PROBABILITY_RTOL,
            "max_absolute_feature_difference":float(np.max(feature_diffs)),"max_absolute_probability_difference":float(np.max(probability_diffs)),
            "changed_classifications":int(np.sum(changed)),"independent_literal_formula_max_error":literal_error,
            "duplicate_values_test":{"input_value":value,"multiplicity":16,"expected_output":expected.tolist(),"actual_output":repeat_result.tolist(),"passed":repeated_pass},
            "features_within_tolerance":feature_pass,"probabilities_within_tolerance":probability_pass}
        reports[method]=result
        np.savez_compressed(output/f"permutation_{method}_details.npz",feature_max_difference_by_trial_and_sample=np.array(feature_diffs),
            probability_max_difference_by_trial_and_sample=np.array(probability_diffs),changed_classifications=np.array(changed),**examples[method])
    pd.DataFrame(per_trial).to_csv(output/"permutation_trials.csv",index=False)
    write_json(output/"order_independence_checks.json",reports)
    if not all(r["status"]=="PASSED" for r in reports.values()):raise RuntimeError("Order-independence checks failed; performance plotting is blocked")
    # Old C is assessed using its saved model; it is not retrained or ranked here.
    legacy=joblib.load(source/"models/C_no_sum_diagnostic.joblib")
    def old_c(values):
        return np.exp(-np.square(values[:,:,None]-legacy["centers"][None,None,:])/(2*legacy["sigma"]**2)).reshape(len(values),256)
    b,p=old_c(x),old_c(np.take_along_axis(x,permutations[0],axis=1))
    old_c_review={"target_compatible":False,"output_dimensions":256,
        "reason":"Retaining input positions permutes feature blocks; it is a valid previous software diagnostic but does not implement the current order-invariant output.",
        "single_permutation_max_feature_difference":float(np.max(np.abs(b-p))),
        "single_permutation_changed_predictions":int(np.sum(legacy["classifier"].predict(b)!=legacy["classifier"].predict(p))),
        "classifier_fits_this_run":0,"included_in_target_performance_comparison":False}
    write_json(output/"legacy_B_C_model_review.json",{"old_B":{"target_compatible":True,"output_dimensions":16,"definition_unchanged":True},"old_C":old_c_review})
    return reports


def evaluate(included, x, bundles, summary, selection, checks, output):
    rows=[];predictions=[];class_metrics=[];arrays={"scaled_inputs":x,"labels":included["class"].to_numpy(),"sample_ids":included.sample_id.to_numpy(),"official_splits":included.split.to_numpy()}
    train=included.split.to_numpy()=="train";labels=included["class"].to_numpy()
    parameters=[]
    for method,bundle in bundles.items():
        model=bundle["classifier"];z=x if method=="A" else gaussian_sum(x,bundle["sigma"],bundle["centers"])
        start=time.perf_counter();yp=model.predict(z);prob=model.predict_proba(z);prediction_seconds=time.perf_counter()-start
        arrays[method]=z
        ts,es=scores(labels[train],yp[train]),scores(labels[~train],yp[~train])
        cv=summary[summary.config_id==selection["models"][method]["config_id"]].iloc[0]
        centers=bundle["centers"]
        parameters.append({"method":method,"output_dimensions":16,"center_mode":bundle["center_mode"],"sigma":bundle["sigma"],
            "centers":None if centers is None else centers.tolist(),"Gaussian_peak":1.,"classifier":{**SOLVER,"C":bundle["regularization_C"],"penalty":"l2"},
            "coordinate_units":"dimensionless normalized input; no pulse-time or current calibration assumed"})
        rows.append({"method":method,"output_dimensions":16,"train_samples":int(train.sum()),"test_samples":int((~train).sum()),
            "train_accuracy":ts["accuracy"],"train_macro_f1":ts["macro_f1"],"validation_accuracy_mean":float(cv.mean_validation_accuracy),
            "validation_accuracy_std":float(cv.std_validation_accuracy),"validation_macro_f1_mean":float(cv.mean_validation_macro_f1),
            "validation_macro_f1_std":float(cv.std_validation_macro_f1),"test_accuracy":es["accuracy"],"test_macro_f1":es["macro_f1"],
            "center_mode":bundle["center_mode"],"sigma":bundle["sigma"],"classifier_C":bundle["regularization_C"],
            "actual_centers":None if centers is None else json.dumps(centers.tolist()),"order_independence":"not_applicable" if method=="A" else checks[method]["status"],
            "prediction_seconds_all_samples_this_run":prediction_seconds,"CV_reused":True})
        for i,row in included.iterrows():
            item={"method":method,"sample_id":int(row.sample_id),"filename":row.filename,"split":row.split,"true_class":labels[i],"predicted_class":yp[i],"correct":bool(yp[i]==labels[i])}
            item.update({f"prob_{cls}":float(prob[i,list(model.classes_).index(cls)]) for cls in CLASSES});predictions.append(item)
        for role,mask in [("train",train),("test",~train)]:
            cm=confusion_matrix_local(labels[mask],yp[mask])
            pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(output/f"confusion_{method}_{role}.csv")
            for i,cls in enumerate(CLASSES):
                precision=float(cm[i,i]/cm[:,i].sum()) if cm[:,i].sum() else 0.
                recall=float(cm[i,i]/cm[i,:].sum())
                class_metrics.append({"method":method,"split":role,"class":cls,"true_count":int(cm[i,:].sum()),
                    "predicted_count":int(cm[:,i].sum()),"true_positives":int(cm[i,i]),"precision":precision,"recall":recall,
                    "f1":2*precision*recall/(precision+recall) if precision+recall else 0.})
    result=pd.DataFrame(rows);result.to_csv(output/"results_table.csv",index=False)
    write_json(output/"complete_metrics.json",result.astype(object).where(pd.notna(result),None).to_dict("records"))
    write_json(output/"actual_model_parameters.json",parameters)
    pd.DataFrame(predictions).to_csv(output/"sample_predictions.csv",index=False)
    pd.DataFrame(class_metrics).to_csv(output/"per_class_metrics.csv",index=False)
    np.savez_compressed(output/"final_features.npz",**arrays)
    return result


def confusion_matrix_local(labels,prediction):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(labels,prediction,labels=CLASSES)


def draw_panel_one(fig,spec,example,scaled,output_values):
    inner=spec.subgridspec(2,2,width_ratios=[1.05,1.65],wspace=.4,hspace=.65)
    ax=fig.add_subplot(inner[:,0]);ax.axis("off")
    if example["image"] is not None:ax.imshow(example["image"])
    else:ax.text(.5,.5,"Local video/frame unavailable",ha="center",va="center",transform=ax.transAxes)
    ax.set_title("(1) Local training example\n"+example["class"]+f"; frame {example['frame_index']}, {example['time_seconds']:.3f}s",fontsize=10)
    ax.text(.5,-.09,"\n".join(example["filename"][i:i+27] for i in range(0,len(example["filename"]),27)),ha="center",va="top",transform=ax.transAxes,fontsize=7)
    ax=fig.add_subplot(inner[0,1]);ax.plot(range(1,17),scaled,"o-",color=COLORS["A"],ms=4)
    ax.set(title="16-D input",xlabel="Input position i (1-16)",ylabel="Scaled frame change",xticks=[1,4,8,12,16]);ax.grid(alpha=.2)
    ax=fig.add_subplot(inner[1,1]);ax.bar(range(1,17),output_values,color=COLORS["G_adjusted"])
    ax.set(title="16-D Gaussian-sum output",xlabel="Kernel j (1-16), one sum per kernel",ylabel="Summed response",xticks=[1,4,8,12,16]);ax.grid(axis="y",alpha=.2)


def draw_panel_two(ax,base,permuted,check):
    ax.plot(range(1,17),base,"o-",color=COLORS["G_default"],lw=2,ms=6,label="Original order")
    ax.plot(range(1,17),permuted,"x--",color=COLORS["G_adjusted"],lw=1.3,ms=7,label="Permuted order")
    ax.set(title="(2) Same values, permuted input positions",xlabel="Kernel j (1-16)",ylabel="Summed Gaussian response",xticks=range(1,17))
    ax.legend(loc="upper right");ax.grid(alpha=.2)
    ax.text(.02,.03,f"{check['permutations_per_sample']} permutations x {check['sample_count']} samples\nMax feature difference: {check['max_absolute_feature_difference']:.2e}\nMax probability difference: {check['max_absolute_probability_difference']:.2e}\nPrediction changes: {check['changed_classifications']}",transform=ax.transAxes,fontsize=9,bbox={"facecolor":"white","alpha":.85,"edgecolor":"#cbd5e1"})


def draw_panel_three(ax,metrics):
    x=np.arange(3);validation=metrics.validation_accuracy_mean.to_numpy()*100;test=metrics.test_accuracy.to_numpy()*100;sd=metrics.validation_accuracy_std.to_numpy()*100
    a=ax.bar(x-.18,validation,width=.36,yerr=sd,capsize=4,color=COLORS["G_default"],label="Validation mean +/- fold SD")
    b=ax.bar(x+.18,test,width=.36,color=COLORS["G_adjusted"],label="Official test (one evaluation)")
    for bar,value,error in [*(zip(a,validation,sd)),*(zip(b,test,np.zeros(3)))]:
        ax.text(bar.get_x()+bar.get_width()/2,value+error+1.5,f"{value:.2f}%",ha="center",fontsize=9)
    ax.set(ylim=(0,100),yticks=np.arange(0,101,20),xticks=x,xticklabels=["A","G default","G adjusted"],ylabel="Accuracy (%)",title="(3) Validation and test accuracy")
    ax.legend(loc="upper right",fontsize=8);ax.grid(axis="y",alpha=.2)


def draw_panel_four(fig,ax,cm):
    image=ax.imshow(cm,cmap="Blues",vmin=0)
    ax.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,
        xlabel="Predicted class",ylabel="True class",title="(4) CV-selected Gaussian: official test")
    for i in range(6):
        for j in range(6):ax.text(j,i,int(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(image,ax=ax,fraction=.045,pad=.03,label="Video count")


def plots(project,output,metrics,selection):
    records,included,_=load_records(project/"outputs/hmdb51_six_split1/feature_cache.joblib")
    row=included[(included.split=="train")&(included["class"]==CLASSES[0])].sort_values("filename").iloc[0]
    index=int(row.name);frame_index=int(row.sampled_indices[8]);frame=None
    path=project/row.path
    if path.is_file():
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,read=cap.read();cap.release()
        if ok:frame=cv2.cvtColor(read,cv2.COLOR_BGR2RGB)
    example={"filename":row.filename,"sample_id":int(row.sample_id),"class":row["class"],"frame_index":frame_index,
             "time_seconds":frame_index/row.fps,"image":frame,"local_frame_available":frame is not None}
    if frame is not None:plt.imsave(output/"example_frame.png",frame)
    write_json(output/"example_training_sample.json",{k:v for k,v in example.items() if k!="image"})
    data=np.load(output/"final_features.npz",allow_pickle=True)
    perm=np.load(output/"permutation_G_adjusted_details.npz",allow_pickle=True)
    checks=json.loads((output/"order_independence_checks.json").read_text())
    cm=pd.read_csv(output/"confusion_G_adjusted_test.csv",index_col=0).to_numpy()
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.titlesize":11,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8})
    fig=plt.figure(figsize=(17,11),dpi=240)
    outer=fig.add_gridspec(2,2,left=.055,right=.97,bottom=.065,top=.88,wspace=.25,hspace=.46)
    draw_panel_one(fig,outer[0,0],example,data["scaled_inputs"][index],data["G_adjusted"][index])
    draw_panel_two(fig.add_subplot(outer[0,1]),perm["base"][index],perm["permuted"][index],checks["G_adjusted"])
    draw_panel_three(fig.add_subplot(outer[1,0]),metrics)
    draw_panel_four(fig,fig.add_subplot(outer[1,1]),cm)
    g=selection["models"]["G_adjusted"]
    fig.suptitle("HMDB51 six classes | exploratory software screening",fontsize=19,y=.975)
    fig.text(.5,.933,f"Target: 16 scalar Gaussian sums | CV-selected: {g['center_mode']}, sigma={g['sigma']:g}, classifier C={g['regularization_C']:g}",ha="center",fontsize=11)
    fig.savefig(output/"four_panel_summary.png",bbox_inches="tight");plt.close(fig)
    fig=plt.figure(figsize=(10,5.8),dpi=220);spec=fig.add_gridspec(1,1,left=.05,right=.97,bottom=.16,top=.9)
    draw_panel_one(fig,spec[0,0],example,data["scaled_inputs"][index],data["G_adjusted"][index]);fig.savefig(output/"panel_1_input_output.png",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),dpi=220);draw_panel_two(ax,perm["base"][index],perm["permuted"][index],checks["G_adjusted"]);fig.tight_layout();fig.savefig(output/"panel_2_order_independence.png",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),dpi=220);draw_panel_three(ax,metrics);fig.tight_layout();fig.savefig(output/"panel_3_performance.png",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5.8),dpi=220);draw_panel_four(fig,ax,cm);fig.tight_layout();fig.savefig(output/"panel_4_confusion.png",bbox_inches="tight");plt.close(fig)
    headers=["Method","Output\ndim.","Train\naccuracy","Validation accuracy\nmean +/- SD","Validation\nMacro-F1","Test\naccuracy","Test\nMacro-F1","Actual parameters","Order\nindependence"]
    rows=[]
    for _,r in metrics.iterrows():
        params=f"Raw features; C={r.classifier_C:g}" if r.method=="A" else f"mu=linspace({json.loads(r.actual_centers)[0]:.6f}, {json.loads(r.actual_centers)[-1]:.6f}, 16)\nsigma={r.sigma:g}; peak=1; C={r.classifier_C:g}"
        rows.append([r.method,"16",f"{r.train_accuracy:.2%}",f"{r.validation_accuracy_mean:.2%} +/- {r.validation_accuracy_std*100:.2f} pp",f"{r.validation_macro_f1_mean:.4f}",f"{r.test_accuracy:.2%}",f"{r.test_macro_f1:.4f}",params,"N/A" if r.method=="A" else "PASS"])
    fig,ax=plt.subplots(figsize=(20,4.2),dpi=240);ax.axis("off")
    table=ax.table(cellText=rows,colLabels=headers,cellLoc="center",loc="center",colWidths=[.085,.045,.065,.14,.075,.065,.075,.28,.075])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,3)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#d5dce5")
        if r==0:cell.set_facecolor("#23364d");cell.set_text_props(color="white",weight="bold")
        else:cell.set_facecolor("#edf4fa" if r%2 else "#ffffff")
    ax.set_title("HMDB51 split 1: 412 train / 176 test | same 3 validation folds",fontsize=15,pad=20)
    fig.text(.5,.12,"Validation SD is between-fold sample SD. Primary selection: validation Macro-F1; accuracy breaks ties. Test scores do not select models.",ha="center",fontsize=10)
    fig.savefig(output/"results_table.png",bbox_inches="tight");plt.close(fig)


def write_documents(output,metrics,selection,runtime):
    lookup=metrics.set_index("method");a=lookup.loc["A"];d=lookup.loc["G_default"];g=lookup.loc["G_adjusted"]
    config=selection["models"]["G_adjusted"]
    # The judgment is scoped to these measured six-class software results.
    permutations=json.loads((output/"order_independence_checks.json").read_text())["G_adjusted"]["permutations_per_sample"]
    brief=(f"每核对16个输入计算高斯响应并求和，每核输出1值，共16维。旧B符合；旧C保留位置，不符合目标。{permutations}次排列检查通过，预测未变。"
        f"G调整测试准确率/Macro-F1为{g.test_accuracy:.2%}/{g.test_macro_f1:.4f}，A为{a.test_accuracy:.2%}/{a.test_macro_f1:.4f}，G默认为{d.test_accuracy:.2%}/{d.test_macro_f1:.4f}。"
        "调参未提高准确率，相比A仅小幅改善。"
        "当前16维输入与所测高斯方案效果不足，暂不支持直接进入硬件实验。先检查输入和错误来源；保留软件候选：训练P1～P99均匀16中心、"
        f"σ={config['sigma']:g}、StandardScaler+逻辑回归C={config['regularization_C']:g}。差异包含高斯映射与求和；本轮仅为探索性筛选。")
    if len(brief)>300:raise RuntimeError(f"Brief exceeds 300 characters: {len(brief)}")
    (output/"简短中文说明.md").write_text(brief+"\n",encoding="utf-8")
    captions=["该训练示例的16个画面变化量经每核求和后仍为16维，视频示例帧取自本地原片。",
        "相同16个数重排后，各核合计值及分类预测在记录的浮点容差内保持一致。",
        "误差线仅表示3折验证准确率的折间标准差，测试柱只显示实际单次评价值。",
        "该Gaussian配置先由训练交叉验证选定，混淆矩阵展示其正式测试错误分布，未按测试成绩挑选。"]
    files=["panel_1_input_output.png","panel_2_order_independence.png","panel_3_performance.png","panel_4_confusion.png"]
    (output/"四联图中文图注.md").write_text("# 四联图及子图说明\n\n"+"\n\n".join(f"{i+1}. {caption}\n\n![Panel {i+1}]({file})" for i,(caption,file) in enumerate(zip(captions,files))),encoding="utf-8")
    parameters=json.loads((output/"actual_model_parameters.json").read_text())
    actual=next(p for p in parameters if p["method"]=="G_adjusted")
    cm=pd.read_csv(output/"confusion_G_adjusted_test.csv",index_col=0)
    pairs=sorted([(int(cm.iloc[i,j]),CLASSES[i],CLASSES[j]) for i in range(6) for j in range(6) if i!=j],reverse=True)[:5]
    rows="\n".join(f"| {r.method} | {r.train_accuracy:.2%} | {r.validation_accuracy_mean:.2%} ± {r.validation_accuracy_std:.2%} | {r.validation_macro_f1_mean:.4f} | {r.test_accuracy:.2%} | {r.test_macro_f1:.4f} |" for _,r in metrics.iterrows())
    notes=f"""# 探索性软件筛选记录

{brief}

## 可核查的结果

| 方案 | 训练准确率 | 验证准确率均值±SD | 验证Macro-F1均值 | 测试准确率 | 测试Macro-F1 |
|---|---:|---:|---:|---:|---:|
{rows}

G调整相对A：测试Accuracy {(g.test_accuracy-a.test_accuracy)*100:+.2f}个百分点、Macro-F1 {g.test_macro_f1-a.test_macro_f1:+.4f}；相对G默认：Accuracy {(g.test_accuracy-d.test_accuracy)*100:+.2f}个百分点、Macro-F1 {g.test_macro_f1-d.test_macro_f1:+.4f}。验证均值及折间波动见表，不将一次测试差异称为稳定优势。

## 参数与判断依据

软件候选采用 `{actual['center_mode']}`，完整训练拟合后中心为 `linspace({actual['centers'][0]:.9f}, {actual['centers'][-1]:.9f}, 16)`，σ={actual['sigma']:g}、峰值=1、每核全部16输入求和、16维输出；StandardScaler+多分类逻辑回归C={actual['classifier']['C']:g}，lbfgs、L2、max_iter=5000、tol=1e-6、seed=42。单位始终是缩放后的输入坐标，不赋予未知脉冲时间或电流单位。全部实际中心在 actual_model_parameters.json，模型在 models/。这是保留以供软件复核的候选，并非进入硬件的推荐配置。

选定G在176个测试视频中正确 {int(round(g.test_accuracy*176))} 个、错误 {176-int(round(g.test_accuracy*176))} 个；较多错误为 {'；'.join(f'{a}→{b}：{n}个' for n,a,b in pairs)}。这种识别水平和类别混淆，不足以支持当前六类动作识别目标直接进入硬件。该判断没有设定事先的准确率门槛，也不只依据是否高于随机猜测。它仅针对当前16维输入和本轮所测Gaussian配置，不代表HMDB51永久不可用或器件不可行。

特别是walk类为 {int(cm.loc['walk','walk'])}/{int(cm.loc['walk'].sum())} 正确，不能用总体的小幅提升掩盖单类失效。各类的精确率、召回率、F1和预测数量见 per_class_metrics.csv。

推荐下一步仍在软件中检查当前平均帧差对人物动作、背景活动与镜头变化的区分能力，以及窗口是否覆盖动作。本轮没有改变输入、视频预处理或器件模型。A与G的差别同时包含Gaussian映射与求和，不能全部归因于Gaussian非线性。旧C的256维位置展开不满足本轮目标的顺序无关要求，其旧成绩没有纳入此处的目标可行性比较。

## 选择与复用

本轮候选为2种中心×3种宽度×3种分类器C（18组G）及3组A，共21组、63个折内拟合；它们的样本、3折清单、原始缓存、缩放、中心公式、分类器设置、代码AST及依赖均核对一致，因此全部复用，未重训CV。G默认固定均匀[0,1]、σ=0.1，但分类器C也按相同规则选择；所以它未必等于旧B(C=1)。本轮G默认的C={d.classifier_C:g}，G调整的C={g.classifier_C:g}，A的C={a.classifier_C:g}。

选择规则预先保持为：{RULE}。限制搜索空间是在查看本轮测试评价之前固定的；参数和混淆矩阵展示对象在 parameters_frozen_before_test.json 锁定。正式测试集此前已查看过，属于探索性软件筛选，不是独立最终验证。没有可靠同源ID，沿用既有分层3折并说明可能存在同源/同场景跨折，不猜测来源分组。

运行耗时、复用和新增拟合数见 runtime.json 与 final_model_fit_provenance.json。复用的旧fit耗时保留在CV表中并明确标注历史来源，不计为本轮新训练耗时。无GPU、无硬件操作、无下载。输入排列核查和重复值累加的逐项结果见 order_independence_checks.json，容差在模型拟合前固定。

## 运行

```bash
cd /home/ziyan/FYP_XZH
.venv/bin/python software_feasibility.py
```

仅重绘已完成结果、不重新训练：

```bash
.venv/bin/python software_feasibility.py --render-only
```
"""
    (output/"软件筛选说明.md").write_text(notes,encoding="utf-8")
    return len(brief)


def write_model_definition(output):
    text="""# 目标模型核对

采用现有工程中已经明确的标量Gaussian与核内相加关系，继续复用旧B的计算定义。

1. 先将原始16维输入除以相应训练部分所有数值的全局最大值，得到x_i。验证/测试超界截断；训练最大值为0时使用除数1并记录。
2. 一个输入x_i在第j个核内产生 r_ij=exp(-(x_i-mu_j)^2/(2*sigma^2))，单输入峰值固定为1。mu与sigma的单位是归一化输入坐标；没有引入未经校准的脉冲、电路、拖尾或器件记忆。
3. 核内16个响应按加法叠加：y_j=sum_{i=1}^{16} r_ij。重复的输入仍按其出现次数相加，不去重。
4. 一个核最终交给分类器的仅为一个合计数y_j；16个核总输出[y_1,...,y_16]，每一维表示对应中心和宽度下的响应总和。中间响应R是[N,16,16]，不是分类器的最终输出。
5. 对任意输入排列pi，sum_i g_j(x_pi(i))=sum_i g_j(x_i)，因此最终特征数学上不变。训练后固定的StandardScaler和分类器对相同特征也应产生相同概率与类别；浮点加法顺序造成的微小误差单独核查。

旧B与目标一致，不另造模型。旧C将R按输入位置展开为256维，输入交换会移动特征块；它是此前的位置保留诊断，但不满足本轮目标输出的顺序无关要求。本轮不训练C、不将C成绩当作目标Gaussian结果。

默认：mu_j=linspace(0,1,16)，sigma=0.1，峰值1。有限候选：默认布局或相应训练输入P1至P99内均匀16中心；sigma=0.05/0.1/0.2，含义始终是上述标量Gaussian的统一标准差。三组分类器均比较C=0.1/1/10。中心重合时保留全部16核，不加抖动、不改变维度。未加入时间分箱、排序输出或未知器件动态。
"""
    (output/"模型核对说明.md").write_text(text,encoding="utf-8")
    write_json(output/"model_definition.json",{"input_dimensions":16,"kernels":16,"single_response":"exp(-(x_i-mu_j)^2/(2*sigma^2))",
        "within_kernel_aggregation":"sum over all 16 inputs, with multiplicity","output_dimensions":16,"dimension_j":"sum of scalar Gaussian j over all input positions",
        "peak":1.,"default_centers":np.linspace(0,1,16).tolist(),"default_sigma":DEFAULT_SIGMA,"candidate_width_factors":[.5,1.,2.],
        "candidate_sigmas":SIGMAS,"candidate_center_modes":MODES,"classifier_C":REGULARIZATIONS,"selection_rule":RULE,
        "permutation_invariant":True,"old_B_compatible":True,"old_C_compatible":False,
        "feature_atol":FEATURE_ATOL,"feature_rtol":FEATURE_RTOL,"probability_atol":PROBABILITY_ATOL,"probability_rtol":PROBABILITY_RTOL})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("outputs/hmdb51_software_feasibility_v3"))
    parser.add_argument("--permutations",type=int,default=100)
    parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args();project=Path(__file__).resolve().parent;output=args.output.resolve()
    source=project/"outputs/hmdb51_diagnostics_v2";output.mkdir(parents=True,exist_ok=True)
    if args.permutations<2:raise ValueError("At least two permutations are required")
    signature=run_signature(project,source,args.permutations)
    if (output/"completed.json").exists() and not args.render_only:
        stored=json.loads((output/"run_signature.json").read_text())
        if stored!=signature:
            raise RuntimeError("Completed-run conditions changed; specify a new --output directory instead of mixing experiments")
        print(f"Completed run exists at {output}; no experiments repeated. Use --render-only to regenerate figures.")
        return
    start=time.perf_counter();timings={}
    try:
        with threadpool_limits(limits=1):
            if args.render_only:
                if json.loads((output/"run_signature.json").read_text())!=signature:
                    raise RuntimeError("Source conditions changed; use preserved source artifacts to render the earlier run")
                metrics=pd.read_csv(output/"results_table.csv");selection=json.loads((output/"parameters_frozen_before_test.json").read_text())
                plots(project,output,metrics,selection)
                runtime=json.loads((output/"runtime.json").read_text());runtime["last_render_only_seconds"]=time.perf_counter()-start
                write_documents(output,metrics,selection,runtime);write_json(output/"runtime.json",runtime)
                print("Re-rendered saved results; no fits performed.");return
            roots=[project/"outputs"/name for name in ["hmdb51_six_split1","hmdb51_gaussian_tuning_v1","hmdb51_diagnostics_v2"]]
            write_json(output/"run_signature.json",signature)
            old_hashes=file_manifest(roots);write_json(output/"prior_result_hashes.json",old_hashes)
            write_model_definition(output)
            included,raw,summary,audit=audit_and_reuse(project,source,output)
            timings["matching_checks_and_CV_reuse_seconds"]=time.perf_counter()-start
            selection=lock_selection(summary,output)
            print("Model confirmed: 16 Gaussian sums. Reused all 63 matching CV fits; selection frozen:",selection,flush=True)
            phase=time.perf_counter();x,bundles,logs=final_models(source,included,raw,selection,output)
            timings["final_model_load_or_fit_seconds"]=time.perf_counter()-phase
            phase=time.perf_counter();order=check_permutations(x,included,bundles,args.permutations,output,source)
            timings["permutation_and_duplicate_checks_seconds"]=time.perf_counter()-phase
            phase=time.perf_counter();metrics=evaluate(included,x,bundles,summary,selection,order,output)
            timings["final_evaluation_seconds"]=time.perf_counter()-phase
            phase=time.perf_counter();plots(project,output,metrics,selection)
            timings["plot_seconds"]=time.perf_counter()-phase
            assert all(sha256(p["path"])==p["sha256"] for p in old_hashes)
            runtime={"total_seconds":time.perf_counter()-start,"timings":timings,"CV_fits_reused":63,"CV_fits_new":0,
                "final_models_reused":sum(r["reused"] for r in logs),"final_model_fits_new":sum(not r["reused"] for r in logs),
                "peak_process_RSS_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"GPU_used":False,
                "CPU_logical_count":psutil.cpu_count(),"memory_total_bytes":psutil.virtual_memory().total,"BLAS_threads":1,
                "python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"opencv":cv2.__version__,
                "previous_results_unchanged":True,"hardware_operated":False,"download_performed":False}
            write_json(output/"runtime.json",runtime)
            length=write_documents(output,metrics,selection,runtime)
            code=output/"source_code";code.mkdir(exist_ok=True)
            for name in ["software_feasibility.py","diagnose_hmdb51.py","run_experiment.py","tune_gaussian.py","requirements.txt","README.md"]:
                shutil.copy2(project/name,code/name)
            (output/"run_command.txt").write_text("cd /home/ziyan/FYP_XZH\n"+shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+"\n",encoding="utf-8")
            (output/"environment_freeze.txt").write_text("\n".join(f"{dist.metadata['Name']}=={dist.version}" for dist in __import__('importlib.metadata',fromlist=['distributions']).distributions() if dist.metadata['Name'])+"\n",encoding="utf-8")
            write_json(output/"completed.json",{"status":"PASSED","brief_summary_characters":length,"all_order_checks_passed":True,
                "final_parameters_frozen_before_test":True,"CV_fits_reused":63,"CV_fits_new":0,"previous_results_unchanged":True})
            print(metrics[["method","train_accuracy","validation_accuracy_mean","validation_accuracy_std","validation_macro_f1_mean","test_accuracy","test_macro_f1","classifier_C","sigma"]].to_string(index=False))
            print("New fits:",runtime["final_model_fits_new"],"; Saved:",output)
    except Exception as exc:
        write_json(output/"failure.json",{"status":"FAILED","error":str(exc),"traceback":traceback.format_exc(),"stage_timings":timings})
        raise


if __name__=="__main__":main()
