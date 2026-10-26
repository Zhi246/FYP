#!/usr/bin/env python3
"""Matched old temporal16 vs new region4 x time4 input ablation.

Reuse unchanged old fits. New representation gets train-only preprocessing,
centers, calibration and heads. The original hardware math is unchanged.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

import run_kth_cycle_robustness as cycle

SCHEMES=["A_linear","G0_mean","G1_head_cycle","I_ideal"]
PRIMARY=SCHEMES[:3]
OLD="temporal16"
NEW="region4_time4"


def fit_models(bank,q,y,pool,centers,cycles,api):
    start=time.perf_counter()
    assert cycles.shape==(len(q),8) and set(cycles.ravel())<=set(pool)
    mean=cycle.mean_response(bank,q,pool)
    indices=bank._resolve_level_indices(q)
    augmented=bank.response_samples[indices[:,None,:],cycles[:,:,None]].reshape(-1,256)
    begin=time.perf_counter()
    transformer=cycle.new_transform(api,len(q)).fit(mean,q,preset_centers=centers)
    cal_seconds=time.perf_counter()-begin
    mean_q=q.astype(float).mean(axis=0)
    std_q=np.maximum(q.astype(float).std(axis=0),1e-12)
    features={"A_linear":(q-mean_q)/std_q,"G0_mean":transformer.transform(mean),
              "G1_head_cycle":transformer.transform(augmented),"I_ideal":transformer.ideal_pc_transform(q)}
    models={};timings=[]
    for name in SCHEMES:
        x=features[name];targets=np.repeat(y,8) if name=="G1_head_cycle" else y
        anchor=1. if name=="A_linear" else cycle.HEAD_ALPHA_ANCHOR
        begin=time.perf_counter();head=cycle.fit_head(x,targets,anchor,api)
        seconds=time.perf_counter()-begin
        state=head.cpu_state_dict();prediction=cycle.shared.numpy_scores(x,state).argmax(axis=1)
        models[name]={"kind":"linear" if name=="A_linear" else "ideal" if name=="I_ideal" else "hardware",
            "transform":transformer.state_dict(),"head":state,"mean":mean_q,"std":std_q,
            "classes":cycle.shared.CLASSES,"fit_split":"train","unique_train_n":len(q),
            "head_training_rows":len(x),"ridge_anchor_alpha":anchor,
            "sum_form_head_alpha":cycle.sum_alpha(anchor,len(x)),
            "average_loss_head_lambda":anchor/cycle.ANCHOR_N,
            "fit_training_row_metrics":cycle.shared.score(targets,prediction)}
        timings.append({"scheme":name,"head_fit_seconds":seconds,"head_rows":len(x),
                        "sum_form_head_alpha":cycle.sum_alpha(anchor,len(x)),
                        "calibration_sum_alpha":transformer.calibration_alpha,
                        "calibration_fit_seconds":cal_seconds if name=="G0_mean" else 0.})
    return models,{"mean":mean,"augmented":augmented,"features":features,"cycles":cycles,
                   "timings":timings,"fit_seconds":time.perf_counter()-start}


def mathematical_checks(models,training,y,api):
    records=[]
    for name in SCHEMES:
        model=models[name];x=training["features"][name].astype(float)
        labels=np.repeat(y,8) if name=="G1_head_cycle" else y
        z=np.column_stack((np.ones(len(x)),x));target=np.eye(6)[labels]
        reg=np.eye(z.shape[1])*model["sum_form_head_alpha"];reg[0,0]=0
        residual=(z.T@z+reg)@np.asarray(model["head"]["combined_weights"])-z.T@target
        relative=float(np.linalg.norm(residual)/max(np.linalg.norm(z.T@target),1e-30))
        assert relative<1e-8
        records.append({"scheme":name,"normal_equation_relative_residual":relative})
    transformer=api["transform"].from_state_dict(models["G0_mean"]["transform"])
    grouped=training["mean"].astype(float).reshape(-1,16,16)
    np.testing.assert_array_equal(grouped.mean(axis=0),transformer.channel_mean_)
    np.testing.assert_array_equal(np.maximum(grouped.std(axis=0),1e-12),transformer.channel_std_)
    product=np.ones((12,256))
    for d in range(16):
        design=np.column_stack((np.ones(12),(grouped[:12,d]-transformer.channel_mean_[d])/transformer.channel_std_[d]))
        product*=np.clip(design@transformer.calibration_weights_[d],1e-6,1.)
    phi=transformer.transform(training["mean"][:12])
    np.testing.assert_allclose(phi,product,atol=2e-7,rtol=2e-6)
    for field in ("channel_mean","channel_std","centers","calibration_weights"):
        np.testing.assert_array_equal(models["G0_mean"]["transform"][field],models["G1_head_cycle"]["transform"][field])
    return {"passed":True,"checks":records,"independent_product_max_abs_error":float(np.abs(phi-product.astype(np.float32)).max()),
            "train_only_response_statistics_exact":True,"g0_g1_calibration_identical":True,
            "inference_uses_only_hardware_responses_for_gaussian":True}


def evaluate(models,bank,q,y,manifest,pool,refs,split,condition,repeat,seed,expected_cycles,api):
    if condition=="pool_mean":
        h=cycle.mean_response(bank,q,pool);cycles=np.full(len(q),-1)
    else:
        h,cycles=cycle.draw_response(bank,q,pool,seed,refs,noise=condition=="cycle_plus_simulator_noise")
    np.testing.assert_array_equal(cycles,expected_cycles)
    metrics=[];predictions=[];values={};phi=None
    ideal=cycle.model_features(models["I_ideal"],q,h,api)
    diagnostics=[]
    for name in SCHEMES:
        begin=time.perf_counter()
        if name in ("G0_mean","G1_head_cycle"):
            if phi is None:phi=cycle.model_features(models[name],q,h,api)
            features=phi
        elif name=="I_ideal":features=ideal
        else:features=cycle.model_features(models[name],q,h,api)
        feature_seconds=time.perf_counter()-begin
        assert features.shape==(len(q),16 if name=="A_linear" else 256) and np.isfinite(features).all()
        begin=time.perf_counter();scores,predicted=cycle.predict(models[name],features)
        metrics.append(dict(representation=NEW,scheme=name,split=split,condition=condition,repeat=repeat,seed=seed,
                            unique_n=len(q),**cycle.shared.score(y,predicted),feature_seconds=feature_seconds,
                            prediction_seconds=time.perf_counter()-begin))
        part=manifest[["sample_id","subject","class","split"]].copy()
        part["representation"]=NEW;part["scheme"]=name;part["condition"]=condition;part["repeat"]=repeat
        part["seed"]=seed;part["bank_cycle_index_zero_based"]=cycles
        part["predicted"]=[cycle.shared.CLASSES[i] for i in predicted];part["correct"]=predicted==y
        for i,cls in enumerate(cycle.shared.CLASSES):part["score_"+cls]=scores[:,i]
        predictions.append(part);values[name]=features
        if name in ("G0_mean","G1_head_cycle"):
            diagnostics.append({"representation":NEW,"scheme":name,"split":split,"condition":condition,"repeat":repeat,
                "relative_joint_feature_rmse":float(np.linalg.norm(features-ideal)/max(np.linalg.norm(ideal),1e-30)),
                "all_zero_joint_rows":int((features==0).all(axis=1).sum())})
    return metrics,predictions,values,diagnostics


def select_validation(metrics):
    subset=metrics[(metrics.split=="validation")&(metrics.condition=="cycle")]
    values=[]
    for (representation,scheme),frame in subset.groupby(["representation","scheme"]):
        values.append({"representation":representation,"scheme":scheme,"macro_f1_mean":float(frame.macro_f1.mean()),
                       "accuracy_mean":float(frame.accuracy.mean()),"macro_f1_std":float(frame.macro_f1.std(ddof=1))})
    def key(row):
        return (-row["macro_f1_mean"],-row["accuracy_mean"],row["macro_f1_std"],
                [OLD,NEW].index(row["representation"]),SCHEMES.index(row["scheme"]))
    return {"gaussian":min((r for r in values if r["scheme"] in ("G0_mean","G1_head_cycle")),key=key),
            "linear":min((r for r in values if r["scheme"]=="A_linear"),key=key),
            "best_input_per_primary_scheme":{name:min((r for r in values if r["scheme"]==name),key=key) for name in PRIMARY},
            "selection_uses_validation_cycle_metrics_only":True,"all_candidates":values}


def group_cv(raw,y,manifest,source,bank,pools,quantizer,refs,api,output):
    results=[];scalings=[];timings=[]
    by_id=pd.Index(manifest.sample_id)
    for fold in range(4):
        old_dir=source/"group_cv"/f"fold_{fold}"
        old=np.load(old_dir/"inputs_and_assignments.npz",allow_pickle=False)
        train=by_id.get_indexer(old["fit_sample_ids"]);val=by_id.get_indexer(old["validation_sample_ids"])
        assert (train>=0).all() and (val>=0).all()
        assert not set(manifest.iloc[train].subject)&set(manifest.iloc[val].subject)
        qt,qv,scaling=cycle.prepare_inputs(raw[train],raw[val],quantizer)
        centers=api["kmeans"](qt,256,42,300)
        models,training=fit_models(bank,qt,y[train],pools["train"],centers,old["training_cycles"],api)
        destination=output/"group_cv"/NEW/f"fold_{fold}"
        destination.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination/"inputs_and_assignments.npz",fit_sample_ids=old["fit_sample_ids"],
            validation_sample_ids=old["validation_sample_ids"],quantized_fit=qt,quantized_validation=qv,
            centers=centers,training_cycles=training["cycles"])
        for name,model in models.items():
            model["input_scaling"]=scaling
            joblib.dump(model,destination/(name+".joblib"),compress=3)
        for repeat in range(5):
            seed=4042+100*fold+repeat
            h,cycles=cycle.draw_response(bank,qv,pools["validation"],seed,refs)
            for name in SCHEMES:
                features=cycle.model_features(models[name],qv,h,api);_,prediction=cycle.predict(models[name],features)
                results.append(dict(representation=NEW,fold=fold,repeat=repeat,seed=seed,scheme=name,
                                    validation_n=len(val),train_n=len(train),**cycle.shared.score(y[val],prediction)))
        scalings.append(dict(fold=fold,train_subjects=sorted(set(manifest.iloc[train].subject)),
                            validation_subjects=sorted(set(manifest.iloc[val].subject)),**scaling))
        timings.extend([dict(fold=fold,**r) for r in training["timings"]])
        print(f"New-input subject-grouped fold {fold+1}/4 completed; old-input fits reused.",flush=True)
    old_results=pd.read_csv(source/"group_cv_results.csv")
    old_results=old_results[old_results.scheme.isin(SCHEMES)].copy();old_results["representation"]=OLD
    combined=pd.concat([old_results,pd.DataFrame(results)],ignore_index=True)
    combined.to_csv(output/"group_cv_results.csv",index=False)
    means=combined.groupby(["representation","scheme","fold"])[["accuracy","macro_f1"]].mean().reset_index()
    means.to_csv(output/"group_cv_fold_means.csv",index=False)
    means.groupby(["representation","scheme"])[["accuracy","macro_f1"]].agg(["mean","std"]).to_csv(output/"group_cv_summary.csv")
    cycle.shared.write_json(output/"group_cv_new_input_scaling.json",scalings)
    pd.DataFrame(timings).to_csv(output/"group_cv_new_training_times.csv",index=False)
    shutil.copy2(source/"group_cv_folds.csv",output/"group_cv_folds.csv")


def summarize(metrics,training_models):
    rows=[]
    for (representation,scheme),subset in metrics.groupby(["representation","scheme"]):
        row={"representation":representation,"scheme":scheme,"output_dimension":16 if scheme=="A_linear" else 256,
             "train_unique_n":760,"validation_unique_n":767,"test_unique_n":862,"fit_reused":representation==OLD}
        for split in ("train","validation","test"):
            frame=subset[(subset.split==split)&(subset.condition=="pool_mean")].iloc[0]
            for key in ("accuracy","macro_f1"):row[split+"_pool_mean_"+key]=float(frame[key])
        for split,condition,prefix in (("validation","cycle","validation_cycle"),("test","cycle","test_cycle"),
                                       ("test","cycle_plus_simulator_noise","test_noise")):
            frame=subset[(subset.split==split)&(subset.condition==condition)]
            for key in ("accuracy","macro_f1"):
                row[prefix+"_"+key+"_mean"]=float(frame[key].mean())
                row[prefix+"_"+key+"_std"]=float(frame[key].std(ddof=1))
        row["test_cycle_accuracy_drop_from_pool_mean"]=row["test_pool_mean_accuracy"]-row["test_cycle_accuracy_mean"]
        if representation==NEW:
            row["fit_training_row_accuracy"]=training_models[scheme]["fit_training_row_metrics"]["accuracy"]
            row["fit_training_row_macro_f1"]=training_models[scheme]["fit_training_row_metrics"]["macro_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def verify(output,api):
    signature=json.loads((output/"signature.json").read_text())
    for path,digest in signature["source_hashes"].items():assert cycle.shared.sha256(path)==digest,f"Changed source: {path}"
    source=Path(signature["old_results"])
    manifest=pd.read_csv(output/"sample_manifest.csv")
    old=np.load(source/"inputs.npz");new=np.load(output/"new_inputs.npz")
    split=pd.read_csv(output/"cycle_split.csv")
    pools={name:split.loc[split.split==name,"bank_cycle_index_zero_based"].to_numpy() for name in ("train","validation","test")}
    quantizer=api["quantizer"].load(cycle.UPSTREAM/"dataset/csv/differential_levels.csv",input_features=16,
                                   expected_levels=256,raw_minimum=0,raw_maximum=100)
    bank=api["bank"].load(cycle.UPSTREAM/"dataset/empirical/hardware_response_samples.npz",expected_groups=16,expected_levels=quantizer.levels)
    refs=np.load(output/"training_noise_magnitude_references.npy")
    np.testing.assert_array_equal(refs,np.quantile(np.abs(bank.response_samples[:,pools["train"]]),.95,axis=(0,1)))
    frame=pd.read_csv(output/"sample_predictions.csv")
    verified=[]
    for representation,data in ((OLD,old),(NEW,new)):
        for name in SCHEMES:
            model=joblib.load(output/"models"/representation/(name+".joblib"))
            for (section,condition,repeat),part in frame[(frame.representation==representation)&(frame.scheme==name)].groupby(["split","condition","repeat"]):
                mask=manifest.split.to_numpy()==section;q=data["quantized"][mask]
                part=part.set_index("sample_id").loc[data["sample_ids"][mask]]
                if condition=="pool_mean":h=cycle.mean_response(bank,q,pools[section])
                else:
                    h,cycles=cycle.draw_response(bank,q,pools[section],int(part.seed.iloc[0]),refs,noise=condition=="cycle_plus_simulator_noise")
                    np.testing.assert_array_equal(cycles,part.bank_cycle_index_zero_based.to_numpy())
                features=cycle.model_features(model,q,h,api);scores,prediction=cycle.predict(model,features)
                np.testing.assert_allclose(scores,part[["score_"+c for c in cycle.shared.CLASSES]].to_numpy(),rtol=1e-11,atol=1e-11)
                assert [cycle.shared.CLASSES[i] for i in prediction]==part.predicted.tolist()
            verified.append(representation+"/"+name)
    cycle.shared.write_json(output/"postrun_verification.json",{"passed":True,"saved_models_and_scores_reproduced":verified,
        "source_hashes_unchanged":True,"score_tolerance_atol":1e-11,"score_tolerance_rtol":1e-11})
    print("All eight saved models reproduce every draw-level score and label; source hashes unchanged.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("outputs/kth_cycle_robustness_v1"))
    parser.add_argument("--output",type=Path,default=Path("outputs/kth_spatiotemporal_v1"))
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    torch.set_num_threads(1);torch.manual_seed(42);api=cycle.imports()
    if args.verify_only:verify(args.output,api);return
    started=time.perf_counter();pre=args.output/"preprocessing"
    assert json.loads((pre/"COMPLETE.json").read_text())["completed"]
    oldsig=json.loads((args.source/"signature.json").read_text())
    for path,digest in oldsig["source_hashes"].items():assert cycle.shared.sha256(path)==digest,f"Old-fit condition changed: {path}"
    versions={name:importlib.metadata.version(name) for name in oldsig["versions"]}
    assert versions==oldsig["versions"],"Old-fit numerical environment changed"
    files=[Path(__file__).resolve(),Path(cycle.__file__).resolve(),Path(cycle.shared.__file__).resolve(),
           pre/"raw_features.npz",pre/"signature.json",pre/"extraction_checks.json",
           args.source/"signature.json",args.source/"inputs.npz",args.source/"sample_manifest.csv",
           args.source/"all_metrics.csv",args.source/"sample_predictions.csv",args.source/"group_cv_results.csv",
           args.source/"cycle_split.csv",args.source/"evaluation_cycle_assignments.npz",args.source/"training_responses.npz",
           args.source/"training_noise_magnitude_references.npy"]
    files += [args.source/"models"/(name+".joblib") for name in SCHEMES]
    files += [Path(path) for path in oldsig["source_hashes"] if "rbf_hardware" in path]
    signature={"source_hashes":{str(p):cycle.shared.sha256(p) for p in files},"old_results":str(args.source),
               "versions":versions,"seed":42,"new_representation":NEW,"model_schemes":SCHEMES}
    if (args.output/"signature.json").exists():
        if json.loads((args.output/"signature.json").read_text())!=signature:
            raise RuntimeError("Training signature changed. Preserve results and use a new --output.")
    else:cycle.shared.write_json(args.output/"signature.json",signature)
    if (args.output/"COMPLETE.json").is_file():
        print("Identical completed spatial-temporal experiment reused; no fitting repeated.",flush=True)
        print(pd.read_csv(args.output/"results_table.csv").to_string(index=False));return
    for directory in ("models/"+OLD,"models/"+NEW,"features","source_code"):(args.output/directory).mkdir(parents=True,exist_ok=True)
    checks=json.loads((pre/"extraction_checks.json").read_text());assert checks["passed"] and checks["new_exclusions"]==0
    old=np.load(args.source/"inputs.npz");new=np.load(pre/"raw_features.npz")
    for key in ("sample_ids","labels","splits"):np.testing.assert_array_equal(old[key],new[key])
    np.testing.assert_allclose(old["raw"],new["reconstructed_temporal"],atol=1e-7,rtol=0)
    raw=new["spatiotemporal"];labels=new["labels"];ids=new["sample_ids"];splits=new["splits"]
    assert raw.shape==(2389,16) and np.isfinite(raw).all()
    masks={name:splits==name for name in ("train","validation","test")}
    assert [int(m.sum()) for m in masks.values()]==[760,767,862]
    manifest=pd.read_csv(args.source/"sample_manifest.csv");assert manifest.sample_id.tolist()==ids.tolist()
    assert manifest.groupby("subject").split.nunique().max()==1 and manifest.groupby("video_sha256").split.nunique().max()==1
    manifest.to_csv(args.output/"sample_manifest.csv",index=False)
    manifest.groupby(["split","class"]).size().unstack(fill_value=0).reindex(columns=cycle.shared.CLASSES).to_csv(args.output/"class_counts.csv")
    quantizer=api["quantizer"].load(cycle.UPSTREAM/"dataset/csv/differential_levels.csv",input_features=16,
                                   expected_levels=256,raw_minimum=0,raw_maximum=100)
    bank=api["bank"].load(cycle.UPSTREAM/"dataset/empirical/hardware_response_samples.npz",expected_groups=16,expected_levels=quantizer.levels)
    assignments=pd.read_csv(args.source/"cycle_split.csv")
    pools={name:assignments.loc[assignments.split==name,"bank_cycle_index_zero_based"].to_numpy() for name in masks}
    for name in pools:np.testing.assert_array_equal(pools[name],cycle.cycle_partition()[name])
    refs=np.load(args.source/"training_noise_magnitude_references.npy")
    _,q,scaling=cycle.prepare_inputs(raw[masks["train"]],raw,quantizer)
    scaling["clipped_value_fraction_by_split"]={name:float(np.mean((raw[mask]/scaling["denominator"]>1)|(raw[mask]<0))) for name,mask in masks.items()}
    cycle.shared.write_json(args.output/"new_input_scaling.json",scaling)
    protocol={"written_before_new_fits_or_test_evaluation":True,"primary_comparison":"same 16 input dimensions, old temporal16 vs region4 x time4",
        "feature_order":"top_left times1..4; top_right times1..4; bottom_left times1..4; bottom_right times1..4",
        "only_video_input_encoding_changes":True,"video_sampling_indices_identical":True,
        "sigma":.25,"factor_clip":[1e-6,1.],"joint_centers":256,"seed":42,"new_parameter_search":False,
        "linear_anchor_head_alpha":1.,"gaussian_anchor_head_alpha":.1,"calibration_anchor_alpha":.01,
        "average_loss_anchor_unique_n":760,"g1_augmented_rows":6080,"g1_sum_form_head_alpha":.8,
        "input_scaling":"Representation-specific training global max, held-out clipped; not per-video or per-position",
        "centers_and_all_new_fit_parameters_train_only":True,"g1_standardization_and_calibration_frozen_g0":True,
        "cycle_partitions_and_training_draws_exactly_reused":True,
        "group_cv":"Reuse original 4 subject folds and Cycle assignments; refit all new-input fold parameters using fold train only",
        "selection":"Validation 10-Cycle Macro-F1 mean, accuracy mean, smaller F1 std, old input, simpler scheme in that order",
        "linear_input_selection_same_validation_rule":True,"ideal_reference_is_diagnostic_not_hardware_candidate":True,
        "test_status":"Exploratory: old KTH test split viewed before; new-input tests evaluated after validation lock",
        "no_foreground_tracking_segmentation_cnn_or_new_device_dynamics":True,
        "interpretation":"Changes spatial retention AND temporal aggregation; does not isolate a pure spatial or Gaussian nonlinear effect"}
    cycle.shared.write_json(args.output/"preregistered_protocol.json",protocol)
    centers=api["kmeans"](q[masks["train"]],256,42,300)
    np.savez_compressed(args.output/"new_inputs.npz",raw=raw,quantized=q,labels=labels,splits=splits,sample_ids=ids,centers=centers)
    training_cycles=np.load(args.source/"training_responses.npz")["cycle_indices"]
    models,training=fit_models(bank,q[masks["train"]],labels[masks["train"]],pools["train"],centers,training_cycles,api)
    cycle.shared.write_json(args.output/"implementation_checks.json",mathematical_checks(models,training,labels[masks["train"]],api))
    for name,model in models.items():
        model["input_scaling"]=scaling;model["representation"]=NEW
        joblib.dump(model,args.output/"models"/NEW/(name+".joblib"),compress=3)
        shutil.copy2(args.source/"models"/(name+".joblib"),args.output/"models"/OLD/(name+".joblib"))
        np.savez_compressed(args.output/"features"/(name+"_head_training.npz"),features=training["features"][name],
                            labels=np.repeat(labels[masks["train"]],8) if name=="G1_head_cycle" else labels[masks["train"]])
    np.savez_compressed(args.output/"new_training_responses.npz",mean=training["mean"],augmented=training["augmented"],
                        cycle_indices=training["cycles"],sample_ids=ids[masks["train"]])
    pd.DataFrame(training["timings"]).to_csv(args.output/"new_model_training_times.csv",index=False)
    reused={"old_full_heads":4,"old_primary_heads":3,"old_fold_heads":16,"old_fold_primary_heads":12,
            "identical_samples_features_model_parameters_cycle_assignments_and_environment_verified":True,
            "new_full_heads":4,"new_full_primary_heads":3,"new_full_calibration_fits":1,
            "notes":"I_ideal is the fourth head, a software diagnostic. Old-input fits/predictions are reused unchanged."}
    cycle.shared.write_json(args.output/"reuse_audit.json",reused)
    for filename in ("cycle_split.csv","response_bank_provenance.json","training_noise_magnitude_references.npy",
                     "previous_exclusions.csv","previous_official_missing_videos.csv","previous_data_and_cost_stats.json"):
        shutil.copy2(args.source/filename,args.output/filename)
    print("New-input main models fitted; original-input models reused. Checking the same 4 subject folds.",flush=True)
    begin=time.perf_counter()
    group_cv(raw,labels,manifest,args.source,bank,pools,quantizer,refs,api,args.output)
    cv_seconds=time.perf_counter()-begin
    old_metrics=pd.read_csv(args.source/"all_metrics.csv")
    old_metrics=old_metrics[old_metrics.scheme.isin(SCHEMES)].copy();old_metrics["representation"]=OLD
    old_predictions=pd.read_csv(args.source/"sample_predictions.csv")
    old_predictions=old_predictions[old_predictions.scheme.isin(SCHEMES)].copy();old_predictions["representation"]=OLD
    all_metrics=[old_metrics];all_predictions=[old_predictions];all_diagnostics=[]
    expected=np.load(args.source/"evaluation_cycle_assignments.npz")
    def consume(split,condition,repeat,seed):
        mask=masks[split];key=f"{split}_{condition}_{repeat}"
        rows,parts,features,diagnostics=evaluate(models,bank,q[mask],labels[mask],manifest.loc[mask],pools[split],refs,
                                                split,condition,repeat,seed,expected[key],api)
        all_metrics.append(pd.DataFrame(rows));all_predictions.extend(parts);all_diagnostics.extend(diagnostics)
        np.savez_compressed(args.output/"features"/(key+".npz"),sample_ids=ids[mask],**features)
    consume("train","pool_mean",-1,-1);consume("validation","pool_mean",-1,-1)
    for repeat in range(10):consume("validation","cycle",repeat,2042+repeat)
    validation=pd.concat(all_metrics,ignore_index=True)
    validation=validation[validation.split.isin(["train","validation"])]
    validation.to_csv(args.output/"validation_before_new_test.csv",index=False)
    locked=select_validation(validation)
    cycle.shared.write_json(args.output/"locked_selection_before_new_test.json",locked)
    print(f"Validation lock: Gaussian={locked['gaussian']['representation']}/{locked['gaussian']['scheme']}; Linear={locked['linear']['representation']}. Now evaluating new-input tests.",flush=True)
    consume("test","pool_mean",-1,-1)
    for repeat in range(10):
        consume("test","cycle",repeat,3042+repeat)
        consume("test","cycle_plus_simulator_noise",repeat,3042+repeat)
    metrics=pd.concat(all_metrics,ignore_index=True);predictions=pd.concat(all_predictions,ignore_index=True)
    metrics.to_csv(args.output/"all_metrics.csv",index=False);predictions.to_csv(args.output/"sample_predictions.csv",index=False)
    pd.DataFrame(all_diagnostics).to_csv(args.output/"new_feature_diagnostics.csv",index=False)
    table=summarize(metrics,models)
    table.to_csv(args.output/"results_table.csv",index=False)
    table[table.scheme.isin(PRIMARY)].to_csv(args.output/"primary_results_table.csv",index=False)
    for (representation,name,condition),part in predictions[predictions.split=="test"].groupby(["representation","scheme","condition"]):
        for suffix,subset in (("all_draws",part),("first_draw",part[part.repeat==part.repeat.min()])):
            yy=[cycle.shared.CLASSES.index(c) for c in subset["class"]];pp=[cycle.shared.CLASSES.index(c) for c in subset.predicted]
            matrix=confusion_matrix(yy,pp,labels=range(6));assert np.trace(matrix)==subset.correct.sum()
            prefix=f"{representation}_{name}_{condition}_{suffix}"
            pd.DataFrame(matrix,index=cycle.shared.CLASSES,columns=cycle.shared.CLASSES).to_csv(args.output/("confusion_"+prefix+".csv"))
            cycle.shared.write_json(args.output/("classes_"+prefix+".json"),classification_report(
                yy,pp,labels=range(6),target_names=cycle.shared.CLASSES,output_dict=True,zero_division=0))
    per_person=[]
    for keys,part in predictions.groupby(["representation","scheme","split","condition","repeat","subject"]):
        stats=cycle.shared.score([cycle.shared.CLASSES.index(c) for c in part["class"]],[cycle.shared.CLASSES.index(c) for c in part.predicted])
        per_person.append(dict(zip(("representation","scheme","split","condition","repeat","subject"),keys),n=len(part),**stats))
    pd.DataFrame(per_person).to_csv(args.output/"per_subject_metrics.csv",index=False)
    for path in (Path(__file__),Path(cycle.__file__),Path(cycle.shared.__file__),Path("extract_kth_spatiotemporal.py"),Path("requirements_joint.txt")):
        shutil.copy2(path,args.output/"source_code"/path.name)
    verify(args.output,api)
    costs={"training_evaluation_and_verification_wall_seconds":time.perf_counter()-started,
        "new_main_fit_seconds":training["fit_seconds"],"new_group_cv_seconds":cv_seconds,
        "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "cpu":cycle.shared.cpu_name(),"logical_cpus":psutil.cpu_count(),"torch_threads":torch.get_num_threads(),
        "system_ram_bytes":psutil.virtual_memory().total,"gpu_used":False,"new_main_primary_head_fits":3,
        "new_main_diagnostic_head_fits":1,"new_main_calibration_fits":1,"new_cv_head_fits":16,"new_cv_calibration_fits":4,
        "old_main_head_fits_reused":4,"old_cv_head_fits_reused":16,"new_raw_feature_bytes":raw.nbytes,
        "new_augmented_response_bytes":training["augmented"].nbytes,"one_test_joint_feature_matrix_bytes":862*256*4,
        "note":"Video extraction measured separately in preprocessing/cost_and_environment.json; no hardware latency or energy measured."}
    cycle.shared.write_json(args.output/"cost_and_environment.json",costs)
    cycle.shared.write_json(args.output/"COMPLETE.json",{"completed":True,"old_results_preserved":True,
        "selected_gaussian_by_validation":locked["gaussian"],"selected_linear_by_validation":locked["linear"],
        "source_hashes_unchanged":True,"same_successful_samples":2389})
    print(table[["representation","scheme","validation_cycle_macro_f1_mean","test_pool_mean_accuracy",
                 "test_cycle_accuracy_mean","test_cycle_macro_f1_mean"]].to_string(index=False),flush=True)
    print(f"COMPLETE: {args.output}; model stage {costs['training_evaluation_and_verification_wall_seconds']:.2f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):main()
