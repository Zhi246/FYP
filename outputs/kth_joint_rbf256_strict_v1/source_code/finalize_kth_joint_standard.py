#!/usr/bin/env python3
"""Select the screenshot-strict factor clip [1e-6,1] from completed fits.

No training is repeated. The earlier upper=1.2 branch remains intact, explicitly
labelled an extended software diagnostic rather than the strict saved standard.
"""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_kth_joint_rbf as core


def main():
    start = time.perf_counter()
    parent = Path("outputs/kth_joint_rbf256_v1").resolve()
    output = Path("outputs/kth_joint_rbf256_strict_v1").resolve()
    assert (parent/"COMPLETE.json").is_file()
    import sys
    sys.path.insert(0,"/home/ziyan/SURF2026/rbf_hardware/src")
    from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer
    from rbf_hardware.inference.empirical_response import EmpiricalHardwareResponseBank
    api={"transform":FullJointGaussianTransformer}
    signature={"parent_signature_sha256":core.sha256(parent/"signature.json"),
               "parent_search_sha256":core.sha256(parent/"validation_search.csv"),
               "code_sha256":core.sha256(__file__), "factor_upper_fixed":1., "new_head_fits":0,
               "reason":"User screenshot saved standard fixes clipping to [1e-6,1]; upper1.2 is diagnostic only."}
    output.mkdir(parents=True,exist_ok=True)
    if (output/"signature.json").exists():
        assert json.loads((output/"signature.json").read_text())==signature
    core.write_json(output/"signature.json",signature)
    if (output/"COMPLETE.json").is_file():
        print("Identical completed strict selection reused; no training or evaluations repeated.")
        return
    for folder in ("models","figures","source_code"):
        (output/folder).mkdir(exist_ok=True)
    for filename in ("input_and_bank_mean.npz","centers.npy","input_checks.json",
                     "scaling_and_adapter.json","sample_manifest.csv","class_counts.csv",
                     "preregistered_search.json","previous_exclusions.csv",
                     "previous_official_missing_videos.csv","previous_independent_video_audit.json",
                     "previous_decoder_warning_audit.json","previous_data_and_cost_stats.json"):
        shutil.copy2(parent/filename,output/filename)
    for filename in ("run_kth_joint_rbf.py","finalize_kth_joint_standard.py","test_kth_joint_rbf.py","requirements_joint.txt"):
        shutil.copy2(Path(filename),output/"source_code"/filename)
    archive=np.load(parent/"input_and_bank_mean.npz")
    q,h,y,splits=[archive[k] for k in ("quantized","bank_mean","labels","splits")]
    masks={split:splits==split for split in ("train","validation","test")}
    manifest=pd.read_csv(parent/"sample_manifest.csv")
    search=pd.read_csv(parent/"validation_search.csv").to_dict("records")
    eligible=[r for r in search if r["family"]=="hardware" and r["factor_upper"]==1.]
    strict=min(eligible,key=core.selection_key)
    # This restricted selection is a contract correction, not a test-driven change.
    core.write_json(output/"strict_selection_locked_before_new_test_evaluation.json",{
        "selection_rule":"validation Macro-F1, accuracy, smaller head_alpha/sigma/calibration_alpha; upper fixed1",
        "candidate_count":len(eligible),"strict_selected":strict,
        "parent_upper1p2_result_not_a_strict_replication":True,
        "test_not_used_to_select":True})
    print("Strict configuration locked from validation only:",strict,flush=True)
    previous_table=pd.read_csv(parent/"results_table.csv")
    previous_predictions=pd.read_csv(parent/"sample_predictions.csv")
    previous_person=pd.read_csv(parent/"per_subject_metrics.csv")
    selected=[(r.scheme,r.scheme,r._asdict()) for r in previous_table.itertuples(index=False)
              if r.scheme not in ("Hardware_joint256_tuned","Ideal_RBF256_matched_to_hardware")]
    selected += [("Extended_clip_1p2_diagnostic","Hardware_joint256_tuned",
                  previous_table.set_index("scheme").loc["Hardware_joint256_tuned"].to_dict())]
    selected += [("Hardware_joint256_tuned",None,strict)]
    ideal=min([r for r in search if r["family"]=="ideal" and r["sigma"]==strict["sigma"]
               and r["head_alpha"]==strict["head_alpha"]],key=core.selection_key)
    selected += [("Ideal_RBF256_matched_to_hardware",None,ideal)]
    final_rows,all_predictions,all_subjects,models,matrices=[],[],[],{},{}
    for name,old_name,row in selected:
        if old_name is not None:
            payload=joblib.load(parent/"models"/(old_name+".joblib"))
            local=previous_predictions[previous_predictions.scheme==old_name].copy()
            local["scheme"]=name
            result=dict(row,scheme=name,fit_reused=True)
            people=previous_person[previous_person.scheme==old_name].copy()
            people["scheme"]=name
        else:
            payload=joblib.load(parent/"search_cache"/(row["id"]+".joblib"))
            payload["scaling_maximum"]=json.loads((parent/"scaling_and_adapter.json").read_text())["raw_global_maximum"]
            payload["classes"]=core.CLASSES
            x=core.features_from_payload(payload,q,h,api)
            result=dict(row,scheme=name,train_n=int(masks["train"].sum()),validation_n=int(masks["validation"].sum()),
                        test_n=int(masks["test"].sum()),fit_reused=True)
            parts,person_rows=[],[]
            for split,mask in masks.items():
                scores=core.numpy_scores(x[mask],payload["head"])
                pred=scores.argmax(axis=1)
                result.update({split+"_"+k:v for k,v in core.score(y[mask],pred).items()})
                part=manifest.loc[mask,["sample_id","video_id","subject","class","split"]].copy()
                part["scheme"]=name;part["predicted"]=[core.CLASSES[i] for i in pred];part["correct"]=pred==y[mask]
                for i,cls in enumerate(core.CLASSES):part["score_"+cls]=scores[:,i]
                parts.append(part)
                for subject,group in part.groupby("subject"):
                    person_rows.append(dict(scheme=name,split=split,subject=int(subject),n=len(group),
                                            **core.score([core.CLASSES.index(c) for c in group["class"]],
                                                         [core.CLASSES.index(c) for c in group.predicted])))
            local=pd.concat(parts,ignore_index=True);people=pd.DataFrame(person_rows)
            np.savez_compressed(output/"models"/(name+"_features.npz"),features=x,sample_ids=archive["sample_ids"],
                                labels=y,splits=splits)
        models[name]=payload
        joblib.dump(payload,output/"models"/(name+".joblib"),compress=3)
        final_rows.append(result);all_predictions.append(local);all_subjects.append(people)
        test=local[local.split=="test"].set_index("sample_id").loc[archive["sample_ids"][masks["test"]]]
        predicted=np.array([core.CLASSES.index(c) for c in test.predicted])
        from sklearn.metrics import confusion_matrix,classification_report
        matrix=confusion_matrix(y[masks["test"]],predicted,labels=range(6))
        matrices[name]=matrix
        pd.DataFrame(matrix,index=core.CLASSES,columns=core.CLASSES).to_csv(output/("confusion_"+name+".csv"))
        core.write_json(output/("classification_report_"+name+".json"),classification_report(
            y[masks["test"]],predicted,labels=range(6),target_names=core.CLASSES,output_dict=True,zero_division=0))
    table=pd.DataFrame(final_rows)
    table.to_csv(output/"results_table.csv",index=False)
    predictions=pd.concat(all_predictions,ignore_index=True)
    predictions.to_csv(output/"sample_predictions.csv",index=False)
    pd.concat(all_subjects,ignore_index=True).to_csv(output/"per_subject_metrics.csv",index=False)
    core.mathematical_checks(models,q,h,masks,api,output)
    core.verify_saved(output,q,h,y,splits,api)
    core.write_json(output/"paired_subject_bootstrap.json",core.subject_bootstrap(predictions))
    # Reuse the identical default frozen-model stress tests. Only the newly selected
    # strict model receives new cycle-draw evaluations; no model is retrained.
    old_robust=pd.read_csv(parent/"bank_cycle_robustness.csv")
    records=old_robust[old_robust.scheme!="Hardware_joint256_tuned"].to_dict("records")
    bank_archive=np.load("/home/ziyan/SURF2026/rbf_hardware/dataset/empirical/hardware_response_samples.npz")
    bank=EmpiricalHardwareResponseBank(bank_archive["differential_levels"].astype(float),
                                      bank_archive["response_samples"],bank_archive["group_magnitude_references"])
    saved_preds={}
    for condition,bounds in (("recorded_cycle_only",(0.,0.)),("recorded_cycle_plus_simulator_noise",(.01,.05))):
        for repeat in range(10):
            drawn=bank.simulate(q[masks["test"]],sampling_mode="empirical",random_generator=np.random.default_rng(1042+repeat),
                                minimum_noise_rate=bounds[0],maximum_noise_rate=bounds[1])
            x=core.features_from_payload(models["Hardware_joint256_tuned"],q[masks["test"]],drawn,api)
            pred=core.numpy_scores(x,models["Hardware_joint256_tuned"]["head"]).argmax(axis=1)
            records.append(dict(condition=condition,repeat=repeat,seed=1042+repeat,scheme="Hardware_joint256_tuned",
                                n=len(pred),**core.score(y[masks["test"]],pred)))
            saved_preds[f"{condition}_{repeat}"]=pred
    robustness=pd.DataFrame(records)
    robustness.to_csv(output/"bank_cycle_robustness.csv",index=False)
    robustness.groupby(["condition","scheme"])[["accuracy","macro_f1"]].agg(["mean","std","min","max"]).to_csv(
        output/"bank_cycle_robustness_summary.csv")
    np.savez_compressed(output/"strict_cycle_predictions.npz",labels=y[masks["test"]],**saved_preds)
    # Plotting uses only upper1 for the authoritative heatmap. The full parent
    # search, including the upper1.2 diagnostic, remains separately available.
    core.make_figures(output,table,pd.DataFrame(search),matrices,models,q,h,masks,robustness,api)
    primary_names=["Linear16_standardized","Hardware_joint256_default","Hardware_joint256_default_head_tuned","Hardware_joint256_tuned"]
    table[table.scheme.isin(primary_names)].to_csv(output/"primary_results_table.csv",index=False)
    core.write_json(output/"COMPLETE.json",{"completed":True,"new_head_fits":0,"all_fits_reused_from_parent":True,
        "additional_wall_seconds":time.perf_counter()-start,"strict_factor_lower":1e-6,"strict_factor_upper":1.})
    print(table[["scheme","train_accuracy","validation_accuracy","test_accuracy","test_macro_f1"]].to_string(index=False),flush=True)
    print("Strict standard complete:",output,flush=True)


if __name__=="__main__":
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
