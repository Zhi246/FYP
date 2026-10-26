#!/usr/bin/env python3
"""Render actual input/recognition diagnostics without training any model."""
import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_kth_spatiotemporal as task

DISPLAY={"A_linear":"Linear16","G0_mean":"G0 mean training","G1_head_cycle":"G1 Cycle-trained head","I_ideal":"Ideal joint RBF (diagnostic)"}
ENCODINGS={task.OLD:"Temporal16",task.NEW:"Region4 x Time4"}
COLORS=["#64748b","#248c87","#d28b43"]


def bootstrap_pairs(predictions,root):
    data=predictions[(predictions.split=="test")&(predictions.condition=="cycle")]
    arrays={};baseline=None;subjects=None
    for (representation,scheme),part in data.groupby(["representation","scheme"]):
        part=part.sort_values(["repeat","sample_id"])
        if baseline is None:
            baseline=part.sample_id.to_numpy();subjects=part[part.repeat==0].subject.to_numpy()
        np.testing.assert_array_equal(part.sample_id.to_numpy(),baseline)
        arrays[(representation,scheme)]=part.correct.to_numpy(dtype=float).reshape(10,862)
    cases=[((task.NEW,name),(task.OLD,name),"new_input_minus_old_input") for name in task.SCHEMES]
    cases += [((task.NEW,"G1_head_cycle"),(task.NEW,"A_linear"),"new_G1_minus_new_Linear"),
              ((task.NEW,"G0_mean"),(task.NEW,"A_linear"),"new_G0_minus_new_Linear"),
              ((task.NEW,"G1_head_cycle"),(task.NEW,"G0_mean"),"new_G1_minus_new_G0")]
    people=np.unique(subjects);counts=np.array([(subjects==p).sum() for p in people]);records=[]
    for first,second,label in cases:
        differences=arrays[first]-arrays[second]
        sums=np.column_stack([differences[:,subjects==p].sum(axis=1) for p in people])
        rng=np.random.default_rng(42)
        person_draws=rng.integers(0,len(people),size=(2000,len(people)))
        cycle_draws=rng.integers(0,10,size=(2000,10))
        boot=sums[cycle_draws[:,:,None],person_draws[:,None,:]].sum(axis=(1,2))/(10*counts[person_draws].sum(axis=1))
        low,high=np.quantile(boot,[.025,.975])
        records.append({"comparison":label,"first_representation":first[0],"first_scheme":first[1],
                        "second_representation":second[0],"second_scheme":second[1],
                        "accuracy_difference":float(differences.mean()),"paired_95_low":float(low),"paired_95_high":float(high),
                        "subjects":9,"bank_draw_repeats":10,"bootstrap_repeats":2000,"seed":42,
                        "independent_new_test_data":False})
    frame=pd.DataFrame(records);frame.to_csv(root/"paired_comparisons.csv",index=False)
    return frame


def draw_table(root,table):
    rows=[]
    for name in task.PRIMARY:
        for representation in (task.OLD,task.NEW):
            row=table.loc[(representation,name)]
            rows.append([ENCODINGS[representation],DISPLAY[name],f"{row.train_pool_mean_accuracy*100:.2f}%",
                f"{row.validation_cycle_macro_f1_mean:.4f}",
                f"{row.test_cycle_accuracy_mean*100:.2f} +/- {row.test_cycle_accuracy_std*100:.2f}%",
                f"{row.test_cycle_macro_f1_mean:.4f}",f"{row.test_noise_accuracy_mean*100:.2f}%"])
    fig,ax=plt.subplots(figsize=(16,4.8));ax.axis("off")
    cells=ax.table(cellText=rows,colLabels=["Input (16D)","Classifier path","Train mean Acc","Val Cycle F1",
        "Test Cycle Acc","Test Cycle F1","Cycle + noise Acc"],colWidths=[.15,.22,.13,.11,.17,.12,.13],loc="center",cellLoc="center")
    cells.auto_set_font_size(False);cells.set_fontsize(10);cells.scale(1,2.)
    for column in range(7):
        cells[0,column].set_facecolor("#d8ebe9");cells[6,column].set_facecolor("#edf7f5")
    ax.set_title("KTH | same 760 / 767 / 862 clips and same Cycle draws | Gaussian sigma fixed at 0.25\n+/- = SD over 10 bank draws, not subject uncertainty; old-input fits reused",pad=14)
    fig.tight_layout();fig.savefig(root/"results_table.png",bbox_inches="tight");plt.close(fig)


def input_panel(ax,example,new_scale):
    heat=example["new_raw"]/new_scale
    ax.imshow(heat.reshape(4,4),cmap="viridis",vmin=0,vmax=max(float(heat.max()),.1),aspect="auto")
    ax.set(xticks=np.arange(4),xticklabels=["0-.25 s",".25-.5 s",".5-.75 s",".75-1 s"],
           yticks=np.arange(4),yticklabels=["Top left","Top right","Bottom left","Bottom right"],
           xlabel="Four groups of four temporal differences",ylabel="Fixed image region",
           title="New input: 4 regions x 4 time groups = 16 values\n"+example["sample_id"])
    for r in range(4):
        for t in range(4):
            text_color="#172b36" if heat.reshape(4,4)[r,t]>.65*max(float(heat.max()),.1) else "white"
            ax.text(t,r,f"x{r*4+t+1}\n{heat.reshape(4,4)[r,t]:.3f}",ha="center",va="center",color=text_color,fontsize=9)


def performance_panel(ax,table):
    for index,(representation,color) in enumerate(zip((task.OLD,task.NEW),COLORS)):
        subset=table.loc[representation].loc[task.PRIMARY]
        bars=ax.bar(np.arange(3)+(index-.5)*.34,subset.test_cycle_accuracy_mean*100,.34,
                    yerr=subset.test_cycle_accuracy_std*100,capsize=3,color=color,label=ENCODINGS[representation])
        ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=9)
    ax.set(ylim=(0,100),xticks=np.arange(3),xticklabels=["Linear16","G0","G1"],ylabel="Test accuracy (%)",
           title="Single held-out Cycle | matched clips and draws")
    ax.legend(fontsize=9)


def ideal_panel(ax,table):
    settings=[("I_ideal","test_cycle_accuracy_mean","Ideal RBF (software)",COLORS[0]),
              ("G0_mean","test_pool_mean_accuracy","Bank G0 / mean reference",COLORS[2]),
              ("G1_head_cycle","test_cycle_accuracy_mean","Bank G1 / single Cycle",COLORS[1])]
    for index,(scheme,key,label,color) in enumerate(settings):
        values=[table.loc[(representation,scheme),key]*100 for representation in (task.OLD,task.NEW)]
        bars=ax.bar(np.arange(2)+(index-1)*.25,values,.25,label=label,color=color)
        ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
    ax.set(ylim=(0,100),xticks=np.arange(2),xticklabels=[ENCODINGS[task.OLD],ENCODINGS[task.NEW]],ylabel="Test accuracy (%)",
           title="Ideal versus bank-calibrated joint Gaussian\nIdeal is a diagnostic, not a hardware result")
    ax.legend(fontsize=8)


def confusion_panel(ax,root,locked):
    name=f"confusion_{locked['representation']}_{locked['scheme']}_cycle_first_draw.csv"
    matrix=pd.read_csv(root/name,index_col=0).to_numpy()
    task.cycle.shared.confusion_panel(ax,matrix,"Validation-selected new input + G1\nFirst fixed test draw (seed=3042), 862 clips")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("outputs/kth_spatiotemporal_v1"))
    args=parser.parse_args();root=args.output.resolve();started=time.perf_counter()
    assert json.loads((root/"COMPLETE.json").read_text())["completed"]
    signature=json.loads((root/"signature.json").read_text());source=Path(signature["old_results"])
    table=pd.read_csv(root/"results_table.csv").set_index(["representation","scheme"])
    predictions=pd.read_csv(root/"sample_predictions.csv")
    paired=bootstrap_pairs(predictions,root)
    locked=json.loads((root/"locked_selection_before_new_test.json").read_text())["gaussian"]
    raw_new=np.load(root/"new_inputs.npz");raw_old=np.load(source/"inputs.npz")
    manifest=pd.read_csv(root/"sample_manifest.csv")
    examples=np.load(root/"preprocessing/training_examples.npz")
    ex_index=np.flatnonzero(examples["classes"]=="walking")[0]
    example={key:examples[key][ex_index] for key in ("sample_ids","old_raw","new_raw","gray_frames_u8","sampled_indices","fps")}
    example["sample_id"]=str(example.pop("sample_ids"))
    new_scale=json.loads((root/"new_input_scaling.json").read_text())["denominator"]
    old_scale=float(raw_old["raw"][raw_old["splits"]=="train"].max())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"savefig.dpi":240,
                         "axes.spines.top":False,"axes.spines.right":False})
    (root/"figures").mkdir(exist_ok=True)
    draw_table(root,table)
    functions=[("input_encoding",lambda ax:input_panel(ax,example,new_scale)),
               ("performance",lambda ax:performance_panel(ax,table)),
               ("ideal_bank_comparison",lambda ax:ideal_panel(ax,table)),
               ("confusion",lambda ax:confusion_panel(ax,root,locked))]
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    for (_,function),ax in zip(functions,axes.ravel()):function(ax)
    fig.tight_layout();fig.savefig(root/"four_panel_summary.png");plt.close(fig)
    for name,function in functions:
        fig,ax=plt.subplots(figsize=(9,6));function(ax);fig.tight_layout()
        fig.savefig(root/"figures"/("panel_"+name+".png"));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    axes[0].imshow(example["gray_frames_u8"][8],cmap="gray",vmin=0,vmax=255)
    axes[0].axhline(31.5,color="#26aaa0");axes[0].axvline(31.5,color="#26aaa0")
    middle_index=int(example["sampled_indices"][8])
    axes[0].set(title=f"Real frame #{middle_index}, t={middle_index/float(example['fps']):.2f} s\nFixed 2x2 image grid",xlabel="Pixel column",ylabel="Pixel row")
    axes[1].plot(np.arange(1,17),example["old_raw"]/old_scale,"o-",color=COLORS[0],label="Old temporal16")
    axes[1].plot(np.arange(1,17),example["new_raw"]/new_scale,"s--",color=COLORS[1],label="New region-major16")
    axes[1].set(xlabel="Input position (different meanings)",ylabel="Training-global-max scaled input",title="Same clip, two 16-dimensional encodings")
    axes[1].legend(fontsize=8)
    input_panel(axes[2],example,new_scale)
    fig.tight_layout();fig.savefig(root/"figures/input_example_with_real_frame.png");plt.close(fig)
    cv=pd.read_csv(root/"group_cv_fold_means.csv")
    fig,ax=plt.subplots(figsize=(10,5.5))
    for representation,marker in ((task.OLD,"o"),(task.NEW,"s")):
        for scheme,color in (("A_linear",COLORS[0]),("G1_head_cycle",COLORS[1])):
            subset=cv[(cv.representation==representation)&(cv.scheme==scheme)].sort_values("fold")
            ax.plot(subset.fold+1,subset.accuracy*100,marker+"-" if representation==task.NEW else marker+"--",
                    color=color,label=ENCODINGS[representation]+" / "+DISPLAY[scheme])
    ax.set(ylim=(0,100),xticks=[1,2,3,4],xlabel="Same subject-grouped fold",ylabel="Validation accuracy (%)",
           title="Training-people-only stability audit | five Cycle draws per fold")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/"figures/group_cv_input_comparison.png");plt.close(fig)
    stat_rows=[];train=manifest.split.to_numpy()=="train";bins=np.linspace(0,1,41)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharex=True,sharey=True)
    for ax,(representation,data,scale) in zip(axes,[(task.OLD,raw_old,old_scale),(task.NEW,raw_new,new_scale)]):
        for cls in task.cycle.shared.CLASSES:
            mask=train&(manifest["class"].to_numpy()==cls);values=(data["raw"][mask]/scale).ravel()
            ax.hist(values,bins=bins,density=True,histtype="step",label=cls,linewidth=1.3)
            row={"representation":representation,"class":cls,"n_clips":int(mask.sum()),"minimum":values.min(),
                 "maximum":values.max(),"mean":values.mean(),"std":values.std()}
            row.update({"q"+str(int(p*100)):float(np.quantile(values,p)) for p in [.01,.05,.25,.5,.75,.95,.99]})
            stat_rows.append(row)
        ax.set(xlabel="Scaled raw input (shared bin edges)",ylabel="Density",title=ENCODINGS[representation]+" | training data only")
        ax.legend(fontsize=8)
    pd.DataFrame(stat_rows).to_csv(root/"training_input_distribution_statistics.csv",index=False)
    fig.tight_layout();fig.savefig(root/"figures/training_input_distributions.png");plt.close(fig)
    confusion_rows=[]
    for representation in (task.OLD,task.NEW):
        for name in task.PRIMARY:
            matrix=pd.read_csv(root/f"confusion_{representation}_{name}_cycle_all_draws.csv",index_col=0)
            for true in task.cycle.shared.CLASSES:
                support=int(matrix.loc[true].sum())
                for predicted in task.cycle.shared.CLASSES:
                    confusion_rows.append({"representation":representation,"scheme":name,"true_class":true,"predicted_class":predicted,
                        "count_over_10_draws":int(matrix.loc[true,predicted]),"support_over_10_draws":support,
                        "unique_clip_support":support//10,"fraction":float(matrix.loc[true,predicted]/support)})
    pd.DataFrame(confusion_rows).to_csv(root/"class_confusion_rates.csv",index=False)
    scenario_rows=[]
    test=predictions[(predictions.split=="test")&(predictions.condition=="cycle")].merge(
        manifest[["sample_id","scenario"]],on="sample_id",validate="many_to_one")
    for keys,part in test.groupby(["representation","scheme","scenario","repeat"]):
        metric=task.cycle.shared.score([task.cycle.shared.CLASSES.index(c) for c in part["class"]],
                                        [task.cycle.shared.CLASSES.index(c) for c in part.predicted])
        scenario_rows.append(dict(zip(("representation","scheme","scenario","repeat"),keys),n=len(part),**metric))
    pd.DataFrame(scenario_rows).to_csv(root/"test_cycle_per_scenario_metrics.csv",index=False)
    analysis={"paired_comparisons":paired.to_dict("records"),"selected_by_validation":locked,
        "new_input_gain_G1_accuracy_pp":float((table.loc[(task.NEW,"G1_head_cycle"),"test_cycle_accuracy_mean"]-
                                               table.loc[(task.OLD,"G1_head_cycle"),"test_cycle_accuracy_mean"])*100),
        "new_input_gain_linear_accuracy_pp":float((table.loc[(task.NEW,"A_linear"),"test_cycle_accuracy_mean"]-
                                                   table.loc[(task.OLD,"A_linear"),"test_cycle_accuracy_mean"])*100),
        "render_and_analysis_wall_seconds":time.perf_counter()-started,
        "render_process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    task.cycle.shared.write_json(root/"analysis_summary.json",analysis)
    for filename in (Path(__file__).name,"verify_kth_spatiotemporal.py","test_kth_spatiotemporal_encoding.py","test_kth_spatiotemporal_selection.py"):
        shutil.copy2(filename,root/"source_code"/filename)
    print(paired.to_string(index=False),flush=True)
    print("Input examples, table, four-panel figures and paired diagnostics saved:",root,flush=True)


if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
