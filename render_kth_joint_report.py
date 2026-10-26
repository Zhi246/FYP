#!/usr/bin/env python3
"""Render a concise meeting table and real-video illustration; never fit models."""
import json
import io
from pathlib import Path
import subprocess
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_kth_joint_rbf as core


def main():
    start=time.perf_counter()
    root=Path("outputs/kth_joint_rbf256_strict_v1")
    sys.path.insert(0,"/home/ziyan/SURF2026/rbf_hardware/src")
    from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer
    api={"transform":FullJointGaussianTransformer}
    table=pd.read_csv(root/"results_table.csv").set_index("scheme")
    names=["Linear16_standardized","Hardware_joint256_default",
           "Hardware_joint256_default_head_tuned","Hardware_joint256_tuned"]
    display=["Linear16 + train standardization","Joint G default",
             "Default G + head selection","Joint G tuned (strict clip)"]
    columns=["output_dimension","train_accuracy","validation_accuracy","validation_macro_f1",
             "test_accuracy","test_macro_f1","sigma","calibration_alpha","factor_upper","head_alpha"]
    primary=table.loc[names,columns].copy()
    primary.insert(0,"label",display)
    primary.to_csv(root/"group_meeting_table.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"savefig.dpi":240})
    rows=[]
    for label,name in zip(display,names):
        r=table.loc[name]
        parameters=(f"head alpha={r.head_alpha:g}" if name.startswith("Linear") else
                    f"sigma={r.sigma:g}; cal={r.calibration_alpha:g}\nclip=[1e-6,1]; head={r.head_alpha:g}")
        rows.append([label,int(r.output_dimension),f"{r.train_accuracy*100:.2f}%",
                     f"{r.validation_accuracy*100:.2f}%",f"{r.validation_macro_f1:.4f}",
                     f"{r.test_accuracy*100:.2f}%",f"{r.test_macro_f1:.4f}",parameters])
    fig,ax=plt.subplots(figsize=(17,4.2));ax.axis("off")
    cells=ax.table(cellText=rows,colLabels=["Method","Dim","Train Acc","Val Acc","Val F1","Test Acc","Test F1","Actual parameters"],
        colWidths=[.25,.04,.08,.08,.08,.08,.08,.22],loc="center",cellLoc="center")
    cells.auto_set_font_size(False);cells.set_fontsize(10);cells.scale(1,2.9)
    for c in range(8):cells[0,c].set_facecolor("#d8ebe9")
    ax.set_title("KTH | 760 train / 767 validation / 862 test | original bank-calibrated joint RBF256",pad=15)
    fig.tight_layout();fig.savefig(root/"group_meeting_table.png",bbox_inches="tight");plt.close(fig)
    archive=np.load(root/"input_and_bank_mean.npz")
    manifest=pd.read_csv(root/"sample_manifest.csv")
    index=manifest.index[(manifest.split=="train")&(manifest["class"]=="walking")][0]
    row=manifest.loc[index]
    indices=np.asarray(json.loads(row.sampled_indices))
    chosen_indices=indices[[0,8,16]]
    # Existing Torch environment has NumPy2 but a NumPy1 OpenCV binary. Decode
    # presentation frames in the original working video environment instead;
    # do not change packages or regenerate any training feature.
    decoder = Path(__file__).resolve().parent/".venv/bin/python"
    decoder_code = """import sys,json,io,cv2,numpy as np
indices=json.loads(sys.argv[2]); capture=cv2.VideoCapture(sys.argv[1]); frames=[]
for index in range(max(indices)+1):
    ok,bgr=capture.read()
    if not ok: raise RuntimeError('Cannot read requested presentation frame')
    if index in indices: frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
capture.release()
buffer=io.BytesIO();np.save(buffer,np.stack(frames),allow_pickle=False)
sys.stdout.buffer.write(buffer.getvalue())
"""
    decoded = subprocess.run([str(decoder),"-c",decoder_code,row.path,
                              json.dumps(chosen_indices.tolist())],capture_output=True,check=True)
    frames=np.load(io.BytesIO(decoded.stdout),allow_pickle=False)
    assert len(frames)==3
    payload=joblib.load(root/"models/Hardware_joint256_tuned.joblib")
    q,h=archive["quantized"],archive["bank_mean"]
    original=core.features_from_payload(payload,q[index:index+1],h[index:index+1],api)[0]
    permutation=np.random.default_rng(42).permutation(16)
    permuted_h=h[index:index+1].reshape(1,16,16)[:,permutation].reshape(1,256)
    permuted=core.features_from_payload(payload,q[index:index+1],permuted_h,api)[0]
    maxdiff=float(np.max(np.abs(original-permuted)))
    original_pred=int(core.numpy_scores(original[None,:],payload["head"]).argmax())
    permuted_pred=int(core.numpy_scores(permuted[None,:],payload["head"]).argmax())
    matrix=pd.read_csv(root/"confusion_Hardware_joint256_tuned.csv",index_col=0).to_numpy()

    def inputs(ax):
        ax.plot(np.arange(1,17),q[index],"o-",color="#576d8b",linewidth=1.6,markersize=4)
        ax.set(xlim=(.5,16.5),xlabel="Temporal interval (1-16)",ylabel="Quantized frame change",
               title="Real training window and 16 labelled inputs")
        ax.set_ylim(0,max(.2,float(q[index].max())*1.6))
        for j,(frame,idx) in enumerate(zip(frames,chosen_indices)):
            inset=ax.inset_axes([.03+j*.32,.69,.27,.25])
            inset.imshow(frame);inset.axis("off");inset.set_title(f"{idx/row.fps:.2f} s",fontsize=8,pad=1)
        ax.text(.01,.02,row.sample_id,transform=ax.transAxes,fontsize=8)

    def outputs(ax):
        ax.plot(np.arange(1,257),original,color="#258a88",label="Original dimension order",linewidth=1.3)
        ax.plot(np.arange(1,257),permuted,"--",color="#d3883c",alpha=.8,label="Input-only permutation",linewidth=1.)
        ax.set(xlabel="256 full-vector center indices",ylabel="Reconstructed joint RBF feature",
               title=f"Dimension identity matters | max difference = {maxdiff:.3g}")
        ax.legend(fontsize=8,loc="lower right")

    def performance(ax):
        three=[names[0],names[1],names[-1]]
        for k,(split,color) in enumerate((("validation","#d3883c"),("test","#258a88"))):
            bars=ax.bar(np.arange(3)+(k-.5)*.32,table.loc[three,split+"_accuracy"]*100,
                        width=.32,color=color,label=split.capitalize())
            ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=9)
        ax.set(ylim=(0,100),ylabel="Accuracy (%)",xticks=np.arange(3),
               xticklabels=["Linear16","G default","G strict tuned"],title="Fixed official subject holdouts")
        ax.legend(fontsize=9)

    def confusion(ax):
        core.confusion_panel(ax,matrix,"Strict Gaussian selected on validation | test")

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for function,ax in zip((inputs,outputs,performance,confusion),axes.ravel()):function(ax)
    fig.tight_layout();fig.savefig(root/"group_meeting_four_panel.png");plt.close(fig)
    for name,function in (("input_example",inputs),("dimension_identity",outputs),
                          ("performance",performance),("confusion",confusion)):
        fig,ax=plt.subplots(figsize=(8,6));function(ax);fig.tight_layout()
        fig.savefig(root/"figures"/("meeting_panel_"+name+".png"));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,frame,idx in zip(axes,frames,chosen_indices):
        ax.imshow(frame);ax.axis("off");ax.set_title(f"Frame {idx}, t={idx/row.fps:.2f} s")
    fig.suptitle(row.sample_id+" | real training video, not representative of all walking clips",fontsize=9)
    fig.tight_layout();fig.savefig(root/"figures/real_training_frames.png");plt.close(fig)
    core.write_json(root/"presentation_sample.json",{
        "sample_id":row.sample_id,"video":row.path,"selected_frame_indices":chosen_indices,
        "times_seconds":chosen_indices/row.fps,"raw16":archive["raw"][index],"quantized16":q[index],
        "joint_output256":original,"input_only_permutation":permutation,
        "input_only_permutation_feature_max_difference":maxdiff,
        "original_prediction":core.CLASSES[original_pred],"permuted_prediction":core.CLASSES[permuted_pred],
        "input_only_invariance_is_not_required_by_original_model":True,
        "new_frames_for_visualization_only":3,"feature_cache_unchanged":True,
        "decoder_executable":str(decoder),
        "decoder_environment_note":"Original NumPy1/OpenCV environment; classifier fitting used existing Torch/NumPy2 environment.",
        "presentation_render_seconds":time.perf_counter()-start})
    print("Presentation table and real-video panels saved:",root)


if __name__=="__main__":main()
