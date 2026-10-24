#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from collections import Counter
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["walk", "run", "jump", "sit", "stand", "wave"]
LABEL_ZH = {"walk":"走路 walk", "run":"跑步 run", "jump":"跳跃 jump",
            "sit":"坐下 sit", "stand":"起立 stand", "wave":"挥手 wave"}
SEED = 42


def find_split_file(root: Path, cls: str) -> Path:
    hits = list(root.rglob(f"{cls}_test_split1.txt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one split file for {cls}, found {hits}")
    return hits[0]


def find_video(root: Path, cls: str, name: str) -> Path | None:
    direct = root / cls / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    hits = [p for p in hits if p.parent.name == cls]
    return hits[0] if len(hits) == 1 else None


def read_manifest(video_root: Path, split_root: Path) -> pd.DataFrame:
    rows = []
    mapping = {1: "train", 2: "test", 0: "unused"}
    for cls in CLASSES:
        sf = find_split_file(split_root, cls)
        for line in sf.read_text(encoding="utf-8").splitlines():
            name, flag_s = line.rsplit(maxsplit=1)
            flag = int(flag_s)
            path = find_video(video_root, cls, name)
            rows.append({"class": cls, "filename": name, "official_flag": flag,
                         "split": mapping[flag], "path": str(path) if path else "",
                         "exists": path is not None})
    return pd.DataFrame(rows)


def extract_feature(path: Path) -> tuple[np.ndarray | None, dict, str | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, {}, "cannot_open"
    fps = float(cap.get(cv2.CAP_PROP_FPS)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = {"fps": fps, "total_frames": n, "width": width, "height": height,
            "file_bytes": path.stat().st_size}
    if not np.isfinite(fps) or fps <= 0 or n <= 0:
        cap.release(); return None, meta, "invalid_fps_or_frame_count"
    duration = n / fps
    meta["duration_seconds"] = duration
    if duration < 1.0:
        cap.release(); return None, meta, "shorter_than_1_second"
    start = (duration - 1.0) / 2.0
    times = start + np.linspace(0.0, 1.0, 17)
    frames = []
    sampled_indices = []
    for sec in times:
        idx = min(n - 1, max(0, int(round(sec * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release(); return None, meta, f"frame_read_failed_at_{idx}"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        frames.append(gray); sampled_indices.append(idx)
    cap.release()
    arr = np.stack(frames)
    feature = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2)).astype(np.float32)
    meta.update({"sampled_indices": sampled_indices, "sampled_frame_bytes": int(arr.nbytes),
                 "feature_bytes": int(feature.nbytes)})
    return feature, meta, None


def gaussian_sum(x: np.ndarray, sigma: float, centers: np.ndarray | None = None) -> np.ndarray:
    """Apply each scalar Gaussian to all input positions, then sum positions."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    x = np.asarray(x, dtype=np.float64)
    centers = np.linspace(0.0, 1.0, 16) if centers is None else np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 16 or centers.shape != (16,):
        raise ValueError("Expected N x 16 inputs and exactly 16 centers")
    if not np.isfinite(x).all() or not np.isfinite(centers).all():
        raise ValueError("Inputs and centers must be finite")
    response = np.exp(-((x[:, :, None] - centers[None, None, :]) ** 2) / (2.0 * sigma ** 2))
    return response.sum(axis=1)


def setup_plot_font() -> bool:
    from matplotlib import font_manager
    candidates = ["Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Micro Hei", "SimHei"]
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in names:
            plt.rcParams["font.sans-serif"] = [name]; plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot_confusion(cm, title, out, chinese):
    labels = [LABEL_ZH[c] if chinese else c for c in CLASSES]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax, fraction=.046)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
           xlabel="预测类别" if chinese else "Predicted class",
           ylabel="真实类别" if chinese else "True class", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(6):
        for j in range(6): ax.text(j, i, int(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > threshold else "black")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", type=Path, default=Path("data/videos"))
    ap.add_argument("--split-root", type=Path, default=Path("data/splits"))
    ap.add_argument("--output", type=Path, default=Path("outputs/hmdb51_six_split1"))
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    start_wall = time.perf_counter(); proc = psutil.Process()
    manifest = read_manifest(args.video_root, args.split_root)
    manifest.to_csv(out / "official_manifest_all.csv", index=False)
    selected = manifest[manifest.split.isin(["train", "test"])].copy().reset_index(drop=True)
    cache = out / "feature_cache.joblib"
    if cache.exists() and not args.rebuild_cache:
        payload = joblib.load(cache); records = payload["records"]; preprocessing_seconds = 0.0; cache_reused = True
    else:
        records = []; t0 = time.perf_counter()
        for i, row in selected.iterrows():
            base = row.to_dict()
            if not row["exists"]:
                records.append({**base, "included": False, "exclude_reason": "file_missing"}); continue
            ft, meta, err = extract_feature(Path(row["path"]))
            records.append({**base, **meta, "included": err is None, "exclude_reason": err or "",
                            "feature": ft.tolist() if ft is not None else None})
            if (i + 1) % 50 == 0: print(f"preprocessed {i+1}/{len(selected)}", flush=True)
        preprocessing_seconds = time.perf_counter() - t0; cache_reused = False
        joblib.dump({"records": records}, cache, compress=3)
    rec = pd.DataFrame(records)
    serial = rec.copy(); serial["feature"] = serial["feature"].map(lambda x: json.dumps(x) if isinstance(x, list) else "")
    serial.to_csv(out / "sample_manifest_and_features.csv", index=False)
    exclusions = rec[~rec.included].copy(); exclusions.to_csv(out / "exclusions.csv", index=False)
    ok = rec[rec.included].copy().reset_index(drop=True)
    if ok.empty: raise RuntimeError("No readable videos; see exclusions.csv")
    Xraw = np.asarray(ok.feature.tolist(), dtype=np.float64); y = ok["class"].to_numpy()
    splits = ok["split"].to_numpy(); train = splits == "train"; test = splits == "test"
    if Xraw.shape[1] != 16 or len(Xraw) != len(y): raise AssertionError("feature/label alignment failed")
    scale_max = float(Xraw[train].max())
    if not np.isfinite(scale_max) or scale_max <= 0: raise RuntimeError(f"Training global max invalid: {scale_max}")
    Xscaled_unclipped = Xraw / scale_max
    clipped_values = int(np.count_nonzero((Xscaled_unclipped[test] < 0) | (Xscaled_unclipped[test] > 1)))
    test_value_count = int(Xscaled_unclipped[test].size)
    Xscaled = Xscaled_unclipped.copy(); Xscaled[test] = np.clip(Xscaled[test], 0.0, 1.0)
    XB = gaussian_sum(Xscaled, args.sigma)
    if XB.shape != Xscaled.shape: raise AssertionError("Gaussian output must be N x 16")
    np.savez_compressed(out / "features_arrays.npz", raw=Xraw, scaled=Xscaled, gaussian=XB,
                        labels=y, splits=splits)
    (out / "scaling.json").write_text(json.dumps({"training_global_max": scale_max,
        "test_clipped_values": clipped_values, "test_total_values": test_value_count,
        "test_clipped_ratio": clipped_values / test_value_count if test_value_count else 0.0,
        "training_only_fit": True}, indent=2), encoding="utf-8")

    result_rows=[]; predictions = ok[["class","filename","split","path"]].copy()
    models={}
    for name, X in [("A_linear", Xscaled), ("B_gaussian_sum", XB)]:
        model = Pipeline([("standard_scaler", StandardScaler()), ("logistic_regression",
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-6, random_state=SEED))])
        t=time.perf_counter(); model.fit(X[train], y[train]); train_s=time.perf_counter()-t
        t=time.perf_counter(); pred=model.predict(X[test]); predict_s=time.perf_counter()-t
        pred_train=model.predict(X[train]); predictions.loc[test, f"prediction_{name}"]=pred
        predictions.loc[train, f"prediction_{name}"]=pred_train
        cm=confusion_matrix(y[test], pred, labels=CLASSES)
        result_rows.append({"method":name,"train_samples":int(train.sum()),"test_samples":int(test.sum()),
            "accuracy":accuracy_score(y[test],pred),"macro_f1":f1_score(y[test],pred,average="macro",labels=CLASSES),
            "feature_computation_seconds":preprocessing_seconds,"classifier_training_seconds":train_s,
            "prediction_seconds":predict_s,"sigma":args.sigma if name.startswith("B") else None})
        models[name]=model; joblib.dump(model,out/f"model_{name}.joblib")
        chinese_font = setup_plot_font()
        if chinese_font:
            title = "A 基线混淆矩阵" if name.startswith("A") else f"B Gaussian 求和混淆矩阵 (σ={args.sigma})"
        else:
            title = "A baseline confusion matrix" if name.startswith("A") else f"B Gaussian-sum confusion matrix (sigma={args.sigma})"
        plot_confusion(cm, title, out/f"confusion_{name}.png", chinese_font)
        pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f"confusion_{name}.csv")
    predictions.to_csv(out/"predictions_all_samples.csv",index=False)
    metrics=pd.DataFrame(result_rows); metrics.to_csv(out/"metrics.csv",index=False)

    chinese=setup_plot_font(); labels=[LABEL_ZH[c] if chinese else c for c in CLASSES]
    fig,ax=plt.subplots(figsize=(11,6),dpi=180)
    examples=[]
    for cls,label in zip(CLASSES,labels):
        candidates=ok[(ok["split"]=="train")&(ok["class"]==cls)].sort_values("filename")
        row=candidates.iloc[0]; idx=row.name; ax.plot(range(1,17),Xscaled[idx],marker="o",ms=3,label=f"{label}: {row.filename}"); examples.append(row.filename)
    ax.set(xlabel="输入位置 (1–16)" if chinese else "Input position (1–16)",ylabel="缩放后的画面变化量" if chinese else "Scaled frame-change magnitude",title="每类一个固定训练样本的 16 维输入" if chinese else "16-D input: one fixed training sample per class",xticks=range(1,17)); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/"input_examples.png",bbox_inches="tight"); plt.close(fig)
    bins=np.linspace(0,1,31); fig,ax=plt.subplots(figsize=(10,6),dpi=180)
    for cls,label in zip(CLASSES,labels): ax.hist(Xscaled[train & (y==cls)].ravel(),bins=bins,density=True,histtype="step",linewidth=1.8,label=label)
    ax.set(xlabel="缩放后的画面变化量" if chinese else "Scaled frame-change magnitude",ylabel="归一化密度" if chinese else "Normalized density",title="训练集各类输入值分布（统一分箱）" if chinese else "Training input distributions (shared bins)"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/"input_distribution.png",bbox_inches="tight"); plt.close(fig)

    counts=ok.groupby(["class","split"]).size().unstack(fill_value=0).reindex(CLASSES,fill_value=0)
    counts.to_csv(out/"included_counts_by_class.csv")
    excluded_counts=exclusions.groupby(["class","split","exclude_reason"]).size().rename("count").reset_index(); excluded_counts.to_csv(out/"excluded_counts.csv",index=False)
    mean_seconds = preprocessing_seconds / len(selected) if not cache_reused else None
    data_stats={"included_videos":len(ok),"candidate_videos":len(selected),"original_video_bytes":int(ok.file_bytes.fillna(0).sum()),
        "total_duration_seconds":float(ok.duration_seconds.fillna(0).sum()),"total_frames":int(ok.total_frames.fillna(0).sum()),
        "sampled_frame_bytes_total":int(ok.sampled_frame_bytes.fillna(0).sum()),"final_raw_feature_bytes":int(Xraw.nbytes),
        "preprocessing_seconds_this_run":preprocessing_seconds,"cache_reused":cache_reused,
        "mean_preprocessing_seconds_per_candidate_video":mean_seconds,
        "estimated_seconds_for_all_1331_six_class_videos":mean_seconds*1331 if mean_seconds is not None else None,
        "estimate_method":"measured mean per 600 split-1 candidate videos multiplied by 1331 extracted six-class videos"}
    (out/"data_and_cost_stats.json").write_text(json.dumps(data_stats,indent=2),encoding="utf-8")
    peak_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system={"cpu":platform.processor() or platform.machine(),"logical_cpu_count":psutil.cpu_count(),
        "memory_total_bytes":psutil.virtual_memory().total,"gpu_used":False,"peak_process_rss_bytes":int(peak_kb*1024),
        "python":platform.python_version(),"opencv":cv2.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,
        "total_experiment_seconds":time.perf_counter()-start_wall,
        "classifier_n_iter":{"A_linear":models["A_linear"].named_steps["logistic_regression"].n_iter_.tolist(),
                             "B_gaussian_sum":models["B_gaussian_sum"].named_steps["logistic_regression"].n_iter_.tolist()},
        "classifier_converged":all(int(v) < 5000 for m in models.values() for v in m.named_steps["logistic_regression"].n_iter_)}
    (out/"environment_and_runtime.json").write_text(json.dumps(system,indent=2),encoding="utf-8")
    delta = metrics[["accuracy", "macro_f1"]].iloc[1] - metrics[["accuracy", "macro_f1"]].iloc[0]
    summary=f"""# HMDB51 六类预实验说明\n\n- 官方 split 1；类别顺序：{', '.join(CLASSES)}。\n- 实际训练/测试样本：{int(train.sum())}/{int(test.sum())}；排除 {len(exclusions)}。\n- 训练集全局缩放最大值：{scale_max:.8f}；测试截断比例：{clipped_values}/{test_value_count} = {clipped_values/test_value_count:.6%}。\n- A：16维输入；B：16个中心、σ={args.sigma} 的逐标量 Gaussian 响应沿16个输入求和，输出仍为16维。\n- 两组均使用仅在各自训练特征拟合的 StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=5000, tol=1e-6, seed=42)。\n- A 测试准确率 {metrics.iloc[0].accuracy:.4f}，Macro-F1 {metrics.iloc[0].macro_f1:.4f}。\n- B 测试准确率 {metrics.iloc[1].accuracy:.4f}，Macro-F1 {metrics.iloc[1].macro_f1:.4f}。\n- B-A：准确率 {delta.accuracy:+.4f}，Macro-F1 {delta.macro_f1:+.4f}。此变化同时包含 Gaussian 映射与跨位置求和，不应只归因于非线性。\n- 本结论仅适用于所选六类、当前预处理和 σ={args.sigma}。\n\n完整逐样本预测、混淆矩阵、资源成本和排除原因见同目录 CSV/JSON。\n"""
    (out/"实验说明.md").write_text(summary,encoding="utf-8")
    print(metrics.to_string(index=False)); print(f"Results: {out.resolve()}")

if __name__ == "__main__": main()

