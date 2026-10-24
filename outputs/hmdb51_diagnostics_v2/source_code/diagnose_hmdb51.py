#!/usr/bin/env python3
"""Audit the cached HMDB51 experiment, then compare A/B and position-preserving C."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import shlex
import shutil
import sys
import time
import warnings
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_experiment import CLASSES, extract_feature, gaussian_sum, plot_confusion, read_manifest
from tune_gaussian import scale_fit_and_eval, write_json

SEED = 42
SIGMAS = [.01, .02, .05, .1, .2]
REGULARIZATIONS = [.1, 1., 10.]
CENTER_MODES = ["uniform_0_1", "uniform_p01_p99", "quantile_p01_p99"]
SOLVER = {"solver": "lbfgs", "max_iter": 5000, "tol": 1e-6, "random_state": SEED}
RULE = "mean validation Macro-F1 desc; mean validation accuracy desc; Macro-F1 SD asc; C asc; configuration ID asc"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(project, output):
    destination = output / "previous_results_snapshot"
    if destination.exists():
        return
    destination.mkdir()
    hashes = []
    for source in [project / "outputs/hmdb51_six_split1", project / "outputs/hmdb51_gaussian_tuning_v1"]:
        shutil.copytree(source, destination / source.name)
        for path in sorted(source.rglob("*")):
            if path.is_file():
                hashes.append({"original_path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    for name in ["README.md", "requirements.txt", "run_experiment.py", "tune_gaussian.py", "diagnose_hmdb51.py"]:
        shutil.copy2(project / name, destination / name)
    write_json(destination / "original_file_hashes.json", hashes)


def original_results_unchanged(output):
    hashes = json.loads((output / "previous_results_snapshot/original_file_hashes.json").read_text())
    return all(sha256(item["original_path"]) == item["sha256"] for item in hashes)


def load_records(cache):
    records = pd.DataFrame(joblib.load(cache)["records"])
    records["sample_id"] = np.arange(len(records))
    included = records[records.included].copy().reset_index(drop=True)
    x = np.asarray(included.feature.tolist(), dtype=np.float64)
    return records, included, x


def gaussian_response(x, centers, sigma):
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 16 and centers.shape == (16,)
    assert np.isfinite(sigma) and sigma > 0
    response = np.exp(-((x[:, :, None] - centers[None, None, :]) ** 2) / (2 * sigma ** 2))
    assert response.shape == (len(x), 16, 16) and np.isfinite(response).all()
    return response


def centers_from_training(mode, scaled_train):
    if mode == "uniform_0_1":
        centers = np.linspace(0., 1., 16)
    elif mode == "uniform_p01_p99":
        lo, hi = np.quantile(scaled_train, [.01, .99])
        centers = np.linspace(lo, hi, 16)
    elif mode == "quantile_p01_p99":
        centers = np.quantile(scaled_train, np.linspace(.01, .99, 16))
    else:
        raise ValueError(mode)
    assert centers.shape == (16,) and np.isfinite(centers).all()
    assert np.all(np.diff(centers) >= 0) and 0 <= centers.min() <= centers.max() <= 1
    # Repeated/degenerate centers are retained, so feature dimensions never change.
    return centers


def features(x, family, centers=None, sigma=None):
    if family == "A":
        return x
    response = gaussian_response(x, centers, sigma)
    return response.sum(axis=1) if family == "B" else response.reshape(len(x), 256)


def scores(labels, prediction):
    cm = confusion_matrix(labels, prediction, labels=CLASSES)
    accuracy = float(accuracy_score(labels, prediction))
    assert np.isclose(accuracy, np.trace(cm) / cm.sum())
    return {"accuracy": accuracy,
        "macro_f1": float(f1_score(labels, prediction, average="macro", labels=CLASSES, zero_division=0))}


def fit_classifier(x_train, y_train, x_eval, y_eval, regularization):
    model = Pipeline([("standard_scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(C=regularization, **SOLVER))])
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    messages = [str(w.message) for w in caught]
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    iterations = int(model.named_steps["logistic_regression"].n_iter_.max())
    converged = converged and iterations < SOLVER["max_iter"]
    start = time.perf_counter(); train_prediction = model.predict(x_train)
    train_predict_seconds = time.perf_counter() - start
    start = time.perf_counter(); eval_prediction = model.predict(x_eval)
    eval_predict_seconds = time.perf_counter() - start
    train_score, eval_score = scores(y_train, train_prediction), scores(y_eval, eval_prediction)
    scaler = model.named_steps["standard_scaler"]
    assert scaler.n_samples_seen_ == len(x_train)
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))
    assert np.allclose(scaler.var_, x_train.var(axis=0))
    stats = {"train_accuracy": train_score["accuracy"], "train_macro_f1": train_score["macro_f1"],
        "validation_accuracy": eval_score["accuracy"], "validation_macro_f1": eval_score["macro_f1"],
        "converged": converged, "n_iter": iterations, "warnings": json.dumps(messages),
        "classifier_training_seconds": fit_seconds, "train_prediction_seconds": train_predict_seconds,
        "validation_prediction_seconds": eval_predict_seconds}
    return model, train_prediction, eval_prediction, stats


def sequential_sample(row):
    """Independent sequential decoder; cache creation used frame seeks."""
    cap = cv2.VideoCapture(str(row.path))
    assert cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    targets = (duration - 1) / 2 + np.linspace(0, 1, 17)
    indices = [min(frame_count - 1, max(0, round(float(t * fps)))) for t in targets]
    wanted = set(indices)
    color, timestamps = {}, {}
    for index in range(max(indices) + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Sequential decode failed: {row.filename}, index={index}")
        if index in wanted:
            color[index] = frame.copy()
            timestamps[index] = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()
    gray_u8 = np.stack([cv2.resize(cv2.cvtColor(color[i], cv2.COLOR_BGR2GRAY), (64, 64),
                                  interpolation=cv2.INTER_AREA) for i in indices])
    gray_float = gray_u8.astype(np.float32) / 255.
    sequential_feature = np.abs(gray_float[1:] - gray_float[:-1]).mean(axis=(1, 2))
    independent_feature = np.array([sum(abs(float(b) - float(a)) for a, b in
        zip(gray_float[i].ravel(), gray_float[i+1].ravel())) / (64*64) for i in range(16)])
    safe_vs_cache = float(np.max(np.abs(sequential_feature - np.asarray(row.feature))))
    manual_error = float(np.max(np.abs(independent_feature - np.asarray(row.feature))))
    assert safe_vs_cache <= 1e-7 and manual_error <= 1e-7
    assert indices == row.sampled_indices
    sought_feature, _, error = extract_feature(Path(row.path))
    assert error is None and np.max(np.abs(sought_feature - sequential_feature)) <= 1e-7
    return {"color": [cv2.cvtColor(color[i], cv2.COLOR_BGR2RGB) for i in indices],
        "gray": gray_float, "indices": indices, "target_times": targets.tolist(),
        "decoded_timestamps": [timestamps[i] for i in indices],
        "manual_feature_max_error": manual_error, "sequential_cache_max_error": safe_vs_cache}


def contact_sheets(included, output):
    directory = output / "sampling_contact_sheets"
    directory.mkdir(exist_ok=True)
    selections, verification = [], []
    overview = []
    for cls in CLASSES:
        chosen = included[(included["class"] == cls) & (included.split == "train")].sort_values("filename").head(2)
        fig, axes = plt.subplots(6, 6, figsize=(16, 11), dpi=170)
        for ax in axes.ravel(): ax.axis("off")
        for example, (_, row) in enumerate(chosen.iterrows()):
            sample = sequential_sample(row)
            overview.append((cls, row.filename, sample))
            for i, (frame, index, target) in enumerate(zip(sample["color"], sample["indices"], sample["target_times"])):
                ax = axes[example * 3 + i // 6, i % 6]
                ax.imshow(frame)
                ax.set_title(f"#{i+1} index {index}\ntarget {target:.3f}s; frame {index/row.fps:.3f}s", fontsize=7)
            empty = axes[example*3+2, 5]
            empty.text(0, .5, f"{cls} sample {example+1}\n" + "\n".join(row.filename[i:i+32] for i in range(0,len(row.filename),32)),
                       fontsize=8, wrap=True, transform=empty.transAxes)
            grayfig, grayaxes = plt.subplots(3, 6, figsize=(12, 6), dpi=160)
            for ax in grayaxes.ravel(): ax.axis("off")
            for i in range(17):
                grayaxes.ravel()[i].imshow(sample["gray"][i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                grayaxes.ravel()[i].set_title(f"#{i+1} index={sample['indices'][i]} t={sample['indices'][i]/row.fps:.3f}s", fontsize=7)
            grayfig.suptitle(f"Actual 64x64 model input: {row.filename}", fontsize=9)
            grayfig.tight_layout(rect=[0,0,1,.94]); grayfig.savefig(directory / f"{cls}_{example+1}_actual_gray.png", bbox_inches="tight"); plt.close(grayfig)
            selections.append({"class": cls, "example": example+1, "sample_id": int(row.sample_id),
                "filename": row.filename, "split": row.split, "indices": sample["indices"],
                "target_times_seconds": sample["target_times"],
                "frame_index_times_seconds": [i / row.fps for i in sample["indices"]],
                "decoder_timestamps_seconds": sample["decoded_timestamps"],
                "gray_float_dtype": str(sample["gray"].dtype)})
            verification.append({"class": cls, "filename": row.filename,
                "sequential_cache_max_abs_error": sample["sequential_cache_max_error"],
                "independent_scalar_mean_max_abs_error": sample["manual_feature_max_error"]})
        fig.suptitle(f"{cls}: two fixed training examples, chronological 17-frame samples (30 FPS)", fontsize=13)
        fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(directory / f"{cls}_two_training_videos.png", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(12, 5, figsize=(11, 21), dpi=140)
    for r, (cls, filename, sample) in enumerate(overview):
        for c, idx in enumerate([0, 4, 8, 12, 16]):
            axes[r,c].imshow(sample["color"][idx]); axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
            axes[r,c].set_title(f"{cls} #{r%2+1}, frame {idx+1}, {sample['indices'][idx]/30:.3f}s", fontsize=7)
    fig.suptitle("12 training examples: overview only; full 17-frame sheets saved separately", fontsize=10)
    fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(directory / "overview_12_videos.png", bbox_inches="tight"); plt.close(fig)
    write_json(directory / "selected_samples_and_timestamps.json", selections)
    pd.DataFrame(verification).to_csv(output / "independent_video_feature_verification.csv", index=False)
    return verification


def training_diagnostics(included, raw, scaled, output):
    train = included.split.to_numpy() == "train"
    rows = []
    for representation, matrix in [("raw", raw), ("scaled", scaled)]:
        for cls in ["all_training"] + CLASSES:
            selection = train if cls == "all_training" else train & (included["class"].to_numpy() == cls)
            values = matrix[selection].ravel()
            row = {"representation": representation, "class": cls, "videos": int(selection.sum()),
                "values": len(values), "min": values.min(), "max": values.max(), "mean": values.mean(), "std": values.std()}
            row.update({f"p{p:02d}": float(np.percentile(values, p)) for p in [1,5,25,50,75,95,99]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "training_input_statistics.csv", index=False)
    fig, axes = plt.subplots(2,2, figsize=(13, 8), dpi=180)
    histogram_rows = []
    for r, (representation, matrix) in enumerate([("raw", raw), ("scaled", scaled)]):
        maximum = float(matrix[train].max())
        for c, high in enumerate([maximum, float(np.quantile(matrix[train], .99))]):
            bins = np.linspace(0, high if high > 0 else 1., 41)
            for cls in CLASSES:
                values = matrix[train & (included["class"].to_numpy() == cls)].ravel()
                # Density is normalized using all class values, even in the zoomed panel.
                counts, _ = np.histogram(values, bins)
                density = counts / (len(values) * np.diff(bins))
                axes[r,c].stairs(density, bins, label=cls, lw=1.4)
                for index, count in enumerate(counts):
                    histogram_rows.append({"representation": representation, "range": "full" if c == 0 else "training_p99_zoom",
                        "class": cls, "left": bins[index], "right": bins[index+1], "count": int(count), "density": density[index]})
            axes[r,c].set(title=f"Training {representation}: {'full range' if c==0 else 'zoom to training P99'}",
                          xlabel="Frame-change magnitude", ylabel="Density")
            axes[r,c].grid(alpha=.2)
    axes[0,0].legend(ncol=3)
    fig.tight_layout(); fig.savefig(output / "training_input_histograms.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(histogram_rows).to_csv(output / "training_histogram_bins.csv", index=False)
    default = gaussian_sum(scaled[train], .1)
    channel_stats = []
    for j in range(16):
        values = default[:,j]
        channel_stats.append({"channel": j+1, "center": j/15, "mean": values.mean(), "std": values.std(),
            "min": values.min(), "max": values.max(), "range": np.ptp(values),
            "exactly_constant": bool(np.ptp(values) == 0),
            "low_absolute_std_at_1e_6": bool(values.std() <= 1e-6),
            "low_relative_std_at_1e_3": bool(values.std()/(abs(values.mean())+1e-15) <= 1e-3)})
    table = pd.DataFrame(channel_stats); table.to_csv(output / "default_B_training_channel_statistics.csv", index=False)
    # Keep robust quantiles as well: a few outliers can dominate a channel's SD.
    for j in range(16):
        values = default[:,j]
        table.loc[j,"median"] = np.median(values)
        table.loc[j,"p95"] = np.quantile(values,.95)
        table.loc[j,"p99"] = np.quantile(values,.99)
        table.loc[j,"fraction_response_below_1e_6"] = np.mean(values < 1e-6)
    table.to_csv(output / "default_B_training_channel_statistics.csv", index=False)
    fig, axes = plt.subplots(1,2, figsize=(12,4.5), dpi=180)
    axes[0].errorbar(table.center, table["mean"], yerr=table["std"], fmt="o-", capsize=3)
    axes[0].set(title="Default B: training channel mean +/- SD", xlabel="Gaussian center", ylabel="Summed response")
    axes[1].plot(table.center, table["std"], "o-"); axes[1].set_yscale("log")
    axes[1].set(title="Across-training-video SD per output channel", xlabel="Gaussian center", ylabel="SD (log scale)")
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "default_B_channel_variation.png", bbox_inches="tight"); plt.close(fig)
    return rows[1], table


def run_audit(project, baseline, cache, output):
    start = time.perf_counter()
    records, included, raw = load_records(cache)
    checks = []
    def check(name, condition, detail):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            write_json(output / "implementation_checks.json", {"status": "FAILED", "checks": checks})
            raise RuntimeError(f"Audit failed: {name}: {detail}")
    official = read_manifest(project / "data/videos", project / "data/splits")
    selected = official[official.split.isin(["train", "test"])].reset_index(drop=True)
    merged = records.merge(selected, on="filename", suffixes=("_cache", "_official"), validate="one_to_one")
    check("official_filename_label_split_alignment", len(merged) == len(records) == 600 and
        np.array_equal(merged.class_cache, merged.class_official) and
        np.array_equal(merged.split_cache, merged.split_official) and
        np.array_equal(merged.official_flag_cache, merged.official_flag_official), "Compared all 600 candidate records to the six official split-1 files")
    exists = all(Path(row.path).is_file() and Path(row.path).name == row.filename and
                 Path(row.path).parent.name == row["class"] and Path(row.path).stat().st_size == row.file_bytes
                 for _, row in records.iterrows())
    check("video_filename_class_path_size", exists, "All candidate video files checked against cache metadata")
    train = included.split.to_numpy() == "train"; test = ~train
    train_names, test_names = set(included.loc[train,"filename"]), set(included.loc[test,"filename"])
    check("no_train_test_video_filename_overlap", not train_names & test_names, "Compared filenames and resolved filesystem paths")
    paths_train = {str(Path(p).resolve()) for p in included.loc[train,"path"]}
    paths_test = {str(Path(p).resolve()) for p in included.loc[test,"path"]}
    check("no_train_test_path_overlap", not paths_train & paths_test, "Official training and test files remain separate")
    sample_manifest = included.drop(columns=["feature", "sampled_indices"]).copy()
    sample_manifest["video_sha256"] = [sha256(path) for path in included.path]
    content_train, content_test = set(sample_manifest.loc[train,"video_sha256"]), set(sample_manifest.loc[test,"video_sha256"])
    check("no_train_test_identical_video_bytes", not content_train & content_test, "SHA-256 content comparison across the included training/test videos")
    sample_manifest.to_csv(output / "sample_manifest.csv", index=False)
    records.drop(columns=["feature"]).to_csv(output / "candidate_manifest_all_600.csv", index=False)
    records[~records.included].to_csv(output / "exclusions.csv", index=False)
    previous = np.load(baseline / "features_arrays.npz", allow_pickle=True)
    check("cached_array_label_and_split_alignment", np.array_equal(raw, previous["raw"]) and
        np.array_equal(included["class"], previous["labels"]) and np.array_equal(included.split, previous["splits"]),
        "Compared the original saved arrays to the cache row by row")
    check("raw_feature_shape_and_validity", raw.shape == (588,16) and np.isfinite(raw).all() and
          raw.min() >= 0 and raw.max() <= 1, f"shape={raw.shape}; NaN/Inf={int((~np.isfinite(raw)).sum())}; all-zero samples={int(np.all(raw==0,axis=1).sum())}")
    np.savez_compressed(output / "raw_cached_features.npz", raw=raw, sample_ids=included.sample_id.to_numpy(),
                        labels=included["class"].to_numpy(), splits=included.split.to_numpy())
    fit, evaluation, scale = scale_fit_and_eval(raw[train], raw[test])
    scaled = np.empty_like(raw); scaled[train], scaled[test] = fit, evaluation
    check("global_scale_training_only", scale["training_global_max"] == raw[train].max() and
        np.array_equal(scaled, previous["scaled"]), "Global maximum recomputed from all training entries only; test clipping reproduced")
    write_json(output / "original_scaling_verification.json", scale)
    temporal = []
    for _, row in included.iterrows():
        target = (row.duration_seconds-1)/2 + np.linspace(0,1,17)
        expected = [min(int(row.total_frames)-1,max(0,round(float(t*row.fps)))) for t in target]
        actual = list(row.sampled_indices)
        check_condition = expected == actual and len(actual) == 17 and np.all(np.diff(actual)>=0)
        if not check_condition: check("time_sampling_formula", False, row.filename)
        duplicates = [int(i) for i in sorted(set(actual)) if actual.count(i) > 1]
        temporal.append({"sample_id": row.sample_id, "class": row["class"], "filename": row.filename, "split": row.split,
            "fps": row.fps, "requested_window_start_seconds": float(target[0]), "requested_window_end_seconds": float(target[-1]),
            "requested_span_seconds": float(target[-1]-target[0]), "first_frame_index": actual[0], "last_frame_index": actual[-1],
            "actual_frame_span_seconds": (actual[-1]-actual[0])/row.fps,
            "sampled_indices": json.dumps(actual), "repeated_frame_indices": json.dumps(duplicates),
            "repeat_count": len(actual)-len(set(actual)),
            "max_target_frame_time_error_seconds": float(np.max(np.abs(np.array(actual)/row.fps-target)))})
    temporal = pd.DataFrame(temporal); temporal.to_csv(output / "sampling_span_and_repeated_indices.csv", index=False)
    check("middle_one_second_fps_sampling", True, f"All 588 schedules verified; actual span {temporal.actual_frame_span_seconds.min():.6f} to {temporal.actual_frame_span_seconds.max():.6f}s; videos with repeated indices={int((temporal.repeat_count>0).sum())}")
    print("Structural cache/split/sampling checks passed; decoding 12 training examples independently...", flush=True)
    independent = contact_sheets(included, output)
    check("float_before_subtraction_and_independent_video_recompute", True,
        f"12 videos decoded sequentially; float32 subtraction and independent scalar absolute-difference mean match cache; max error={max(v['independent_scalar_mean_max_abs_error'] for v in independent):.3g}")
    centers = np.linspace(0,1,16); small = scaled[train][:7]
    response = gaussian_response(small, centers, .1)
    literal = np.array([[[math.exp(-((float(x)-float(mu))**2)/(2*.1**2)) for mu in centers] for x in sample] for sample in small])
    summed = response.sum(axis=1)
    permutation = np.random.default_rng(SEED).permutation(16)
    max_error = float(np.max(np.abs(literal-response)))
    perm_error = float(np.max(np.abs(summed-gaussian_sum(small[:,permutation],.1))))
    check("Gaussian_formula_response_sum_dimensions_and_peak", response.shape==(7,16,16) and summed.shape==(7,16) and
        np.allclose(response,literal,atol=1e-14) and np.array_equal(summed,gaussian_sum(small,.1)) and
        np.all(np.diagonal(gaussian_response(centers[None,:],centers,.1),axis1=1,axis2=2)==1),
        f"Independent math.exp response error={max_error:.3g}; response summed over input position axis; peak=1")
    check("B_permutation_invariance", perm_error < 1e-12, f"Maximum absolute error after input-position permutation={perm_error:.3g}")
    check("C_flatten_order_consistent_with_B", np.array_equal(response.reshape(7,256).reshape(7,16,16).sum(axis=1),summed),
          "C column = input_position*16 + Gaussian_center_index (row-major order)")
    previous_predictions = pd.read_csv(baseline / "predictions_all_samples.csv")
    baseline_metrics, counts = [], []
    for family, name, matrix in [("A","A_linear",scaled),("B","B_gaussian_sum",gaussian_sum(scaled,.1))]:
        model = joblib.load(baseline / f"model_{name}.joblib")
        scaler = model.named_steps["standard_scaler"]
        scaler_ok = scaler.n_samples_seen_ == train.sum() and np.allclose(scaler.mean_,matrix[train].mean(axis=0)) and np.allclose(scaler.var_,matrix[train].var(axis=0))
        check(f"{family}_standard_scaler_training_only", scaler_ok, f"Fitted sample count={scaler.n_samples_seen_}")
        classifier = model.named_steps["logistic_regression"]
        check(f"{family}_classifier_iteration_check", classifier.n_iter_.max() < classifier.max_iter,
            f"n_iter={classifier.n_iter_.tolist()}, max_iter={classifier.max_iter}; original saved runtime also records convergence")
        old_counts = pd.read_csv(baseline / f"confusion_{name}.csv", index_col=0).to_numpy()
        for split, mask in [("train",train),("test",test)]:
            labels = included.loc[mask,"class"].to_numpy(); prediction = model.predict(matrix[mask])
            expected = included.loc[mask,["filename"]].merge(previous_predictions[["filename",f"prediction_{name}"]],on="filename",validate="one_to_one")[f"prediction_{name}"].to_numpy()
            check(f"{family}_{split}_saved_predictions_match", np.array_equal(prediction,expected), "All predictions aligned by filename")
            sc = scores(labels,prediction)
            if split == "test": check(f"{family}_accuracy_confusion_diagonal_agree", np.array_equal(confusion_matrix(labels,prediction,labels=CLASSES),old_counts), "Recomputed accuracy agrees with saved confusion-matrix diagonal")
            baseline_metrics.append({"method": f"{family}_default", "split":split,"samples":int(mask.sum()),**sc})
            for cls in CLASSES: counts.append({"method":f"{family}_default","split":split,"class":cls,"predicted_count":int(np.sum(prediction==cls)),"true_count":int(np.sum(labels==cls))})
    pd.DataFrame(baseline_metrics).to_csv(output / "original_AB_train_test_metrics.csv",index=False)
    pd.DataFrame(counts).to_csv(output / "original_AB_prediction_counts.csv",index=False)
    training_diagnostics(included,raw,scaled,output)
    check("previous_results_preserved", original_results_unchanged(output), "Both old result directories remain byte-for-byte unchanged; backup stored in the new directory")
    audit = {"status":"PASSED", "cache_sha256":sha256(cache), "checks":checks,
        "feature_affecting_errors_found":False,"original_features_and_splits_changed":False,
        "sampling_example_count":12,"train_samples":int(train.sum()),"test_samples":int(test.sum()),
        "runtime_seconds":time.perf_counter()-start,
        "limitations":["Independent frame decoding covers 12 fixed training examples; all 588 cached sampling schedules are checked.",
            "Actual timestamp endpoints are quantized to video frames; see sampling_span_and_repeated_indices.csv.",
            "No verified original-source ID table is available; subsequent CV uses stratification, not guessed source groups."]}
    write_json(output / "implementation_checks.json",audit)
    print(f"AUDIT PASSED ({len(checks)} checks), {audit['runtime_seconds']:.2f}s",flush=True)


def cid(family, center_mode, sigma, reg):
    return f"{family}__{center_mode}__sigma_{sigma:g}__C_{reg:g}"


def rank_configurations(summary, family):
    return summary[(summary.family==family) & summary.all_folds_converged].sort_values(
        ["mean_validation_macro_f1","mean_validation_accuracy","std_validation_macro_f1","regularization_C","config_id"],
        ascending=[False,False,True,True,True]).reset_index(drop=True)


def summarize_cv(results, predictions):
    rows = []
    for config, frame in results.groupby("config_id",sort=False):
        row = {k:frame.iloc[0][k] for k in ["family","center_mode","sigma","regularization_C","feature_dimensions"]}
        row.update({"config_id":config,"all_folds_converged":bool(frame.converged.all()),"folds":len(frame)})
        for metric in ["train_accuracy","train_macro_f1","validation_accuracy","validation_macro_f1"]:
            row[f"mean_{metric}"] = float(frame[metric].mean())
            row[f"std_{metric}"] = float(frame[metric].std(ddof=1))
        val=predictions[(predictions.config_id==config)&(predictions.role=="validation")]
        sc=scores(val.true_class,val.predicted_class)
        row.update({"pooled_validation_accuracy":sc["accuracy"],"pooled_validation_macro_f1":sc["macro_f1"],
                    "maximum_classifier_iterations":int(frame.n_iter.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def run_cv(training, output):
    raw=np.asarray(training.feature.tolist(),dtype=np.float64); labels=training["class"].to_numpy()
    splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    splits=list(splitter.split(raw,labels)); results=[]; predictions=[]; fold_data=[]
    folder=output/"cv"; folder.mkdir(exist_ok=True)
    fold_manifest=[]; fold_counts=[]; center_log=[]; scales=[]
    for fold,(fit_idx,val_idx) in enumerate(splits,1):
        fit,val,scaling=scale_fit_and_eval(raw[fit_idx],raw[val_idx]); scales.append({"fold":fold,**scaling})
        for role,indices in [("train",fit_idx),("validation",val_idx)]:
            for index in indices:
                row=training.iloc[index]
                fold_manifest.append({"fold":fold,"role":role,"sample_id":int(row.sample_id),"filename":row.filename,"class":row["class"],"official_split":"train"})
            for cls in CLASSES: fold_counts.append({"fold":fold,"role":role,"class":cls,"samples":int(np.sum(labels[indices]==cls))})
        fold_data.append((fit_idx,val_idx,fit,val))
        saved={"raw_train":raw[fit_idx],"raw_validation":raw[val_idx],"scaled_train":fit,"scaled_validation":val,
            "training_sample_ids":training.iloc[fit_idx].sample_id.to_numpy(),"validation_sample_ids":training.iloc[val_idx].sample_id.to_numpy()}
        plans=[("A","none",0.,reg) for reg in REGULARIZATIONS]
        plans += [("B",mode,sigma,reg) for mode in CENTER_MODES for sigma in SIGMAS for reg in REGULARIZATIONS]
        for family,mode,sigma,reg in plans:
            centers=None if family=="A" else centers_from_training(mode,fit)
            if family=="B" and reg==REGULARIZATIONS[0]:
                center_log.append({"fold":fold,"center_mode":mode,"sigma":sigma,"centers":centers.tolist(),
                    "unique_centers":len(np.unique(centers)),"handling":"retain repeated centers without jitter; always 16 curves"})
            start=time.perf_counter(); ft=features(fit,family,centers,sigma); fv=features(val,family,centers,sigma)
            transform_seconds=time.perf_counter()-start
            model,tp,vp,stats=fit_classifier(ft,labels[fit_idx],fv,labels[val_idx],reg)
            config=cid(family,mode,sigma,reg)
            results.append({"fold":fold,"config_id":config,"family":family,"center_mode":mode,"sigma":sigma,
                "regularization_C":reg,"feature_dimensions":ft.shape[1],"train_samples":len(fit_idx),"validation_samples":len(val_idx),
                "feature_computation_seconds":transform_seconds,**stats})
            for role,idxs,pred in [("train",fit_idx,tp),("validation",val_idx,vp)]:
                for index,p in zip(idxs,pred): predictions.append({"fold":fold,"role":role,"config_id":config,"sample_id":int(training.iloc[index].sample_id),"true_class":labels[index],"predicted_class":p})
            if family=="B" and reg==REGULARIZATIONS[0]:
                saved[f"B__{mode}__sigma_{sigma:g}_train"]=ft; saved[f"B__{mode}__sigma_{sigma:g}_validation"]=fv
        np.savez_compressed(folder/f"fold_{fold}_AB_features.npz",**saved)
        print(f"A/B CV fold {fold}/3 completed: {len(fit_idx)} train / {len(val_idx)} validation",flush=True)
    pd.DataFrame(fold_manifest).to_csv(folder/"fold_sample_manifest.csv",index=False)
    pd.DataFrame(fold_counts).to_csv(folder/"fold_class_counts.csv",index=False)
    write_json(folder/"fold_training_only_scaling.json",scales)
    write_json(folder/"fold_Gaussian_centers.json",center_log)
    detail=pd.DataFrame(results); predframe=pd.DataFrame(predictions)
    summary=summarize_cv(detail,predframe)
    best_A=rank_configurations(summary,"A").iloc[0]; best_B=rank_configurations(summary,"B").iloc[0]
    def selection(row): return {"center_mode":row.center_mode,"sigma":float(row.sigma),"regularization_C":float(row.regularization_C),"config_id":row.config_id}
    frozen={"A":selection(best_A),"B":selection(best_B),"selection_rule":RULE,"test_used_for_selection":False,
            "AB_freeze_time_local":time.strftime("%Y-%m-%d %H:%M:%S %z")}
    write_json(output/"AB_selection_frozen_before_C_and_test.json",frozen)
    for fold,(fit_idx,val_idx,fit,val) in enumerate(fold_data,1):
        centers=centers_from_training(best_B.center_mode,fit); sigma=float(best_B.sigma)
        start=time.perf_counter(); ft=features(fit,"C",centers,sigma); fv=features(val,"C",centers,sigma)
        transform_seconds=time.perf_counter()-start
        assert ft.shape==(len(fit_idx),256) and fv.shape==(len(val_idx),256)
        assert np.allclose(ft.reshape(-1,16,16).sum(axis=1),gaussian_sum(fit,sigma,centers))
        np.savez_compressed(folder/f"fold_{fold}_C_features.npz",gaussian_flat_train=ft,gaussian_flat_validation=fv,centers=centers,
            training_sample_ids=training.iloc[fit_idx].sample_id.to_numpy(),validation_sample_ids=training.iloc[val_idx].sample_id.to_numpy())
        for reg in REGULARIZATIONS:
            model,tp,vp,stats=fit_classifier(ft,labels[fit_idx],fv,labels[val_idx],reg)
            config=cid("C",best_B.center_mode,sigma,reg)
            results.append({"fold":fold,"config_id":config,"family":"C","center_mode":best_B.center_mode,"sigma":sigma,"regularization_C":reg,
                "feature_dimensions":256,"train_samples":len(fit_idx),"validation_samples":len(val_idx),"feature_computation_seconds":transform_seconds,**stats})
            for role,idxs,pred in [("train",fit_idx,tp),("validation",val_idx,vp)]:
                for index,p in zip(idxs,pred): predictions.append({"fold":fold,"role":role,"config_id":config,"sample_id":int(training.iloc[index].sample_id),"true_class":labels[index],"predicted_class":p})
    detail=pd.DataFrame(results); predframe=pd.DataFrame(predictions); summary=summarize_cv(detail,predframe)
    best_C=rank_configurations(summary,"C").iloc[0]; frozen["C"]=selection(best_C)
    frozen["all_freeze_time_local"]=time.strftime("%Y-%m-%d %H:%M:%S %z")
    write_json(output/"all_parameters_frozen_before_test.json",frozen)
    detail.to_csv(folder/"all_fold_metrics.csv",index=False); summary.to_csv(folder/"all_configuration_summary.csv",index=False)
    predframe.to_csv(folder/"all_fold_predictions.csv",index=False)
    for family in ["A","B","C"]: rank_configurations(summary,family).to_csv(folder/f"{family}_validation_ranking.csv",index=False)
    detail[~detail.converged].to_csv(folder/"non_converged_fits.csv",index=False)
    return frozen,summary,detail


def final_evaluation(included, raw, frozen, output):
    labels=included["class"].to_numpy(); train=included.split.to_numpy()=="train"; test=~train
    fit,evaluation,scaling=scale_fit_and_eval(raw[train],raw[test])
    write_json(output/"final_training_only_scaling.json",scaling)
    plans=[("A_default","A",{"center_mode":"none","sigma":0.,"regularization_C":1.}),
           ("B_default","B",{"center_mode":"uniform_0_1","sigma":.1,"regularization_C":1.}),
           ("A_tuned","A",frozen["A"]),("B_tuned","B",frozen["B"]),("C_no_sum_diagnostic","C",frozen["C"])]
    rows=[]; predictions=[]; counts=[]; parameters=[]; arrays={"raw":raw,"labels":labels,"official_splits":included.split.to_numpy(),"sample_ids":included.sample_id.to_numpy()}
    models=output/"models"; models.mkdir(exist_ok=True)
    for name,family,config in plans:
        mode=config["center_mode"]; sigma=config["sigma"]; reg=config["regularization_C"]
        centers=None if family=="A" else centers_from_training(mode,fit)
        start=time.perf_counter(); ft=features(fit,family,centers,sigma); fe=features(evaluation,family,centers,sigma)
        transform_seconds=time.perf_counter()-start
        model,tp,ep,stats=fit_classifier(ft,labels[train],fe,labels[test],reg)
        row={"method":name,"family":family,"dimensions":ft.shape[1],"center_mode":mode,"sigma":sigma,
            "regularization_C":reg,"train_samples":int(train.sum()),"test_samples":int(test.sum()),
            "train_accuracy":stats["train_accuracy"],"train_macro_f1":stats["train_macro_f1"],
            "test_accuracy":stats["validation_accuracy"],"test_macro_f1":stats["validation_macro_f1"],
            "feature_computation_seconds":transform_seconds,"classifier_training_seconds":stats["classifier_training_seconds"],
            "test_prediction_seconds":stats["validation_prediction_seconds"],"converged":stats["converged"],"n_iter":stats["n_iter"],"warnings":stats["warnings"]}
        rows.append(row)
        whole=np.empty((len(raw),ft.shape[1]));whole[train],whole[test]=ft,fe; arrays[name]=whole
        joblib.dump({"classifier":model,"input_scale":scaling,"family":family,"centers":centers,"sigma":sigma,
            "regularization_C":reg,"class_order":CLASSES,"flatten_order":"input position first; Gaussian center second (C row-major)"},models/f"{name}.joblib")
        parameters.append({"method":name,"family":family,"classifier":{**SOLVER,"C":reg,"penalty":"l2","fit_intercept":True},
            "center_mode":mode,"sigma":sigma,"centers":None if centers is None else centers.tolist(),
            "unique_center_count":None if centers is None else len(np.unique(centers)),"dimensions":ft.shape[1]})
        for split,mask,pred in [("train",train,tp),("test",test,ep)]:
            for idx,p in zip(np.flatnonzero(mask),pred): predictions.append({"method":name,"split":split,"sample_id":int(included.iloc[idx].sample_id),
                "filename":included.iloc[idx].filename,"true_class":labels[idx],"predicted_class":p,"correct":bool(labels[idx]==p)})
            cm=confusion_matrix(labels[mask],pred,labels=CLASSES)
            pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(output/f"confusion_{name}_{split}.csv")
            plot_confusion(cm,f"{name}: {split}",output/f"confusion_{name}_{split}.png",False)
            for cls in CLASSES: counts.append({"method":name,"split":split,"class":cls,"predicted_count":int(np.sum(pred==cls)),"true_count":int(np.sum(labels[mask]==cls))})
    metrics=pd.DataFrame(rows);metrics.to_csv(output/"final_default_and_tuned_ABC_metrics.csv",index=False)
    write_json(output/"final_default_and_tuned_ABC_metrics.json",metrics.to_dict("records"))
    pd.DataFrame(predictions).to_csv(output/"all_final_predictions.csv",index=False)
    pd.DataFrame(counts).to_csv(output/"all_final_prediction_counts.csv",index=False)
    write_json(output/"final_model_parameters.json",parameters)
    np.savez_compressed(output/"final_features.npz",**arrays)
    return metrics


def parameter_plots(summary, metrics, output):
    b=summary[summary.family=="B"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),dpi=180,sharey=True)
    for ax,mode in zip(axes,CENTER_MODES):
        for reg in REGULARIZATIONS:
            part=b[(b.center_mode==mode)&(b.regularization_C==reg)].sort_values("sigma")
            ax.errorbar(part.sigma,part.mean_validation_macro_f1,yerr=part.std_validation_macro_f1,
                        marker="o",capsize=3,label=f"Classifier C={reg:g}")
        ax.set(title=mode,xlabel="Gaussian sigma (log scale)",xscale="log")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Validation Macro-F1 (mean +/- fold SD)")
    fig.suptitle("Training-only stratified 3-fold validation; identical folds for every configuration")
    fig.tight_layout();fig.savefig(output/"Gaussian_parameter_comparison.png",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=180)
    for ax,metric in zip(axes,["accuracy","macro_f1"]):
        x=np.arange(len(metrics));ax.bar(x-.18,metrics[f"train_{metric}"],width=.36,label="Train")
        ax.bar(x+.18,metrics[f"test_{metric}"],width=.36,label="Test")
        ax.set_xticks(x,metrics.method,rotation=25,ha="right")
        ax.set(title=metric,ylim=(0,max(.5,float(metrics[f'train_{metric}'].max())+.05)))
        ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(output/"train_test_ABC_comparison.png",bbox_inches="tight");plt.close(fig)


def write_report(output, metrics, summary, runtime):
    audit=json.loads((output/"implementation_checks.json").read_text())
    input_stats=pd.read_csv(output/"training_input_statistics.csv")
    train_values=input_stats[(input_stats.representation=="scaled")&(input_stats["class"]=="all_training")].iloc[0]
    channels=pd.read_csv(output/"default_B_training_channel_statistics.csv")
    lines=["| 方案 | 维度 | 分类器 C | 训练 Accuracy | 训练 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in metrics.iterrows():lines.append(f"| {r.method} | {r.dimensions} | {r.regularization_C:g} | {r.train_accuracy:.2%} | {r.train_macro_f1:.4f} | {r.test_accuracy:.2%} | {r.test_macro_f1:.4f} |")
    cvlines=["| 方案 | 中心 | σ | 分类器 C | 验证 Macro-F1 均值±SD | 验证 Accuracy 均值±SD |",
             "|---|---|---:|---:|---:|---:|"]
    for family in ["A","B","C"]:
        r=rank_configurations(summary,family).iloc[0]
        cvlines.append(f"| {family} | {r.center_mode} | {r.sigma:g} | {r.regularization_C:g} | {r.mean_validation_macro_f1:.4f} ± {r.std_validation_macro_f1:.4f} | {r.mean_validation_accuracy:.2%} ± {r.std_validation_accuracy:.2%} |")
    b0=metrics[metrics.method=="B_default"].iloc[0];bt=metrics[metrics.method=="B_tuned"].iloc[0];ct=metrics[metrics.family=="C"].iloc[0]
    cv_c = rank_configurations(summary,"C").iloc[0]
    cv_b = rank_configurations(summary,"B").iloc[0]
    convergence=pd.read_csv(output/"cv/all_fold_metrics.csv")
    text=f"""# HMDB51 六类：实现检查与低准确率诊断

## 已验证事实

本轮沿用 walk、run、jump、sit、stand、wave 的官方 split 1，原缓存中成功读取 412 训练/176 测试，未更换数据、扩大类别或重新下载。旧代码和两轮结果已备份到 previous_results_snapshot，旧结果文件的 SHA-256 在本轮结束时再次核对。

实现检查状态：{audit['status']}。未发现会改变当前特征或 A/B 预测的实现错误，未修改采样、特征、划分或默认 Gaussian 公式。检查涵盖全部600候选记录的文件/标签/官方划分、588个纳入视频的内容哈希与采样索引、原特征/标签行对应、训练拟合的缩放和 StandardScaler、Gaussian独立计算及输入排列不变性、收敛/预测/混淆矩阵一致性。12段固定训练示例另以顺序解码核对原来按帧定位的结果，并独立逐像素求绝对差均值；完整逐项证据在 implementation_checks.json。独立解码仅覆盖12段示例，并非重新解码全部视频。

采样计划是在每段视频中间固定1秒上取17个时间点，再按实际30 FPS定位；实际离散帧跨度与重复索引在 sampling_span_and_repeated_indices.csv，不能把17个相邻原始帧当成1秒。灰度64×64帧先转float32并除以255，再相减。采样接触图保存每类2段训练视频的17帧彩色内容与实际64×64灰度输入，注明文件名、帧索引和时间；这些示例不代表整类。

训练输入（全局最大值缩放后）的中位数 {train_values.p50:.5f}、P95={train_values.p95:.5f}、P99={train_values.p99:.5f}，最小/最大/均值/标准差及1/5/25/50/75/95/99分位数含原始和缩放两个版本，见 training_input_statistics.csv。直方图每类使用相同分箱，缩放和统计均未用测试数据。默认B中严格常数通道 {int(channels.exactly_constant.sum())}/16；绝对标准差≤1e-6的通道 {int(channels.low_absolute_std_at_1e_6.sum())}/16。单通道变化小不能单独证明它无用；本轮保留全部通道。

## 参数选择方法

当前数据未提供经验证的源视频ID表；文件名前缀不足以验证所有实际来源别名，所以本轮依用户要求采用固定seed=42的分层3折，没有把上轮启发式分组当成可靠ID。可能有同源或同场景片段跨折，验证表现可能偏乐观。所有方案使用相同折；每折从原始16维开始，缩放最大值、P1/P99范围、分位数中心、StandardScaler都只用该折训练部分拟合。验证超界截断数量与比例逐折记录。

A 比较分类器C=0.1、1、10；B比较3种中心×5种σ×3种分类器C，共45组。中心策略严格为：均匀[0,1]、均匀训练P1至P99、训练1%至99%等间隔分位点。σ=0.01、0.02、0.05、0.1、0.2，每个配置16条曲线共用σ且峰值1。重合中心原样保留，退化范围不加抖动、不静默改维度，实际中心逐折保存。

提前固定的排名规则：{RULE}。不收敛配置不参与选优，全部警告/迭代数仍保留。共 {len(convergence)} 次CV拟合，其中未收敛 {int((~convergence.converged).sum())} 次。

C采用B验证选出的中心策略和σ，逐折仅用训练子集拟合中心；把R[n,i,j]按输入位置i优先、中心j其次展开256维，仅选择分类器C=0.1、1、10。C属于软件诊断。B/C比较同时改变位置保留、维度和表达能力，不能把差异全部归因于顺序。C的Gaussian是依据同一批折选出的，因此此处CV不是嵌套交叉验证的无偏性能估计。

{chr(10).join(cvlines)}

## 锁定参数后正式测试

全部参数在 all_parameters_frozen_before_test.json 保存后，再使用完整412训练样本重拟合并评价176测试视频。测试集未参与本轮参数选择；它在前两轮已查看过，所以本轮属于探索性诊断。

{chr(10).join(lines)}

调参B相对默认B：测试Accuracy {(bt.test_accuracy-b0.test_accuracy)*100:+.2f}个百分点、Macro-F1 {bt.test_macro_f1-b0.test_macro_f1:+.4f}。C相对调参B：测试Accuracy {(ct.test_accuracy-bt.test_accuracy)*100:+.2f}个百分点、Macro-F1 {ct.test_macro_f1-bt.test_macro_f1:+.4f}。这就是本轮观察到的变化，不能据此承诺其他划分或任务上的提升。

## 解释边界与下一步

- 是否发现实现错误：在已检查范围内没有发现影响当前输入与结果的错误。无数组行错位、uint8差分溢出、训测文件重叠或高斯求和轴错误的证据。
- 调中心和宽度是否有帮助：本轮按验证Macro-F1选出的B在测试集的两项指标都低于默认B，不支持“只要换高斯参数就能解决低准确率”。参数有敏感性与泛化稳定提升是两件事。
- 保留位置的256维是否有帮助：C的测试两项指标高于调参B，但它同时增加了维度和表达能力，且B/C分别选择了各自的正则化。C折内训练准确率均值 {cv_c.mean_train_accuracy:.2%}，验证准确率均值 {cv_c.mean_validation_accuracy:.2%}；B的验证准确率为 {cv_b.mean_validation_accuracy:.2%}，即C的验证提升只有 {(cv_c.mean_validation_accuracy-cv_b.mean_validation_accuracy)*100:.2f}个百分点。两者合并交叉验证预测分别正确 {round(cv_c.pooled_validation_accuracy*412)} 和 {round(cv_b.pooled_validation_accuracy*412)} / 412。完整训练后C的训练/测试准确率为 {ct.train_accuracy:.2%}/{ct.test_accuracy:.2%}。这个差距提示过拟合或场景差异值得检查，尚不能认定C已稳定解决问题，也不能将提升全归因于时间顺序。
- 下一步优先级：优先检查“中间1秒窗口+整幅平均帧差”是否抓到了目标动作，同时保留位置对照用于后续研究求和的信息损失。原始A/B的训练准确率也只有25%至27%，而C的验证依然不高，当前证据更值得优先检查输入表示、泛化和聚合方式，而不是继续大范围盲搜Gaussian参数。

Gaussian对同样的16个标量进行确定性变换，B还将位置求和；它不能恢复平均帧差阶段已经丢失的空间动作信息。上述原因解释中，“过拟合”和“场景差异”是提示和待验证可能性，不是已单独排除其他因素的因果结论。

本轮保留原预处理，后续是否改变窗口或输入，应由本轮训练/验证/测试结果和采样接触图共同决定。结果只适用于这六类、当前输入和模型，不说明器件不可行，也不能外推到所有视频任务。

## 成本与复现

初次审计和生成接触图 {audit['runtime_seconds']:.2f}秒；参数比较、最终拟合及绘图 {runtime['experiment_seconds']:.2f}秒；复用原始特征缓存，没有新增下载或解压。后续图像排版和独立结果核查耗时分别见 runtime.json 和 post_run_verification.json；原特征提取耗时沿用原实验记录。实验进程峰值RSS {runtime['peak_process_rss_bytes']/1024**2:.1f} MiB。CPU计算，数值库1线程，未使用GPU。

```bash
cd /home/ziyan/FYP_XZH
.venv/bin/python diagnose_hmdb51.py --stage audit
.venv/bin/python diagnose_hmdb51.py --stage experiment
```

模型、参数、逐样本预测、输入及Gaussian输出、每折清单/结果、混淆矩阵均在本目录；cv/下保存所有配置和每折训练表现。查看 sampling_contact_sheets/ 的12段实际输入与 sampling_visual_review.md 的逐示例人工观察。
"""
    (output/"诊断实验说明.md").write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["audit","experiment","all"],default="all")
    parser.add_argument("--output",type=Path,default=Path("outputs/hmdb51_diagnostics_v2"))
    args=parser.parse_args();project=Path(__file__).resolve().parent
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    baseline=project/"outputs/hmdb51_six_split1";cache=baseline/"feature_cache.joblib"
    snapshot(project,output)
    with (output/"run_commands.txt").open("a",encoding="utf-8") as f:f.write(shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+"\n")
    with threadpool_limits(limits=1):
        if args.stage in ("audit","all"):
            run_audit(project,baseline,cache,output)
        if args.stage in ("experiment","all"):
            audit=json.loads((output/"implementation_checks.json").read_text())
            if audit["status"]!="PASSED" or audit["cache_sha256"]!=sha256(cache):
                raise RuntimeError("A passing audit for the unchanged feature cache is required")
            start=time.perf_counter();records,included,raw=load_records(cache)
            config={"seed":SEED,"folds":3,"splitter":"StratifiedKFold(shuffle=True)","reliable_source_IDs_available":False,
                "center_modes":CENTER_MODES,"sigmas":SIGMAS,"classifier_C":REGULARIZATIONS,"solver":SOLVER,
                "selection_rule":RULE,"test_used_for_selection":False,"Gaussian_peak":1,"Gaussian_curves":16,
                "C_is_software_diagnostic_only":True,"blas_threads":1,"cache_sha256":sha256(cache)}
            write_json(output/"experiment_config.json",config)
            training=included[included.split=="train"].copy().reset_index(drop=True)
            frozen,summary,detail=run_cv(training,output)
            print("Parameters frozen before test:",frozen,flush=True)
            metrics=final_evaluation(included,raw,frozen,output)
            parameter_plots(summary,metrics,output)
            assert original_results_unchanged(output)
            runtime={"experiment_seconds":time.perf_counter()-start,"CV_classifier_fits":len(detail),"final_classifier_fits":len(metrics),
                "peak_process_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                "logical_CPU_count":psutil.cpu_count(),"memory_total_bytes":psutil.virtual_memory().total,"GPU_used":False,
                "python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"opencv":cv2.__version__,
                "original_results_unchanged":True,"cache_reused":True,"blas_threads":1}
            write_json(output/"runtime.json",runtime)
            write_report(output,metrics,summary,runtime)
            print(metrics.to_string(index=False));print(f"Saved to {output}")


if __name__=="__main__":
    main()
