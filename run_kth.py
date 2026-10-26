#!/usr/bin/env python3
"""KTH controlled-scene screening with unchanged 16-D frame differences/Gaussian sum."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import platform
from pathlib import Path
import re
import resource
import shlex
import shutil
import sys
import time
import warnings

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from download_kth import BASE, CLASSES, sha256
from run_experiment import gaussian_sum, setup_plot_font
from tune_gaussian import scale_fit_and_eval

SEED = 42
SUBJECTS = {"train": [11, 12, 13, 14, 15, 16, 17, 18],
            "validation": [19, 20, 21, 23, 24, 25, 1, 4],
            "test": [22, 2, 3, 5, 6, 7, 8, 9, 10]}
SIGMAS = [.05, .1, .2]
MODES = ["uniform_0_1", "uniform_p01_p99"]
CS = [.1, 1., 10.]
SOLVER = {"solver": "lbfgs", "max_iter": 5000, "tol": 1e-6, "random_state": SEED}
RULE = "validation Macro-F1 descending; accuracy descending; C ascending; center mode lexical; sigma ascending"
COLORS = ["#64748b", "#2878a3", "#dc7a36"]
ZH = {"walking": "走路", "jogging": "慢跑", "running": "跑步", "boxing": "拳击",
      "handwaving": "挥手", "handclapping": "拍手"}
METHODS = ["A", "G_default", "G_adjusted"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def manifest(root):
    """Use the original 1-based, inclusive annotated subsequences, not entire AVIs."""
    annotation = (root / "00sequences.txt").read_text(encoding="latin1")
    for split, prefix in [("train", "Training"), ("validation", "Validation"), ("test", "Test")]:
        match = re.search(rf"^{prefix}:\s*(.+)$", annotation, re.M)
        numbers = list(map(int, re.findall(r"\d+", match.group(1)))) if match else []
        if numbers != SUBJECTS[split]:
            raise RuntimeError(f"Official subject split changed: {prefix}: {numbers}")
    rows, officially_missing = [], []
    for line_number, line in enumerate(annotation.splitlines(), 1):
        if not line.startswith("person"):
            continue
        missing = re.fullmatch(r"(person(\d+)_(\w+)_d([1-4]))\s+\*missing\*\s*", line)
        if missing:
            stem, person, cls, scenario = missing.groups()
            officially_missing.append({"video_id": stem, "subject": int(person), "class": cls,
                                       "scenario": f"d{scenario}", "annotation_line": line_number,
                                       "exclude_reason": "officially_marked_missing_no_segment_intervals"})
            continue
        match = re.fullmatch(r"(person(\d+)_(\w+)_d([1-4]))\s+frames\s+(.+?)\s*", line)
        if not match:
            raise ValueError(f"Unparsed annotation on line {line_number}: {line}")
        stem, person, cls, scenario, ranges = match.groups()
        person = int(person)
        if cls not in CLASSES:
            raise ValueError(cls)
        split = next(k for k, people in SUBJECTS.items() if person in people)
        candidates = sorted((root / "videos").rglob(stem + "*.avi"))
        if len(candidates) > 1:
            raise RuntimeError(f"Ambiguous video identity: {stem}: {candidates}")
        for segment, span in enumerate(ranges.split(","), 1):
            start, end = map(int, span.strip().split("-"))
            if not 1 <= start <= end:
                raise ValueError(f"Invalid frame interval: {line}")
            rows.append({"sample_id": f"{stem}__segment{segment}", "video_id": stem,
                         "filename": candidates[0].name if candidates else stem + "_uncomp.avi",
                         "path": str(candidates[0].resolve()) if candidates else "",
                         "class": cls, "subject": person, "scenario": f"d{scenario}", "split": split,
                         "segment_number": segment, "start_frame_1based": start, "end_frame_1based": end,
                         "annotation_line": line_number})
    result = pd.DataFrame(rows)
    if result.empty or not result.sample_id.is_unique:
        raise RuntimeError("Empty annotation or duplicate sample IDs")
    result.attrs["officially_missing_videos"] = officially_missing
    return result


def sampling_plan(start_frame, end_frame, fps):
    """Same nominal center-one-second window as HMDB, within an annotated segment."""
    duration = (end_frame - start_frame + 1) / fps
    if duration < 1:
        raise ValueError("segment_shorter_than_1_second")
    start = (start_frame - 1) / fps + (duration - 1.) / 2.
    targets = start + np.linspace(0., 1., 17)
    indices = [min(end_frame - 1, max(start_frame - 1, round(float(t * fps)))) for t in targets]
    return targets, indices


def extract_video(rows):
    started = time.perf_counter()
    path = Path(rows[0]["path"]) if rows[0]["path"] else None
    if path is None or not path.is_file():
        return [{**r, "included": False, "exclude_reason": "file_missing"} for r in rows], {}
    video = {"video_id": rows[0]["video_id"], "path": str(path), "file_bytes": path.stat().st_size,
             "sha256": sha256(path)}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return [{**r, "included": False, "exclude_reason": "cannot_open"} for r in rows], video
    fps, n = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video.update({"fps": fps, "total_frames": n, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
    if not np.isfinite(fps) or fps <= 0 or n <= 0:
        cap.release()
        return [{**r, "included": False, "exclude_reason": "invalid_fps_or_frame_count"} for r in rows], video
    video["duration_seconds"] = n / fps
    records, plans = [], []
    for row in rows:
        record = {**row, "fps": fps, "total_frames": n, "video_sha256": video["sha256"],
                  "segment_duration_seconds": (row["end_frame_1based"] - row["start_frame_1based"] + 1) / fps,
                  "annotation_endpoint_beyond_video": bool(row["end_frame_1based"] > n)}
        try:
            targets, indices = sampling_plan(row["start_frame_1based"], row["end_frame_1based"], fps)
            if max(indices) >= n:
                raise ValueError("requested_sample_outside_actual_video")
            record.update({"target_times_seconds": targets.tolist(), "sampled_indices": indices,
                           "actual_sample_span_seconds": (indices[-1] - indices[0]) / fps,
                           "repeated_index_count": 17 - len(set(indices)),
                           "sampled_frame_bytes": 17 * 64 * 64 * 4, "raw_feature_bytes": 16 * 4})
            plans.append(record)
        except ValueError as error:
            record.update({"included": False, "exclude_reason": str(error)})
            records.append(record)
    wanted = {i for row in plans for i in row["sampled_indices"]}
    frames, timestamps = {}, {}
    if wanted:
        for index in range(max(wanted) + 1):
            if not cap.grab():
                break
            if index in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
                frames[index] = gray.astype(np.float32) / 255.
                timestamps[index] = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
    cap.release()
    for row in plans:
        missing = sorted(set(row["sampled_indices"]) - frames.keys())
        if missing:
            row.update({"included": False, "exclude_reason": f"frame_decode_failed:{missing}"})
        else:
            selected = np.stack([frames[i] for i in row["sampled_indices"]])
            feature = np.abs(np.diff(selected, axis=0)).mean(axis=(1, 2)).astype(np.float32)
            row.update({"feature": feature.tolist(), "included": True, "exclude_reason": "",
                        "decoder_timestamps_seconds": [timestamps[i] for i in row["sampled_indices"]]})
        records.append(row)
    video["preprocessing_seconds"] = time.perf_counter() - started
    video["sampled_unique_frames"] = len(frames)
    return records, video


def extraction_signature(frame):
    tree = ast.parse(Path(__file__).read_text())
    nodes = [ast.dump(n, include_attributes=False) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in {"sampling_plan", "extract_video"}]
    videos = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
               "sha256": sha256(p)} for p in sorted({Path(p) for p in frame.path if p})]
    return {"manifest_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
            "officially_missing_videos": frame.attrs.get("officially_missing_videos", []),
            "extractor_ast_sha256": hashlib.sha256("\n".join(nodes).encode()).hexdigest(),
            "videos": videos, "opencv": cv2.__version__, "numpy": np.__version__}


def load_or_extract(frame, out):
    signature = extraction_signature(frame)
    cache = out / "feature_cache.joblib"
    if cache.exists():
        payload = joblib.load(cache)
        if payload["signature"] != signature:
            raise RuntimeError("Feature-cache inputs/code changed. Use a NEW output directory; old results are preserved.")
        return payload, True
    started = time.perf_counter()
    all_records, videos = [], []
    for i, (_, group) in enumerate(frame.groupby("video_id", sort=True), 1):
        records, video = extract_video(group.to_dict("records"))
        all_records.extend(records)
        if video:
            videos.append(video)
        if i % 25 == 0:
            print(f"Extracted {i}/{frame.video_id.nunique()} source videos", flush=True)
    by_id = {r["sample_id"]: r for r in all_records}
    records = [by_id[k] for k in frame.sample_id]
    payload = {"signature": signature, "records": records, "videos": videos,
               "preprocessing_seconds": time.perf_counter() - started}
    joblib.dump(payload, cache, compress=3)
    return payload, False


def centers_from_train(mode, x):
    if mode == "uniform_0_1":
        return np.linspace(0., 1., 16)
    if mode == "uniform_p01_p99":
        return np.linspace(*np.quantile(x, [.01, .99]), 16)
    raise ValueError(mode)


def score(y, pred):
    cm = confusion_matrix(y, pred, labels=CLASSES)
    acc = float(accuracy_score(y, pred))
    assert np.isclose(acc, np.trace(cm) / cm.sum())
    return {"accuracy": acc, "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0))}


def fit_model(x, y, c):
    model = Pipeline([("standard_scaler", StandardScaler()),
                      ("logistic_regression", LogisticRegression(C=c, **SOLVER))])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    elapsed = time.perf_counter() - started
    scaler = model.named_steps["standard_scaler"]
    assert scaler.n_samples_seen_ == len(x)
    assert np.allclose(scaler.mean_, x.mean(axis=0)) and np.allclose(scaler.var_, x.var(axis=0))
    iterations = int(model.named_steps["logistic_regression"].n_iter_.max())
    converged = iterations < SOLVER["max_iter"] and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, {"fit_seconds": elapsed, "converged": converged, "iterations": iterations,
                   "warnings": [str(w.message) for w in caught]}


def select_parameters(raw, samples, out):
    train = samples.split.to_numpy() == "train"
    val = samples.split.to_numpy() == "validation"
    xt, xv, scaling = scale_fit_and_eval(raw[train], raw[val])
    labels = samples["class"].to_numpy()
    write_json(out / "training_validation_scaling.json", scaling)
    models, rows, predictions = {}, [], []
    configurations = [("A", "none", 0., c) for c in CS]
    configurations += [("G", mode, sigma, c) for mode in MODES for sigma in SIGMAS for c in CS]
    for family, mode, sigma, c in configurations:
        key = f"{family}__{mode}__sigma{sigma:g}__C{c:g}"
        centers = centers_from_train(mode, xt) if family == "G" else None
        started = time.perf_counter()
        ft, fv = (gaussian_sum(xt, sigma, centers), gaussian_sum(xv, sigma, centers)) if centers is not None else (xt, xv)
        transform_s = time.perf_counter() - started
        model, stats = fit_model(ft, labels[train], c)
        entry = {"config_id": key, "family": family, "center_mode": mode, "sigma": sigma, "C": c,
                 "centers": centers.tolist() if centers is not None else None,
                 "unique_centers": len(np.unique(centers)) if centers is not None else 0,
                 "feature_transform_seconds": transform_s, **stats}
        for split, mask, features in [("train", train, ft), ("validation", val, fv)]:
            pred = model.predict(features)
            entry.update({f"{split}_{k}": v for k, v in score(labels[mask], pred).items()})
            predictions.extend({"config_id": key, "sample_id": sid, "split": split,
                                "true_class": actual, "predicted_class": p}
                               for sid, actual, p in zip(samples.loc[mask, "sample_id"], labels[mask], pred))
        rows.append(entry)
        models[key] = model
        print(f"Validation {key}: acc={entry['validation_accuracy']:.4f}, F1={entry['validation_macro_f1']:.4f}", flush=True)
    write_json(out / "all_configuration_metrics.json", rows)
    table = pd.DataFrame(rows)
    table.to_csv(out / "all_configuration_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "all_configuration_predictions_train_validation.csv", index=False)
    if not table.converged.all():
        raise RuntimeError("At least one classifier did not converge; see configuration metrics. No test evaluation.")
    def best(options):
        return options.sort_values(["validation_macro_f1", "validation_accuracy", "C", "center_mode", "sigma"],
                                   ascending=[False, False, True, True, True], kind="stable").iloc[0].to_dict()
    chosen = {"A": best(table[table.family == "A"]),
              "G_default": best(table[(table.family == "G") & (table.center_mode == "uniform_0_1") & (table.sigma == .1)]),
              "G_adjusted": best(table[table.family == "G"])}
    frozen = {"selection_rule": RULE, "chosen": chosen, "test_used_for_selection": False,
              "fit_subjects": SUBJECTS["train"], "validation_subjects": SUBJECTS["validation"],
              "test_subjects": SUBJECTS["test"], "refit_train_plus_validation": False,
              "freeze_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    write_json(out / "parameters_frozen_before_test.json", frozen)
    return chosen, models, table


def independently_decode(row):
    """Separate sequential read path to verify streamed grab/retrieve extraction."""
    cap = cv2.VideoCapture(row.path)
    wanted = set(row.sampled_indices)
    frames = {}
    for index in range(max(wanted) + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Independent decoding failed: {row.sample_id}:{index}")
        if index in wanted:
            frames[index] = frame
    cap.release()
    selected = [frames[i] for i in row.sampled_indices]
    gray_u8 = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 64),
                                  interpolation=cv2.INTER_AREA) for f in selected])
    gray = gray_u8.astype(np.float32) / 255.
    feature = np.array([math.fsum(abs(float(b) - float(a)) for a, b in
                                  zip(gray[i].flat, gray[i + 1].flat)) / 4096 for i in range(16)])
    return selected, gray, feature


def audit_examples(samples, out):
    directory = out / "sampling_contact_sheets"
    directory.mkdir(exist_ok=True)
    audit = []
    for cls in CLASSES:
        candidates = samples[(samples["class"] == cls) & (samples.split == "train")]
        selected = candidates.sort_values(["filename", "segment_number"]).drop_duplicates("video_id").head(2)
        if len(selected) != 2:
            raise RuntimeError(f"Not enough training examples: {cls}")
        for example, (_, row) in enumerate(selected.iterrows(), 1):
            color, gray, independent = independently_decode(row)
            error = float(np.max(np.abs(independent - np.asarray(row.feature))))
            if error > 1e-7:
                raise RuntimeError(f"Independent feature mismatch: {row.sample_id}: {error}")
            fig, axes = plt.subplots(3, 6, figsize=(13, 6), dpi=160)
            for ax in axes.flat:
                ax.axis("off")
            for i, ax in enumerate(list(axes.flat)[:17]):
                ax.imshow(gray[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"#{i + 1}: frame {row.sampled_indices[i]}\nt={row.sampled_indices[i] / row.fps:.3f}s", fontsize=8)
            fig.suptitle(f"Actual 64x64 model input: {row.sample_id}", fontsize=10)
            fig.tight_layout()
            fig.savefig(directory / f"{cls}_{example}_gray.png", bbox_inches="tight")
            plt.close(fig)
            cv2.imwrite(str(directory / f"{cls}_{example}_example.png"), color[8])
            audit.append({"sample_id": row.sample_id, "class": cls, "split": "train", "filename": row.filename,
                          "sampled_indices": row.sampled_indices, "target_times_seconds": row.target_times_seconds,
                          "float64_manual_feature_max_error": error, "tolerance": 1e-7, "passed": True})
    write_json(out / "independent_video_audit.json", audit)
    return audit


def basic_checks(frame, samples, raw, videos):
    assert raw.shape == (len(samples), 16) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
    assert samples.sample_id.is_unique
    official = frame.set_index("sample_id")
    for _, row in samples.iterrows():
        reference = official.loc[row.sample_id]
        for column in ["class", "split", "subject", "video_id", "path", "start_frame_1based", "end_frame_1based"]:
            assert row[column] == reference[column], (row.sample_id, column)
        assert len(row.sampled_indices) == 17
        assert min(row.sampled_indices) >= row.start_frame_1based - 1
        assert max(row.sampled_indices) <= min(row.end_frame_1based - 1, row.total_frames - 1)
    overlaps = []
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a, b = samples[samples.split == left], samples[samples.split == right]
        for column in ["subject", "video_id", "path", "video_sha256"]:
            shared = sorted(set(a[column]) & set(b[column]))
            assert not shared, (left, right, column, shared)
            overlaps.append({"splits": [left, right], "identity": column, "overlap_count": 0})
    raw_hashes = samples[["split"]].copy()
    raw_hashes["feature_sha256"] = [hashlib.sha256(x.tobytes()).hexdigest() for x in raw]
    duplicates = raw_hashes[raw_hashes.feature_sha256.duplicated(keep=False)]
    return {"status": "PASSED", "samples": len(samples), "feature_shape": list(raw.shape),
            "finite_values": True, "alignment": True, "split_overlap_checks": overlaps,
            "all_zero_samples": int(np.all(raw == 0, axis=1).sum()),
            "duplicate_feature_rows_including_within_split": len(duplicates),
            "minimum_sampling_span_seconds": float(samples.actual_sample_span_seconds.min()),
            "maximum_sampling_span_seconds": float(samples.actual_sample_span_seconds.max()),
            "samples_with_repeated_frame_indices": int((samples.repeated_index_count > 0).sum()),
            "samples_with_annotation_end_beyond_video": int(samples.annotation_endpoint_beyond_video.sum()),
            "source_videos": len(videos), "frame_index_convention": "official 1-based inclusive -> decoder 0-based"}


def permutation_checks(model, x, centers, sigma, repetitions=100):
    reference = gaussian_sum(x, sigma, centers)
    probability = model.predict_proba(reference)
    prediction = model.predict(reference)
    rng = np.random.default_rng(SEED)
    feature_max = probability_max = 0.
    changed = 0
    for _ in range(repetitions):
        shuffled = np.take_along_axis(x, np.argsort(rng.random(x.shape), axis=1), axis=1)
        y = gaussian_sum(shuffled, sigma, centers)
        p = model.predict_proba(y)
        feature_max = max(feature_max, float(np.max(np.abs(reference - y))))
        probability_max = max(probability_max, float(np.max(np.abs(probability - p))))
        changed += int(np.count_nonzero(prediction != model.predict(y)))
        assert np.allclose(reference, y, atol=1e-10, rtol=1e-10)
        assert np.allclose(probability, p, atol=1e-10, rtol=1e-9)
    duplicated = np.full((1, 16), .25)
    expected = 16 * np.exp(-np.square(.25 - centers) / (2 * sigma * sigma))
    duplicate_error = float(np.max(np.abs(gaussian_sum(duplicated, sigma, centers)[0] - expected)))
    independent = np.array([[math.fsum(math.exp(-((float(v) - float(mu)) ** 2) / (2 * sigma ** 2))
                                     for v in row) for mu in centers] for row in x[:3]])
    independent_error = float(np.max(np.abs(reference[:3] - independent)))
    assert changed == 0 and duplicate_error <= 1e-10 and independent_error <= 1e-10
    return {"passed": True, "samples": len(x), "permutations_per_sample": repetitions,
            "feature_atol": 1e-10, "feature_rtol": 1e-10, "probability_atol": 1e-10,
            "probability_rtol": 1e-9, "max_feature_abs_difference": feature_max,
            "max_probability_abs_difference": probability_max, "prediction_changes": changed,
            "duplicate_input_sum_max_error": duplicate_error, "independent_math_max_error": independent_error}


def final_evaluation(raw, samples, selected, models, out):
    train = samples.split.to_numpy() == "train"
    scaled = np.empty_like(raw)
    scaled[train], _, _ = scale_fit_and_eval(raw[train], raw[train])
    scalings = {}
    for split in ["validation", "test"]:
        mask = samples.split.to_numpy() == split
        _, scaled[mask], scalings[split] = scale_fit_and_eval(raw[train], raw[mask])
    write_json(out / "final_scaling.json", scalings)
    labels = samples["class"].to_numpy()
    results, prediction_rows, audits, bundles = [], [], {}, {}
    arrays = {"raw": raw, "scaled": scaled, "labels": labels, "splits": samples.split.to_numpy(),
              "sample_ids": samples.sample_id.to_numpy()}
    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    for method in METHODS:
        config = selected[method]
        model = models[config["config_id"]]
        centers = centers_from_train(config["center_mode"], scaled[train]) if method != "A" else None
        started = time.perf_counter()
        feature = gaussian_sum(scaled, config["sigma"], centers) if centers is not None else scaled
        transform_seconds = time.perf_counter() - started
        assert feature.shape == raw.shape and np.isfinite(feature).all()
        bundle = {"method": method, "classifier": model, "centers": centers, "sigma": config["sigma"],
                  "regularization_C": config["C"], "input_divisor": scalings["test"]["divisor"],
                  "class_order": CLASSES, "train_sample_ids": samples.loc[train, "sample_id"].tolist(),
                  "configuration": config, "permutation_invariant": method != "A"}
        joblib.dump(bundle, model_dir / f"{method}.joblib", compress=3)
        bundles[method] = bundle
        arrays[method] = feature
        result = {"method": method, "output_dimension": 16, "train_samples": int(train.sum()),
                  "validation_samples": int((samples.split == "validation").sum()),
                  "test_samples": int((samples.split == "test").sum()),
                  "center_mode": config["center_mode"], "sigma": config["sigma"], "C": config["C"],
                  "centers": centers.tolist() if centers is not None else None,
                  "feature_transform_seconds_all_samples": transform_seconds,
                  "classifier_training_seconds": config["fit_seconds"], "converged": config["converged"]}
        for split in ["train", "validation", "test"]:
            mask = samples.split.to_numpy() == split
            started = time.perf_counter()
            probabilities = model.predict_proba(feature[mask])
            pred = model.classes_[probabilities.argmax(axis=1)]
            result[f"{split}_prediction_seconds"] = time.perf_counter() - started
            result.update({f"{split}_{k}": v for k, v in score(labels[mask], pred).items()})
            result[f"{split}_predicted_class_counts"] = {c: int(np.count_nonzero(pred == c)) for c in CLASSES}
            for index, actual, p, probabilities_row in zip(np.flatnonzero(mask), labels[mask], pred, probabilities):
                prediction_rows.append({"method": method, "sample_id": samples.iloc[index].sample_id,
                                        "filename": samples.iloc[index].filename, "split": split,
                                        "subject": int(samples.iloc[index].subject), "true_class": actual,
                                        "predicted_class": p, **{f"prob_{c}": float(v) for c, v in zip(model.classes_, probabilities_row)}})
            if split == "test":
                cm = confusion_matrix(labels[mask], pred, labels=CLASSES)
                pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{method}.csv")
        if centers is not None:
            audits[method] = permutation_checks(model, scaled, centers, config["sigma"])
            result["order_independence"] = "PASSED"
            pd.DataFrame({"kernel": np.arange(1, 17), "center": centers,
                          "training_mean": feature[train].mean(axis=0), "training_std": feature[train].std(axis=0),
                          "training_min": feature[train].min(axis=0), "training_max": feature[train].max(axis=0)}).to_csv(
                              out / f"gaussian_output_statistics_{method}.csv", index=False)
        else:
            result["order_independence"] = "not applicable"
        results.append(result)
    write_json(out / "order_independence_checks.json", audits)
    write_json(out / "complete_metrics.json", results)
    pd.DataFrame(results).to_csv(out / "results_table.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "sample_predictions.csv", index=False)
    np.savez_compressed(out / "features_arrays.npz", **arrays)
    return pd.DataFrame(results), bundles, scaled


def draw_cm(ax, cm, title, chinese):
    labels = [ZH[c] if chinese else c for c in CLASSES]
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels, title=title,
           xlabel="预测类别" if chinese else "Predicted class", ylabel="真实类别" if chinese else "True class")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)


def render(out, samples, raw, scaled, results, selected, bundles):
    chinese = setup_plot_font()
    train = samples.split.to_numpy() == "train"
    labels = samples["class"].to_numpy()
    q = [.01, .05, .25, .5, .75, .95, .99]
    statistics = []
    for kind, x in [("raw", raw), ("scaled", scaled)]:
        for cls in ["ALL", *CLASSES]:
            values = x[train & ((labels == cls) if cls != "ALL" else True)].ravel()
            statistics.append({"representation": kind, "class": cls, "min": float(values.min()), "max": float(values.max()),
                               "mean": float(values.mean()), "std": float(values.std()),
                               **{f"p{100 * p:g}": float(np.quantile(values, p)) for p in q}})
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        bins = np.linspace(0, float(x[train].max()) or 1., 51)
        for cls in CLASSES:
            ax.hist(x[train & (labels == cls)].ravel(), bins=bins, density=True, histtype="step", linewidth=1.6,
                    label=ZH[cls] if chinese else cls)
        ax.set(xlabel=("原始画面变化量" if kind == "raw" else "缩放后画面变化量") if chinese else f"{kind} frame-change magnitude",
               ylabel="概率密度" if chinese else "Probability density",
               title="仅训练集输入分布（各类统一分箱）" if chinese else "Training inputs only (shared bins)")
        ax.legend(ncol=3)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f"input_distribution_{kind}.png", bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(statistics).to_csv(out / "training_input_statistics.csv", index=False)
    candidates = pd.read_csv(out / "all_configuration_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, sharey=True)
    for ax, mode in zip(axes, MODES):
        for c in CS:
            part = candidates[(candidates.family == "G") & (candidates.center_mode == mode) & (candidates.C == c)].sort_values("sigma")
            ax.plot(part.sigma, part.validation_macro_f1, "o-", label=f"C={c:g}")
        ax.set(xlabel="统一宽度 sigma" if chinese else "Shared width sigma", title=mode, xticks=SIGMAS)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("验证 Macro-F1" if chinese else "Validation Macro-F1")
    fig.suptitle("参数比较：仅使用官方验证集" if chinese else "Parameter comparison: official validation only")
    fig.tight_layout()
    fig.savefig(out / "parameter_comparison.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=200)
    for method, color in [("G_default", COLORS[1]), ("G_adjusted", COLORS[2])]:
        part = pd.read_csv(out / f"gaussian_output_statistics_{method}.csv")
        axes[0].plot(part.kernel, part.training_mean, "o-", color=color, label=method)
        axes[1].plot(part.kernel, part.training_std, "o-", color=color, label=method)
    for ax in axes:
        ax.set_xticks(range(1, 17))
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set(ylabel="训练均值" if chinese else "Training mean", title="Gaussian核输出变化" if chinese else "Gaussian outputs across training samples")
    axes[1].set(xlabel="核编号 j" if chinese else "Kernel j", ylabel="训练标准差" if chinese else "Training standard deviation")
    fig.tight_layout()
    fig.savefig(out / "gaussian_output_statistics.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    for cls in CLASSES:
        index = samples[(samples["class"] == cls) & (samples.split == "train")].sort_values("sample_id").index[0]
        ax.plot(range(1, 17), scaled[index], marker="o", ms=3, label=f"{cls}: {samples.iloc[index].sample_id}")
    ax.set(xticks=range(1, 17), xlabel="输入位置" if chinese else "Input position",
           ylabel="缩放后的帧差" if chinese else "Scaled frame difference",
           title="每类一个固定训练样本" if chinese else "One fixed training sample per class")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "input_examples.png", bbox_inches="tight")
    plt.close(fig)
    for method in METHODS:
        cm = pd.read_csv(out / f"confusion_{method}.csv", index_col=0).to_numpy()
        fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
        draw_cm(ax, cm, f"KTH {method} · " + ("测试集" if chinese else "test"), chinese)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{method}.png", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 3), dpi=220)
    ax.axis("off")
    headers = ["方案", "维度", "训练准确率", "验证准确率", "验证F1", "测试准确率", "测试F1", "参数", "顺序无关"] if chinese else [
        "Method", "Dim", "Train acc", "Val acc", "Val F1", "Test acc", "Test F1", "Parameters", "Invariant"]
    cells = []
    for _, r in results.iterrows():
        mode = "[0,1]" if r.center_mode == "uniform_0_1" else "train P1-P99"
        parameter = f"C={r.C:g}" if r.method == "A" else f"{mode}, sigma={r.sigma:g}, C={r.C:g}"
        cells.append([r.method, 16, f"{r.train_accuracy:.2%}", f"{r.validation_accuracy:.2%}", f"{r.validation_macro_f1:.4f}",
                      f"{r.test_accuracy:.2%}", f"{r.test_macro_f1:.4f}", parameter, "N/A" if r.method == "A" else "PASS"])
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[.11, .04, .10, .10, .08, .10, .08, .23, .07])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.3)
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor("#e8eef4" if i == 0 else ("#f7f9fc" if i % 2 else "white"))
    ax.set_title("KTH：官方跨参与者训练／验证／测试；仅训练8人拟合" if chinese else "KTH: official subject-disjoint train/validation/test; fitted on 8 training subjects", pad=12)
    fig.tight_layout()
    fig.savefig(out / "results_table.png", bbox_inches="tight")
    plt.close(fig)

    example = samples[samples.split == "train"].sort_values("sample_id").iloc[0]
    idx = samples.index[samples.sample_id == example.sample_id][0]
    color, _, _ = independently_decode(example)
    bundle = bundles["G_adjusted"]
    y = gaussian_sum(scaled[idx:idx + 1], bundle["sigma"], bundle["centers"])[0]
    rng = np.random.default_rng(SEED)
    shuffled = scaled[idx][rng.permutation(16)]
    yp = gaussian_sum(shuffled[None], bundle["sigma"], bundle["centers"])[0]
    diff = float(np.max(np.abs(y - yp)))
    same = bundle["classifier"].predict(y[None])[0] == bundle["classifier"].predict(yp[None])[0]

    def panel_input(fig, spec):
        axes = [fig.add_subplot(s) for s in spec.subgridspec(1, 3, width_ratios=[1.1, 1, 1])]
        axes[0].imshow(cv2.cvtColor(color[8], cv2.COLOR_BGR2RGB))
        axes[0].axis("off")
        axes[0].set_title("真实训练帧" if chinese else "Real training frame", fontsize=10)
        axes[1].plot(range(1, 17), scaled[idx], "o-", ms=3, color=COLORS[0])
        axes[1].set(xlabel="输入位置 i" if chinese else "Input position i", ylabel="缩放帧差" if chinese else "Scaled difference")
        axes[1].set_title("16维输入" if chinese else "16-D input", fontsize=10)
        axes[2].plot(range(1, 17), y, "o-", ms=3, color=COLORS[2])
        axes[2].set(xlabel="核编号 j" if chinese else "Kernel j", ylabel="响应总和" if chinese else "Summed response")
        axes[2].set_title("16维Gaussian输出" if chinese else "16-D Gaussian output", fontsize=10)
        axes[0].text(0, -.18, example.sample_id, fontsize=7, transform=axes[0].transAxes)

    def panel_order(ax):
        ax.plot(range(1, 17), y, "o-", label="原排列" if chinese else "Original", color=COLORS[1])
        ax.plot(range(1, 17), yp, "x--", label="随机打乱" if chinese else "Permuted", color=COLORS[2])
        ax.set(xlabel="核编号 j" if chinese else "Kernel j", ylabel="响应总和" if chinese else "Summed response",
               title=("② 顺序无关检查" if chinese else "2. Permutation invariance") + f"\nmax |difference|={diff:.2e}; prediction {'unchanged' if same else 'CHANGED'}")
        ax.legend()
        ax.grid(alpha=.2)

    def panel_performance(ax):
        positions = np.arange(3)
        for offset, metric, color, label in [(-.18, "validation_accuracy", COLORS[1], "验证" if chinese else "Validation"),
                                              (.18, "test_accuracy", COLORS[2], "测试" if chinese else "Test")]:
            bars = ax.bar(positions + offset, results[metric] * 100, .34, label=label, color=color)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set(xticks=positions, xticklabels=METHODS, ylim=(0, 100), ylabel="准确率 (%)" if chinese else "Accuracy (%)",
               title="③ 性能对比（单次官方验证划分，无折间误差线）" if chinese else "3. Performance (single official validation split)")
        ax.legend()
        ax.grid(axis="y", alpha=.2)

    cm = pd.read_csv(out / "confusion_G_adjusted.csv", index_col=0).to_numpy()
    fig = plt.figure(figsize=(18, 11), dpi=220, layout="constrained")
    grid = fig.add_gridspec(2, 2)
    panel_input(fig, grid[0, 0])
    panel_order(fig.add_subplot(grid[0, 1]))
    panel_performance(fig.add_subplot(grid[1, 0]))
    draw_cm(fig.add_subplot(grid[1, 1]), cm, "④ 验证选定G的测试混淆矩阵" if chinese else "4. Validation-selected G: test confusion", chinese)
    fig.suptitle("KTH受控场景：原输入与Gaussian求和的软件验证" if chinese else "KTH controlled-scene software screening", fontsize=17)
    fig.savefig(out / "four_panel_summary.png", bbox_inches="tight")
    plt.close(fig)
    for name, callback in [("panel_order_independence", panel_order), ("panel_performance", panel_performance)]:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=220)
        callback(ax)
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), bbox_inches="tight")
        plt.close(fig)
    fig = plt.figure(figsize=(12, 4), dpi=220, layout="constrained")
    panel_input(fig, fig.add_gridspec(1, 1)[0])
    fig.savefig(out / "panel_input_output.png", bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out / "confusion_G_adjusted.png", out / "panel_confusion.png")


def report(out, results, samples, runtime):
    a, default, g = [results[results.method == name].iloc[0] for name in METHODS]
    counts = samples.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(CLASSES)
    counts.to_csv(out / "included_counts_by_class.csv")
    cm = pd.read_csv(out / "confusion_G_adjusted.csv", index_col=0).to_numpy()
    confusions = sorted([(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(6) for j in range(6) if i != j], reverse=True)[:5]
    per_subject = []
    predictions = pd.read_csv(out / "sample_predictions.csv")
    for (method, split, person), group in predictions.groupby(["method", "split", "subject"]):
        per_subject.append({"method": method, "split": split, "subject": int(person), "samples": len(group),
                            **score(group.true_class, group.predicted_class)})
    pd.DataFrame(per_subject).to_csv(out / "per_subject_metrics.csv", index=False)
    per_class = []
    for method in METHODS:
        mat = pd.read_csv(out / f"confusion_{method}.csv", index_col=0).to_numpy()
        for i, cls in enumerate(CLASSES):
            per_class.append({"method": method, "class": cls, "test_samples": int(mat[i].sum()),
                              "correct": int(mat[i, i]), "recall": float(mat[i, i] / mat[i].sum()),
                              "predicted_count": int(mat[:, i].sum())})
    pd.DataFrame(per_class).to_csv(out / "per_class_metrics.csv", index=False)
    summary = (f"每个动作片段取中间1秒、17帧，得到16个全图平均绝对帧差。每核对全部16个数求Gaussian响应之和，共16维，排列检查通过。"
               f"官方参与者划分、仅训练8人拟合；A/G默认/G调整测试准确率为{a.test_accuracy:.1%}/{default.test_accuracy:.1%}/{g.test_accuracy:.1%}，"
               f"调整G验证准确率{g.validation_accuracy:.1%}，测试F1={g.test_macro_f1:.3f}。"
               "本轮是单次跨参与者软件筛选，不等于硬件验证；是否继续应结合各类错误、参与者差异及编码收益判断。")
    (out / "简短中文说明.md").write_text(summary + "\n", encoding="utf-8")
    lines = ["# KTH 六类受控场景软件预实验", "", summary, "", "## 数据与模型", "",
             f"- 数据来源：{BASE}；标注和原始参与者划分：{BASE}00sequences.txt。",
             "- 类别：walking（走路）、jogging（慢跑）、running（跑步）、boxing（拳击）、handwaving（挥手）、handclapping（拍手）。不是HMDB51的原六类。",
             f"- 实际纳入训练/验证/测试：{int((samples.split == 'train').sum())}/{int((samples.split == 'validation').sum())}/{int((samples.split == 'test').sum())}个官方标注动作子片段。一个原视频可有多个片段，不能将片段数当作视频文件数。",
             f"- 参与者：{SUBJECTS}。同人及同视频所有子片段均在同一集合；不随机混分片段。",
             "- 官方帧边界解释为1起始、闭区间；转为0起始解码索引。每个标注片段中间取1秒，依实际FPS采17帧；逐帧灰度→64×64→float32/255→相邻绝对差的全像素均值。帧网格量化和极短边界截断会使实际首末帧跨度略偏离1秒，记录在清单中。",
             "- 不裁人体、不增加光流/CNN/骨架，不改变HMDB输入定义；仅新增KTH标注片段适配。",
             "- 输入统一除以训练8人的全局最大值；验证/测试越界值截断。零最大值用1作为除数。StandardScaler仅在训练8人的相应特征上拟合。",
             "- R[n,i,j]=exp(-(X[n,i]-mu[j])²/(2*sigma²)); Y[n,j]=sum_i R[n,i,j]。峰值1，16个中心，输出16维；重复值按次数累加，不去重。",
             "- A保留输入位置；G只保留Gaussian求和。这一差异同时包含Gaussian映射和汇总，不能全归因于非线性。", "", "## 参数选择与结果", "",
             "- 采用官方单次8人验证集，而非把同人片段随机做3折。没有折间均值/标准差，不绘制虚构误差线；逐参与者成绩另存。",
             f"- 预先固定候选：中心{MODES}；sigma={SIGMAS}；各方案分类器C={CS}。G默认固定均匀[0,1]和sigma=0.1，但同样选分类器C。",
             f"- 选择规则：{RULE}。中心P1/P99仅由训练8人的缩放输入计算；重合中心保留，不改维数。",
             "- 先保存parameters_frozen_before_test.json，再评价测试。直接复用训练8人拟合且用于验证的获选模型，不额外合并验证集重训。",
             f"- 分类器统一StandardScaler + multinomial LogisticRegression（6类时lbfgs默认多项逻辑回归），{SOLVER}，L2正则化；所有候选收敛才继续。",
             "", "|方案|训练Acc|验证Acc|验证Macro-F1|测试Acc|测试Macro-F1|参数|", "|---|---:|---:|---:|---:|---:|---|"]
    for _, r in results.iterrows():
        lines.append(f"|{r.method}|{r.train_accuracy:.2%}|{r.validation_accuracy:.2%}|{r.validation_macro_f1:.4f}|{r.test_accuracy:.2%}|{r.test_macro_f1:.4f}|{r.center_mode}, sigma={r.sigma:g}, C={r.C:g}|")
    lines += ["", f"- G调整−A：测试准确率{100 * (g.test_accuracy - a.test_accuracy):+.2f}个百分点，Macro-F1 {g.test_macro_f1 - a.test_macro_f1:+.4f}。",
              f"- G调整−G默认：测试准确率{100 * (g.test_accuracy - default.test_accuracy):+.2f}个百分点，Macro-F1 {g.test_macro_f1 - default.test_macro_f1:+.4f}。",
              "- 测试成绩不用于重新选配置或挑选展示的G；四联图混淆矩阵固定展示按验证Macro-F1选定的G调整。",
              "- G调整最常见定向错分：" + "；".join(f"{source}→{target}: {n}" for n, source, target in confusions) + "。",
              "- 不将HMDB与KTH的准确率差直接视为控制背景的因果效应：类别、参与者、数据来源及标注片段协议也不同。", "", "## 检查与可复查材料", "",
              "- 实现检查：implementation_checks.json；12个训练片段独立逐帧解码与float64手工帧差复核：independent_video_audit.json。",
              "- 每个G对全部纳入样本各随机排列100次；特征atol/rtol=1e-10/1e-10、概率1e-10/1e-9；最大误差、预测变化数和重复值累加检查见order_independence_checks.json。",
              "- 清单、原始/缩放/高斯特征、模型、完整候选验证表现、预测概率、混淆矩阵、参与者成绩均保存在本目录。原HMDB目录未修改。", "", "## 图示说明", "",
              "1. 输入输出图：固定训练片段的真实帧及16维输入/输出，仅为示例，不能代表全类动作质量。",
              "2. 顺序无关图：原排列和随机打乱的Gaussian输出应重合；完整样本检查见JSON。",
              "3. 性能图：官方单次验证和测试准确率；没有跨折或重复实验误差线。",
              "4. 混淆矩阵：纵轴真实、横轴预测；格内为测试片段计数。", "", "## 实际成本与下一步", "",
              f"- 逐视频采样预处理：{runtime['preprocessing_seconds_historical']:.2f}秒；本次是否复用特征：{runtime['feature_cache_reused']}。",
              f"- 本次实验墙钟：{runtime['experiment_wall_seconds']:.2f}秒（不含独立下载进程）；CPU运行，未使用GPU；实验进程峰值RSS：{runtime['peak_process_rss_bytes'] / 1024**2:.1f} MiB。",
              "- 原视频大小、去重后视频时长/帧数、采样帧概念存储量和最终特征大小分开记录在data_and_cost_stats.json；采样帧总大小不是峰值内存。",
              "- 下载/校验/解压实测记录见download_extract_timing.json；现有数据复用时下载耗时为未测量，不填0冒充新测量。",
              "- 先看训练表现、参与者差异及动作混淆，再决定是否单独比较更有动作含义的输入。若进一步改输入或调参，需要新的预注册验证协议；本次测试结果已查看，不能继续把它当未见独立最终验证。",
              "- 当前软件结果不等于器件实测，也不证明所有视觉任务可行或不可行。若保留候选，具体中心/宽度/C及训练尺度见models/G_adjusted.joblib与parameters_frozen_before_test.json。"]
    (out / "实验说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/kth"))
    parser.add_argument("--output", type=Path, default=Path("outputs/kth_six_subject_split_v1"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        payload = joblib.load(out / "feature_cache.joblib")
        samples = pd.DataFrame(payload["records"])
        samples = samples[samples.included].reset_index(drop=True)
        arrays = np.load(out / "features_arrays.npz")
        results = pd.DataFrame(json.loads((out / "complete_metrics.json").read_text()))
        selected = json.loads((out / "parameters_frozen_before_test.json").read_text())["chosen"]
        bundles = {m: joblib.load(out / "models" / (m + ".joblib")) for m in METHODS}
        render(out, samples, arrays["raw"], arrays["scaled"], results, selected, bundles)
        report(out, results, samples, json.loads((out / "runtime.json").read_text()))
        print(f"Rendered existing results: {out.resolve()}")
        return
    started = time.perf_counter()
    frame = manifest(args.data_root)
    signature = {"source_sha256": sha256(__file__), "gaussian_source_sha256": sha256(Path(__file__).with_name("run_experiment.py")),
                 "scaling_source_sha256": sha256(Path(__file__).with_name("tune_gaussian.py")),
                 "data": extraction_signature(frame), "subject_splits": SUBJECTS, "sigmas": SIGMAS,
                 "center_modes": MODES, "regularization": CS, "solver": SOLVER, "selection_rule": RULE,
                 "sklearn": sklearn.__version__, "numpy": np.__version__}
    signature_path = out / "run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise RuntimeError("Experiment conditions changed. Choose a NEW --output; old results will not be overwritten.")
    if (out / "COMPLETE.json").exists():
        print("Identical completed experiment reused; no preprocessing or training repeated.")
        print(pd.read_csv(out / "results_table.csv")[["method", "train_accuracy", "validation_accuracy", "test_accuracy", "test_macro_f1"]].to_string(index=False))
        return
    write_json(signature_path, signature)
    frame.to_csv(out / "official_manifest_all.csv", index=False)
    pd.DataFrame(frame.attrs["officially_missing_videos"]).to_csv(out / "official_missing_videos.csv", index=False)
    payload, reused = load_or_extract(frame, out)
    records = pd.DataFrame(payload["records"])
    samples = records[records.included].reset_index(drop=True)
    if set(samples.split) != set(SUBJECTS) or any(set(samples.loc[samples.split == s, "class"]) != set(CLASSES) for s in SUBJECTS):
        raise RuntimeError("A split/class is missing; inspect available KTH files and exclusions before training.")
    raw = np.asarray(samples.feature.tolist(), dtype=np.float64)
    records.to_csv(out / "sample_manifest_and_features.csv", index=False)
    samples.drop(columns="feature").to_csv(out / "sample_manifest.csv", index=False)
    exclusions = records[~records.included].drop(columns="feature", errors="ignore")
    exclusions.to_csv(out / "exclusions.csv", index=False)
    exclusions.groupby(["class", "split", "exclude_reason"]).size().rename("count").reset_index().to_csv(out / "excluded_counts.csv", index=False)
    pd.DataFrame(payload["videos"]).to_csv(out / "video_manifest.csv", index=False)
    checks = basic_checks(frame, samples, raw, payload["videos"])
    checks["independent_video_examples"] = audit_examples(samples, out)
    write_json(out / "implementation_checks.json", checks)
    print(f"Checks passed: {len(samples)} included subsequences; counts={dict(Counter(samples.split))}", flush=True)
    with threadpool_limits(limits=1):
        selected, models, table = select_parameters(raw, samples, out)
        results, bundles, scaled = final_evaluation(raw, samples, selected, models, out)
    render(out, samples, raw, scaled, results, selected, bundles)
    used_ids = set(samples.video_id)
    videos = [v for v in payload["videos"] if v["video_id"] in used_ids]
    cost = {"candidate_subsequences": len(records), "included_subsequences": len(samples),
            "excluded_subsequences": len(exclusions), "included_unique_video_files": len(videos),
            "original_video_bytes_unique": sum(v["file_bytes"] for v in videos),
            "original_video_duration_seconds_unique": sum(v["duration_seconds"] for v in videos),
            "original_video_frames_unique": sum(v["total_frames"] for v in videos),
            "included_annotated_segment_duration_seconds": float(samples.segment_duration_seconds.sum()),
            "sampled_float32_frames_bytes_if_all_stored": int(samples.sampled_frame_bytes.sum()),
            "sampled_frames_materialized_all_at_once": False,
            "one_sample_float32_frames_bytes": 17 * 64 * 64 * 4,
            "raw_float32_feature_bytes": len(raw) * 16 * 4, "saved_float64_raw_array_bytes": raw.nbytes,
            "raw_scaled_two_gaussian_arrays_float64_bytes": raw.nbytes * 4,
            "preprocessing_seconds_historical": payload["preprocessing_seconds"],
            "preprocessing_seconds_this_run": 0. if reused else payload["preprocessing_seconds"],
            "feature_cache_reused": reused,
            "mean_preprocessing_seconds_per_source_video": payload["preprocessing_seconds"] / len(payload["videos"]),
            "mean_preprocessing_seconds_per_candidate_subsequence": payload["preprocessing_seconds"] / len(records)}
    write_json(out / "data_and_cost_stats.json", cost)
    timing = args.data_root / "download_extract_timing.json"
    if timing.exists():
        shutil.copy2(timing, out / timing.name)
    else:
        write_json(out / timing.name, {"download_seconds": None, "extraction_seconds": None,
                                       "reason": "data already existed; acquisition not measured"})
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    cpu_match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
               "sklearn": sklearn.__version__, "cpu": platform.processor() or platform.machine(),
               "cpu_model": cpu_match.group(1) if cpu_match else None,
               "logical_cpu_count": psutil.cpu_count(), "memory_total_bytes": psutil.virtual_memory().total,
               "gpu_used": False, "BLAS_threads": 1, "feature_cache_reused": reused,
               "preprocessing_seconds_historical": payload["preprocessing_seconds"],
               "candidate_classifier_fits": len(table), "final_additional_fits": 0,
               "classifier_fit_seconds_total": float(table.fit_seconds.sum()),
               "experiment_wall_seconds": time.perf_counter() - started,
               "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    write_json(out / "runtime.json", runtime)
    report(out, results, samples, runtime)
    command = "cd " + shlex.quote(str(Path.cwd())) + "\n" + shlex.join([sys.executable, *sys.argv]) + "\n"
    (out / "run_command.txt").write_text(command, encoding="utf-8")
    snapshot = out / "source_code"
    snapshot.mkdir(exist_ok=True)
    for name in ["run_kth.py", "download_kth.py", "verify_kth.py", "test_kth.py", "run_experiment.py", "tune_gaussian.py", "requirements.txt"]:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    write_json(out / "COMPLETE.json", {"status": "COMPLETE", "local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                        "signature_sha256": sha256(signature_path)})
    print(results[["method", "train_accuracy", "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1"]].to_string(index=False))
    print(f"Saved: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
