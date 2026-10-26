#!/usr/bin/env python3
"""Expanded KTH Gaussian screening, reusing audited raw features and matching fits."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from download_kth import sha256
from run_experiment import gaussian_sum, setup_plot_font
from run_kth import (CLASSES, SUBJECTS, SOLVER, basic_checks, draw_cm,
                     extraction_signature, fit_model, manifest, permutation_checks, score)
from tune_gaussian import scale_fit_and_eval

SEED = 42
CENTER_MODES = ["uniform_0_1", "uniform_p01_p99", "uniform_p05_p95", "uniform_0_p95",
                "quantile_p01_p99", "quantile_p05_p95", "power2_0_1", "power3_0_1"]
SIGMAS = [.005, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5, .75, 1., 2.]
CS = [.001, .01, .1, 1., 10., 100., 1000.]
RULE = "converged only; validation Macro-F1 descending; validation accuracy descending; C ascending; center mode lexical; sigma ascending"
METHODS = ["A", "G_default", "G_adjusted"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def centers_from_training(mode, train):
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 16 or not np.isfinite(train).all():
        raise ValueError("Expected finite N x 16 training features")
    t = np.linspace(0., 1., 16)
    if mode == "uniform_0_1":
        result = t
    elif mode in {"uniform_p01_p99", "uniform_p05_p95"}:
        q = [.01, .99] if mode == "uniform_p01_p99" else [.05, .95]
        result = np.linspace(*np.quantile(train, q), 16)
    elif mode == "uniform_0_p95":
        result = np.linspace(0., np.quantile(train, .95), 16)
    elif mode in {"quantile_p01_p99", "quantile_p05_p95"}:
        lo, hi = (.01, .99) if mode == "quantile_p01_p99" else (.05, .95)
        result = np.quantile(train, np.linspace(lo, hi, 16))
    elif mode in {"power2_0_1", "power3_0_1"}:
        result = t ** (2 if mode == "power2_0_1" else 3)
    else:
        raise ValueError(mode)
    if result.shape != (16,) or not np.isfinite(result).all() or np.any(np.diff(result) < 0):
        raise AssertionError("Invalid centers")
    if result.min() < 0 or result.max() > 1:
        raise AssertionError("Centers outside normalized input range")
    # Retain coincident centers and all 16 kernels; no jitter, sorting of inputs or deduplication.
    return result


def config_id(family, mode, sigma, c):
    return f"{family}__{mode}__sigma{sigma:g}__C{c:g}"


def rank(options):
    eligible = options[options.converged].copy()
    if eligible.empty:
        raise RuntimeError("No converged configuration in selection pool")
    return eligible.sort_values(["validation_macro_f1", "validation_accuracy", "C", "center_mode", "sigma"],
                                ascending=[False, False, True, True, True], kind="stable")


def audit_source(project, source, data_root, out):
    started = time.perf_counter()
    signature = json.loads((source / "run_signature.json").read_text())
    if signature["source_sha256"] != sha256(project / "run_kth.py"):
        raise RuntimeError("Original KTH implementation changed; cannot authenticate prior fits")
    for key, name in [("gaussian_source_sha256", "run_experiment.py"), ("scaling_source_sha256", "tune_gaussian.py")]:
        if signature[key] != sha256(project / name):
            raise RuntimeError(f"Original preprocessing changed: {name}")
    if signature["solver"] != SOLVER or signature["sklearn"] != sklearn.__version__ or signature["numpy"] != np.__version__:
        raise RuntimeError("Previous classifier/environment no longer matches")
    official = manifest(data_root)
    if signature["data"] != extraction_signature(official):
        raise RuntimeError("KTH annotations, videos, extraction code or environment changed")
    payload = joblib.load(source / "feature_cache.joblib")
    if payload["signature"] != signature["data"]:
        raise RuntimeError("Feature cache identity mismatch")
    records = pd.DataFrame(payload["records"])
    samples = records[records.included].reset_index(drop=True)
    raw = np.asarray(samples.feature.tolist(), dtype=np.float64)
    saved = np.load(source / "features_arrays.npz", allow_pickle=True)  # our trusted local artifact
    np.testing.assert_array_equal(raw, saved["raw"])
    np.testing.assert_array_equal(samples.sample_id.to_numpy(), saved["sample_ids"])
    np.testing.assert_array_equal(samples["class"].to_numpy(), saved["labels"])
    np.testing.assert_array_equal(samples.split.to_numpy(), saved["splits"])
    checks = basic_checks(official, samples, raw, payload["videos"])
    prior_checks = json.loads((source / "implementation_checks.json").read_text())
    if prior_checks["status"] != "PASSED" or not all(r["passed"] for r in prior_checks["independent_video_examples"]):
        raise RuntimeError("Original real-video independent checks are not passed")
    checks.update({"raw_features_reused": True, "videos_redecoded_this_run": 0,
                   "video_content_SHA256_rechecked": True, "source": str(source.resolve()),
                   "previous_independent_video_audit_reused": True,
                   "audit_seconds": time.perf_counter() - started})
    write_json(out / "implementation_checks.json", checks)
    samples.drop(columns="feature").to_csv(out / "sample_manifest.csv", index=False)
    for name in ["exclusions.csv", "excluded_counts.csv", "official_manifest_all.csv", "official_missing_videos.csv",
                 "included_counts_by_class.csv", "video_manifest.csv", "independent_video_audit.json", "decoder_warning_audit.json"]:
        if (source / name).exists():
            shutil.copy2(source / name, out / name)
    return samples, raw, payload


def authenticate_old_metrics(source, samples):
    rows = json.loads((source / "all_configuration_metrics.json").read_text())
    predictions = pd.read_csv(source / "all_configuration_predictions_train_validation.csv")
    old = {}
    for row in rows:
        for split in ["train", "validation"]:
            expected = samples[samples.split == split]
            part = predictions[(predictions.config_id == row["config_id"]) & (predictions.split == split)]
            if part.sample_id.tolist() != expected.sample_id.tolist() or part.true_class.tolist() != expected["class"].tolist():
                raise AssertionError("Old fit prediction/sample alignment mismatch")
            for metric, value in score(part.true_class, part.predicted_class).items():
                if not np.isclose(value, row[f"{split}_{metric}"], atol=1e-14):
                    raise AssertionError("Old validation/training metric mismatch")
        old[row["config_id"]] = row
    if len(old) != 21 or not all(row["converged"] for row in old.values()):
        raise RuntimeError("The 21 previous converged configurations are incomplete")
    return old, predictions


def encoded_predictions(predictions):
    mapping = {c: i for i, c in enumerate(CLASSES)}
    return np.asarray([mapping[c] for c in predictions], dtype=np.int8)


def search(samples, raw, source, out):
    train = samples.split.to_numpy() == "train"
    val = samples.split.to_numpy() == "validation"
    xt, xv, scaling = scale_fit_and_eval(raw[train], raw[val])
    y = samples["class"].to_numpy()
    write_json(out / "training_validation_scaling.json", scaling)
    old, old_predictions = authenticate_old_metrics(source, samples)
    rows, events = [], []
    cache_dir, feature_dir = out / "config_cache", out / "feature_maps_train_validation"
    cache_dir.mkdir(exist_ok=True)
    feature_dir.mkdir(exist_ok=True)
    feature_cache = {}
    configs = [("A", "none", 0., c) for c in CS]
    configs += [("G", m, s, c) for m in CENTER_MODES for s in SIGMAS for c in CS]
    mapping_started = time.perf_counter()
    for i, (family, mode, sigma, c) in enumerate(configs, 1):
        key = config_id(family, mode, sigma, c)
        folder = cache_dir / key
        folder.mkdir(exist_ok=True)
        metrics_file = folder / "metrics.json"
        if metrics_file.exists():
            row = json.loads(metrics_file.read_text())
            if row["config_id"] != key or not (folder / "predictions.npz").exists():
                raise RuntimeError(f"Incomplete config checkpoint: {key}")
            origin = "same_run_checkpoint"
        else:
            centers = centers_from_training(mode, xt) if family == "G" else None
            if centers is not None:
                map_key = f"{mode}__sigma{sigma:g}"
                if map_key not in feature_cache:
                    started = time.perf_counter()
                    ft, fv = gaussian_sum(xt, sigma, centers), gaussian_sum(xv, sigma, centers)
                    map_seconds = time.perf_counter() - started
                    feature_cache[map_key] = (ft, fv, map_seconds)
                    np.savez_compressed(feature_dir / f"{map_key}.npz", train=ft, validation=fv, centers=centers,
                                        sigma=sigma, train_sample_ids=samples.loc[train, "sample_id"].to_numpy(dtype=str),
                                        validation_sample_ids=samples.loc[val, "sample_id"].to_numpy(dtype=str))
                ft, fv, map_seconds = feature_cache[map_key]
            else:
                ft, fv, map_seconds = xt, xv, 0.
            if key in old:
                row = dict(old[key])
                if centers is not None:
                    np.testing.assert_array_equal(centers, np.asarray(row["centers"]))
                for split, mask in [("train", train), ("validation", val)]:
                    if family == "A":
                        continue
                    # The formula, per-training centers and scaled arrays were independently authenticated above.
                    assert len(samples.loc[mask]) == len(ft if split == "train" else fv)
                part = old_predictions[old_predictions.config_id == key]
                pt = part[part.split == "train"].predicted_class.to_numpy()
                pv = part[part.split == "validation"].predicted_class.to_numpy()
                origin = "previous_identical_fit"
                row["historical_fit_seconds"] = row["fit_seconds"]
            else:
                model, stats = fit_model(ft, y[train], c)
                pt, pv = model.predict(ft), model.predict(fv)
                row = {"config_id": key, "family": family, "center_mode": mode, "sigma": sigma, "C": c,
                       "centers": centers.tolist() if centers is not None else None, **stats,
                       **{f"train_{k}": v for k, v in score(y[train], pt).items()},
                       **{f"validation_{k}": v for k, v in score(y[val], pv).items()}}
                joblib.dump({"classifier": model, "config_id": key, "training_divisor": scaling["divisor"],
                             "train_sample_ids": samples.loc[train, "sample_id"].tolist()}, folder / "model.joblib", compress=3)
                origin = "new_fit"
            row.update({"unique_centers": int(len(np.unique(centers))) if centers is not None else 0,
                        "train_feature_std_min": float(ft.std(axis=0).min()),
                        "train_feature_std_max": float(ft.std(axis=0).max()),
                        "near_constant_channels_std_below_1e_8": int(np.count_nonzero(ft.std(axis=0) < 1e-8)),
                        "original_fit_origin": origin, "feature_transform_seconds": map_seconds})
            np.savez_compressed(folder / "predictions.npz", train=encoded_predictions(pt), validation=encoded_predictions(pv))
            write_json(metrics_file, row)
        events.append({"config_id": key, "action_this_execution": origin,
                       "fit_seconds_this_execution": row["fit_seconds"] if origin == "new_fit" else 0.})
        rows.append(row)
        if i % 25 == 0 or i == len(configs):
            table = pd.DataFrame(rows)
            eligible_g = table[(table.family == "G") & table.converged]
            best = rank(eligible_g).iloc[0] if not eligible_g.empty else None
            print(f"Search {i}/{len(configs)}; best validation F1=" + (f"{best.validation_macro_f1:.4f}, acc={best.validation_accuracy:.4f}, {best.config_id}" if best is not None else "not available"), flush=True)
            write_json(out / "search_progress.json", {"completed": i, "total": len(configs), "updated": now()})
    table = pd.DataFrame(rows)
    table.to_csv(out / "all_configuration_metrics.csv", index=False)
    write_json(out / "all_configuration_metrics.json", rows)
    write_json(out / "execution_events.json", events)
    table[~table.converged].to_csv(out / "nonconverged_configurations.csv", index=False)
    table[(table.family == "G") & table.converged].pipe(rank).head(30).to_csv(out / "top30_validation_configurations.csv", index=False)
    return table, events, time.perf_counter() - mapping_started


def freeze(table, out):
    pools = {"A": table[table.family == "A"],
             "G_default": table[(table.family == "G") & (table.center_mode == "uniform_0_1") & (table.sigma == .1)],
             "G_adjusted": table[table.family == "G"]}
    chosen = {method: rank(pool).iloc[0].to_dict() for method, pool in pools.items()}
    # Reload JSON records to retain native Python values and exact saved center lists.
    all_rows = {r["config_id"]: r for r in json.loads((out / "all_configuration_metrics.json").read_text())}
    chosen = {method: all_rows[row["config_id"]] for method, row in chosen.items()}
    record = {"chosen": chosen, "selection_rule": RULE, "test_used_for_selection": False,
              "test_previously_observed": True, "exploratory_screening": True,
              "validation_previously_used_for_tuning": True, "subject_splits": SUBJECTS,
              "refit_train_plus_validation": False, "frozen_at_Asia_Shanghai": now()}
    write_json(out / "parameters_frozen_before_test.json", record)
    return chosen


def final_evaluation(samples, raw, selected, source, out):
    if not (out / "parameters_frozen_before_test.json").exists():
        raise RuntimeError("Parameters must be locked before evaluating test")
    train = samples.split.to_numpy() == "train"
    labels = samples["class"].to_numpy()
    scaled = np.empty_like(raw)
    scaled[train], _, _ = scale_fit_and_eval(raw[train], raw[train])
    scalings = {}
    for split in ["validation", "test"]:
        mask = samples.split.to_numpy() == split
        _, scaled[mask], scalings[split] = scale_fit_and_eval(raw[train], raw[mask])
    write_json(out / "final_scaling.json", scalings)
    results, prediction_rows, audits, fit_log = [], [], {}, []
    arrays = {"raw": raw, "scaled": scaled, "labels": labels.astype(str),
              "splits": samples.split.to_numpy(dtype=str), "sample_ids": samples.sample_id.to_numpy(dtype=str)}
    destination = out / "models"
    destination.mkdir(exist_ok=True)
    for method in METHODS:
        config = selected[method]
        centers = centers_from_training(config["center_mode"], scaled[train]) if method != "A" else None
        start = time.perf_counter()
        features = gaussian_sum(scaled, config["sigma"], centers) if centers is not None else scaled
        feature_seconds = time.perf_counter() - start
        if features.shape != raw.shape or not np.isfinite(features).all():
            raise AssertionError("Final features must be finite N x 16")
        model, provenance, extra_fit_seconds = None, "", 0.
        candidate = out / "config_cache" / config["config_id"] / "model.joblib"
        if candidate.exists():
            bundle = joblib.load(candidate)
            if bundle["config_id"] != config["config_id"] or bundle["training_divisor"] != scalings["test"]["divisor"]:
                raise AssertionError("Candidate model settings differ")
            if bundle["train_sample_ids"] != samples.loc[train, "sample_id"].tolist():
                raise AssertionError("Candidate training samples differ")
            model, provenance = bundle["classifier"], "expanded_grid_model_reused"
        else:
            for name in METHODS:
                path = source / "models" / f"{name}.joblib"
                bundle = joblib.load(path)
                if bundle["configuration"]["config_id"] == config["config_id"]:
                    if bundle["train_sample_ids"] != samples.loc[train, "sample_id"].tolist():
                        raise AssertionError("Original model training sample mismatch")
                    model, provenance = bundle["classifier"], "original_identical_model_reused"
                    break
        if model is None:
            # The original run saved every metric but only its three winners' models.
            model, stats = fit_model(features[train], labels[train], config["C"])
            if not stats["converged"]:
                raise RuntimeError("A selected historical model could not be reconstructed with convergence")
            extra_fit_seconds = stats["fit_seconds"]
            provenance = "selected_historical_fit_reconstructed_missing_model"
        for split in ["train", "validation"]:
            mask = samples.split.to_numpy() == split
            for metric, value in score(labels[mask], model.predict(features[mask])).items():
                if not np.isclose(value, config[f"{split}_{metric}"], atol=1e-14):
                    raise AssertionError("Final model does not reproduce its selection scores")
        scaler = model.named_steps["standard_scaler"]
        assert scaler.n_samples_seen_ == train.sum()
        np.testing.assert_allclose(scaler.mean_, features[train].mean(axis=0), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scaler.var_, features[train].var(axis=0), atol=1e-12, rtol=1e-12)
        row = {"method": method, "output_dimension": 16, "config_id": config["config_id"],
               "train_samples": int(train.sum()), "validation_samples": int((samples.split == "validation").sum()),
               "test_samples": int((samples.split == "test").sum()), "center_mode": config["center_mode"],
               "sigma": config["sigma"], "C": config["C"], "centers": centers.tolist() if centers is not None else None,
               "feature_seconds_all_samples": feature_seconds, "classifier_training_seconds_original_fit": config["fit_seconds"],
               "additional_selected_model_fit_seconds_this_run": extra_fit_seconds, "model_provenance": provenance,
               "converged": True}
        for split in ["train", "validation", "test"]:
            mask = samples.split.to_numpy() == split
            start = time.perf_counter()
            probability = model.predict_proba(features[mask])
            pred = model.classes_[probability.argmax(axis=1)]
            row[f"{split}_prediction_seconds"] = time.perf_counter() - start
            row.update({f"{split}_{k}": v for k, v in score(labels[mask], pred).items()})
            row[f"{split}_prediction_counts"] = {cls: int(np.count_nonzero(pred == cls)) for cls in CLASSES}
            for i, prediction, prob in zip(np.flatnonzero(mask), pred, probability):
                sample = samples.iloc[i]
                prediction_rows.append({"method": method, "sample_id": sample.sample_id, "filename": sample.filename,
                                        "subject": int(sample.subject), "split": split, "true_class": sample["class"],
                                        "predicted_class": prediction, **{f"prob_{cls}": float(p) for cls, p in zip(model.classes_, prob)}})
            if split == "test":
                cm = confusion_matrix(labels[mask], pred, labels=CLASSES)
                pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{method}.csv")
        if centers is not None:
            audits[method] = permutation_checks(model, scaled, centers, config["sigma"], repetitions=100)
            row["order_independence"] = "PASSED"
            pd.DataFrame({"kernel": np.arange(1, 17), "center": centers, "training_mean": features[train].mean(axis=0),
                          "training_std": features[train].std(axis=0)}).to_csv(out / f"gaussian_output_statistics_{method}.csv", index=False)
        else:
            row["order_independence"] = "not applicable"
        joblib.dump({"classifier": model, "method": method, "centers": centers, "sigma": config["sigma"],
                     "regularization_C": config["C"], "input_divisor": scalings["test"]["divisor"],
                     "class_order": CLASSES, "configuration": config,
                     "train_sample_ids": samples.loc[train, "sample_id"].tolist()}, destination / f"{method}.joblib", compress=3)
        arrays[method] = features
        results.append(row)
        fit_log.append({"method": method, "config_id": config["config_id"], "provenance": provenance,
                        "additional_fit_seconds": extra_fit_seconds})
    write_json(out / "final_model_fit_provenance.json", fit_log)
    write_json(out / "order_independence_checks.json", audits)
    write_json(out / "complete_metrics.json", results)
    pd.DataFrame(results).to_csv(out / "results_table.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "sample_predictions.csv", index=False)
    np.savez_compressed(out / "features_arrays.npz", **arrays)
    return pd.DataFrame(results)


def metrics_from_encoded(actual, predicted):
    cm = np.bincount(6 * np.asarray(actual, dtype=int) + np.asarray(predicted, dtype=int), minlength=36).reshape(6, 6)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    per_class_f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(6, dtype=float), where=denominator != 0)
    return {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(per_class_f1.mean())}


def verify_results(out):
    data = np.load(out / "features_arrays.npz")
    samples = pd.read_csv(out / "sample_manifest.csv")
    table = pd.read_csv(out / "all_configuration_metrics.csv")
    records = json.loads((out / "all_configuration_metrics.json").read_text())
    by_id = {r["config_id"]: r for r in records}
    selected = json.loads((out / "parameters_frozen_before_test.json").read_text())
    result = {r["method"]: r for r in json.loads((out / "complete_metrics.json").read_text())}
    predictions = pd.read_csv(out / "sample_predictions.csv")
    assert len(table) == len(by_id) == len(CS) * (1 + len(CENTER_MODES) * len(SIGMAS))
    assert not selected["test_used_for_selection"] and selected["test_previously_observed"]
    assert data["raw"].shape == (len(samples), 16)
    np.testing.assert_array_equal(samples.sample_id.to_numpy(dtype=str), data["sample_ids"])
    np.testing.assert_array_equal(samples["class"].to_numpy(dtype=str), data["labels"])
    np.testing.assert_array_equal(samples.split.to_numpy(dtype=str), data["splits"])
    train = samples.split.to_numpy() == "train"
    xt, _, _ = scale_fit_and_eval(data["raw"][train], data["raw"][train])
    np.testing.assert_array_equal(xt, data["scaled"][train])
    for split in ["validation", "test"]:
        mask = samples.split.to_numpy() == split
        _, expected, _ = scale_fit_and_eval(data["raw"][train], data["raw"][mask])
        np.testing.assert_array_equal(expected, data["scaled"][mask])
    for row in records:
        saved = np.load(out / "config_cache" / row["config_id"] / "predictions.npz")
        for split in ["train", "validation"]:
            actual = encoded_predictions(samples.loc[samples.split == split, "class"])
            assert saved[split].shape == actual.shape
            for key, value in metrics_from_encoded(actual, saved[split]).items():
                assert np.isclose(value, row[f"{split}_{key}"], atol=1e-14)
    for method in METHODS:
        config = selected["chosen"][method]
        pool = table[table.family == ("A" if method == "A" else "G")]
        if method == "G_default":
            pool = pool[(pool.center_mode == "uniform_0_1") & (pool.sigma == .1)]
        assert rank(pool).iloc[0].config_id == config["config_id"]
        bundle = joblib.load(out / "models" / f"{method}.joblib")
        if method == "A":
            expected = data["scaled"]
        else:
            centers = centers_from_training(config["center_mode"], xt)
            np.testing.assert_array_equal(centers, bundle["centers"])
            expected = gaussian_sum(data["scaled"], config["sigma"], centers)
        np.testing.assert_array_equal(expected, data[method])
        model = bundle["classifier"]
        scaler = model.named_steps["standard_scaler"]
        assert scaler.n_samples_seen_ == train.sum()
        np.testing.assert_allclose(scaler.mean_, expected[train].mean(axis=0), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scaler.var_, expected[train].var(axis=0), atol=1e-12, rtol=1e-12)
        for split in SUBJECTS:
            mask = samples.split.to_numpy() == split
            part = predictions[(predictions.method == method) & (predictions.split == split)]
            assert part.sample_id.tolist() == samples.loc[mask, "sample_id"].tolist()
            assert part.true_class.tolist() == samples.loc[mask, "class"].tolist()
            pred = model.predict(expected[mask])
            np.testing.assert_array_equal(pred, part.predicted_class.to_numpy())
            np.testing.assert_allclose(model.predict_proba(expected[mask]), part[[f"prob_{c}" for c in model.classes_]], atol=1e-12, rtol=1e-10)
            for key, value in score(part.true_class, pred).items():
                assert np.isclose(value, result[method][f"{split}_{key}"], atol=1e-14)
        cm = pd.read_csv(out / f"confusion_{method}.csv", index_col=0)
        assert cm.index.tolist() == CLASSES and cm.columns.tolist() == CLASSES
        assert np.isclose(np.trace(cm) / cm.to_numpy().sum(), result[method]["test_accuracy"])
    assert all(a["passed"] and a["prediction_changes"] == 0 for a in json.loads((out / "order_independence_checks.json").read_text()).values())
    return {"status": "PASSED", "configuration_predictions_recomputed": len(records),
            "final_models_and_predictions_rechecked": 3, "train_only_scaling_centers_StandardScaler_verified": True,
            "selection_reproduced_without_test_scores": True, "classifier_fits_during_verification": 0,
            "checked_at_Asia_Shanghai": now()}


def render(out, source):
    chinese = setup_plot_font()
    results = pd.DataFrame(json.loads((out / "complete_metrics.json").read_text()))
    prior = pd.DataFrame(json.loads((source / "complete_metrics.json").read_text()))
    comparison = []
    for prefix, table in [("previous", prior), ("expanded", results)]:
        for _, r in table.iterrows():
            comparison.append({"version": prefix, "method": r.method,
                               **{k: r[k] for k in ["train_accuracy", "validation_accuracy", "validation_macro_f1",
                                                    "test_accuracy", "test_macro_f1", "center_mode", "sigma", "C"]}})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(out / "comparison_previous_vs_expanded.csv", index=False)
    table = pd.read_csv(out / "all_configuration_metrics.csv")
    g = table[(table.family == "G") & table.converged]
    heat = np.full((len(CENTER_MODES), len(SIGMAS)), np.nan)
    for i, mode in enumerate(CENTER_MODES):
        for j, sigma in enumerate(SIGMAS):
            part = g[(g.center_mode == mode) & (g.sigma == sigma)]
            if not part.empty:
                heat[i, j] = rank(part).iloc[0].validation_macro_f1
    fig, ax = plt.subplots(figsize=(16, 7), dpi=200)
    image = ax.imshow(heat, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set(xticks=range(len(SIGMAS)), xticklabels=[f"{s:g}" for s in SIGMAS],
           yticks=range(len(CENTER_MODES)), yticklabels=CENTER_MODES,
           xlabel="统一宽度 sigma" if chinese else "Shared width sigma",
           title="验证Macro-F1参数热图（每格按验证选择C，不含测试分数）" if chinese else "Validation Macro-F1 (best validation-selected C in each cell; no test scores)")
    for i in range(len(CENTER_MODES)):
        for j in range(len(SIGMAS)):
            ax.text(j, i, f"{heat[i, j]:.3f}" if np.isfinite(heat[i, j]) else "N/C", ha="center", va="center",
                    color="white" if heat[i, j] > np.nanmax(heat) * .63 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Validation Macro-F1")
    fig.tight_layout()
    fig.savefig(out / "validation_parameter_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=180, sharex=True, sharey=True)
    for ax, mode in zip(axes.flat, CENTER_MODES):
        for c in CS:
            part = g[(g.center_mode == mode) & (g.C == c)].sort_values("sigma")
            ax.plot(part.sigma, part.validation_macro_f1, "o-", ms=3, linewidth=1, label=f"C={c:g}")
        ax.set_xscale("log")
        ax.set(title=mode, xlabel="sigma", ylabel="Validation Macro-F1")
        ax.grid(alpha=.2)
    axes.flat[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("扩展Gaussian参数比较：只用验证集" if chinese else "Expanded Gaussian search: validation only")
    fig.tight_layout()
    fig.savefig(out / "validation_parameter_curves.png", bbox_inches="tight")
    plt.close(fig)
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
        cm = pd.read_csv(out / f"confusion_{method}.csv", index_col=0).to_numpy()
        draw_cm(ax, cm, f"KTH expanded {method}: " + ("测试混淆矩阵" if chinese else "test confusion"), chinese)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{method}.png", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    positions = np.arange(len(comparison))
    for offset, metric, label, color in [(-.18, "validation_accuracy", "Validation", "#2878a3"),
                                         (.18, "test_accuracy", "Test", "#dc7a36")]:
        bars = ax.bar(positions + offset, comparison[metric] * 100, .34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set(xticks=positions, xticklabels=[f"{r.version}\n{r.method}" for _, r in comparison.iterrows()],
           ylim=(0, 100), ylabel="Accuracy (%)",
           title="扩展前后对比（单次验证划分，无虚构误差线）" if chinese else "Before / after expanded search (single validation split)")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "performance_comparison.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 4), dpi=220)
    ax.axis("off")
    cells = [[r.version, r.method, f"{r.train_accuracy:.2%}", f"{r.validation_accuracy:.2%}",
              f"{r.validation_macro_f1:.4f}", f"{r.test_accuracy:.2%}", f"{r.test_macro_f1:.4f}",
              f"{r.center_mode}; s={r.sigma:g}; C={r.C:g}"] for _, r in comparison.iterrows()]
    grid = ax.table(cellText=cells, colLabels=["Version", "Method", "Train acc", "Val acc", "Val F1", "Test acc", "Test F1", "Parameters"],
                    colWidths=[.08, .11, .09, .09, .08, .09, .08, .29], loc="center", cellLoc="center")
    grid.auto_set_font_size(False)
    grid.set_fontsize(8)
    grid.scale(1, 2)
    for (i, j), cell in grid.get_celld().items():
        cell.set_facecolor("#e8eef4" if i == 0 else ("#f7f9fc" if i % 2 else "white"))
    ax.set_title("KTH previous and expanded Gaussian search", pad=12)
    fig.tight_layout()
    fig.savefig(out / "results_table.png", bbox_inches="tight")
    plt.close(fig)
    # Actual selected centers and width, independent of the test labels.
    config = json.loads((out / "parameters_frozen_before_test.json").read_text())["chosen"]["G_adjusted"]
    centers = np.asarray(config["centers"])
    xs = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    for mu in centers:
        ax.plot(xs, np.exp(-np.square(xs - mu) / (2 * config["sigma"] ** 2)), linewidth=1)
    ax.set(xlabel="Scaled scalar input", ylabel="Gaussian response (peak=1)",
           title=f"Validation-selected 16 Gaussian kernels: {config['center_mode']}, sigma={config['sigma']:g}")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "selected_gaussian_kernels.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/kth_six_subject_split_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("data/kth"))
    parser.add_argument("--output", type=Path, default=Path("outputs/kth_gaussian_expanded_v2"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    out, source, project = args.output, args.source, Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An expanded KTH experiment is already running in this output directory.")
        if args.verify_only:
            checks = verify_results(out)
            write_json(out / "postrun_verification.json", checks)
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return
        if args.render_only:
            render(out, source)
            print(f"Rendered existing metrics without any fits: {out.resolve()}")
            return
        started = time.perf_counter()
        source_files = ["run_signature.json", "feature_cache.joblib", "features_arrays.npz", "sample_manifest.csv",
                        "all_configuration_metrics.json", "all_configuration_predictions_train_validation.csv",
                        "complete_metrics.json", "parameters_frozen_before_test.json"]
        signature = {"source_files": {name: sha256(source / name) for name in source_files},
                     "script_sha256": sha256(__file__), "classifier_code_sha256": sha256(project / "run_kth.py"),
                     "Gaussian_code_sha256": sha256(project / "run_experiment.py"),
                     "scale_code_sha256": sha256(project / "tune_gaussian.py"),
                     "numpy": np.__version__, "sklearn": sklearn.__version__, "solver": SOLVER,
                     "seed": SEED, "center_modes": CENTER_MODES, "sigmas": SIGMAS, "C": CS,
                     "subject_splits": SUBJECTS, "selection_rule": RULE}
        signature_path = out / "run_signature.json"
        if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
            raise RuntimeError("Inputs, code, environment or settings changed. Use a new output directory to preserve previous results.")
        if (out / "COMPLETE.json").exists():
            print("Identical completed expanded search reused; no decoding or fitting repeated.")
            print(pd.read_csv(out / "results_table.csv")[["method", "validation_accuracy", "test_accuracy", "test_macro_f1"]].to_string(index=False))
            return
        write_json(signature_path, signature)
        write_json(out / "experiment_config.json", {"created_at_Asia_Shanghai": now(), "signature": signature,
                   "candidate_configurations": len(CS) * (1 + len(CENTER_MODES) * len(SIGMAS)),
                   "fixed_kernel_count": 16, "fixed_peak": 1, "output_dimension": 16,
                   "input_preprocessing_unchanged": True, "same_sigma_for_all_16_kernels": True,
                   "gradient_learning": False, "test_and_validation_previously_observed": True,
                   "not_an_independent_final_validation": True,
                   "nonconvergence_policy": "record every warning/iteration count; exclude unconverged fits, no silent retry or tolerance change",
                   "coincident_center_policy": "retain duplicates and all 16 channels"})
        samples, raw, payload = audit_source(project, source, args.data_root, out)
        with threadpool_limits(limits=1):
            table, events, search_seconds = search(samples, raw, source, out)
            selected = freeze(table, out)
            results = final_evaluation(samples, raw, selected, source, out)
            verification = verify_results(out)
        write_json(out / "postrun_verification.json", verification)
        render(out, source)
        if signature["source_files"] != {name: sha256(source / name) for name in source_files}:
            raise RuntimeError("Original result files changed during expanded screening")
        execution_counts = dict(Counter(e["action_this_execution"] for e in events))
        fits = json.loads((out / "final_model_fit_provenance.json").read_text())
        runtime = {"started_date_Asia_Shanghai": now()[:10], "finished_at_Asia_Shanghai": now(),
                   "CPU": platform.processor() or platform.machine(), "logical_cpus": psutil.cpu_count(),
                   "memory_total_bytes": psutil.virtual_memory().total, "gpu_used": False, "BLAS_threads": 1,
                   "numpy": np.__version__, "sklearn": sklearn.__version__, "python": platform.python_version(),
                   "candidate_configurations": len(table), "execution_counts": execution_counts,
                   "nonconverged_candidates": int((~table.converged).sum()), "video_decodes_this_run": 0,
                   "preprocessing_seconds_this_run": 0., "historical_preprocessing_seconds": payload["preprocessing_seconds"],
                   "grid_fit_seconds_this_run": sum(e["fit_seconds_this_execution"] for e in events),
                   "additional_selected_model_fit_seconds": sum(e["additional_fit_seconds"] for e in fits),
                   "search_wall_seconds": search_seconds, "experiment_wall_seconds": time.perf_counter() - started,
                   "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                   "prior_results_unchanged": True, "classification_remains_exploratory": True}
        write_json(out / "runtime.json", runtime)
        (out / "run_command.txt").write_text("cd " + shlex.quote(str(Path.cwd())) + "\n" + shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
        snapshot = out / "source_code"
        snapshot.mkdir(exist_ok=True)
        for name in ["tune_kth_gaussian.py", "run_kth.py", "run_experiment.py", "tune_gaussian.py", "requirements.txt"]:
            shutil.copy2(project / name, snapshot / name)
        write_json(out / "COMPLETE.json", {"status": "COMPLETE", "completed_at_Asia_Shanghai": now(),
                                            "run_signature_sha256": sha256(signature_path)})
        print(results[["method", "train_accuracy", "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1", "center_mode", "sigma", "C"]].to_string(index=False))
        print(f"Saved: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
