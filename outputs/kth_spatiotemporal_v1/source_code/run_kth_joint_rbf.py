#!/usr/bin/env python3
"""KTH software transfer of the ORIGINAL measured-bank joint RBF pipeline.

The upstream project is imported read-only. No hardware access, new video
preprocessing, downloaded data, PenDigits weights, or summed scalar Gaussians.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

CLASSES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
HEAD_ALPHAS = [1e-8, 1e-6, 1e-4, .001, .01, .1, 1., 10., 100.]
SIGMAS = [.05, .1, .15, .2, .25, .3, .35, .5, .7, 1.]
CAL_ALPHAS = [.01, .1, 1.]
FACTOR_UPPERS = [1., 1.2]
UPSTREAM_FILES = ["modeling/joint_gaussian.py", "modeling/ridge_classifier.py",
                  "inference/preprocessing.py", "inference/empirical_response.py"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=jsonable, allow_nan=False), encoding="utf-8")


def score(y, prediction):
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=range(6),
                                      average="macro", zero_division=0))}


def key_of(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def make_transform(config, api):
    return api["transform"](dimensions=16, basis_per_dimension=16, output_features=256,
                            joint_sigma=config["sigma"],
                            calibration_alpha=config["calibration_alpha"],
                            factor_lower=1e-6, factor_upper=config["factor_upper"],
                            epsilon=1e-12, random_state=42, kmeans_max_iter=300)


def fit_head(x, labels, alpha, api):
    model = api["ridge"](input_features=x.shape[1], classes=list(range(6)),
                         ridge_alpha=alpha, fit_intercept=True,
                         regularize_intercept=False, device=torch.device("cpu"),
                         dtype=torch.float64)
    return model.fit(torch.as_tensor(np.asarray(x, dtype=np.float64)),
                     torch.as_tensor(labels, dtype=torch.long))


def numpy_scores(x, state):
    weights = np.asarray(state["combined_weights"], dtype=np.float64)
    return np.column_stack((np.ones(len(x)), x)) @ weights


def features_from_payload(payload, q, h, api):
    config = payload["config"]
    family = config["family"]
    if family == "linear":
        return q.astype(np.float64)
    if family == "linear_standardized":
        return (q - payload["mean"]) / payload["std"]
    if family == "response_direct":
        return (h - payload["mean"]) / payload["std"]
    transform = api["transform"].from_state_dict(payload["transform"])
    return (transform.ideal_pc_transform(q) if family == "ideal"
            else transform.transform(h))


def inspect_alignment(source, arrays, manifest):
    ids, labels, splits, raw = [arrays[k] for k in ("sample_ids", "labels", "splits", "raw")]
    assert raw.shape == (len(ids), 16) and len(set(ids)) == len(ids)
    assert np.isfinite(raw).all() and (raw >= 0).all()
    assert manifest.sample_id.tolist() == ids.tolist()
    assert manifest["class"].tolist() == labels.tolist()
    assert manifest.split.tolist() == splits.tolist()
    expected = {"train": {11, 12, 13, 14, 15, 16, 17, 18},
                "validation": {19, 20, 21, 23, 24, 25, 1, 4},
                "test": {22, 2, 3, 5, 6, 7, 8, 9, 10}}
    assert all(set(manifest.loc[manifest.split == k, "subject"]) == v
               for k, v in expected.items())
    assert manifest.groupby("video_id").split.nunique().max() == 1
    assert manifest.groupby("video_sha256").split.nunique().max() == 1
    # Verify bytes without decoding or changing the cached visual representation.
    unique = manifest.drop_duplicates("path")
    missing, changed = [], []
    for row in unique.itertuples():
        path = Path(row.path)
        if not path.is_file():
            missing.append(str(path))
        elif sha256(path) != row.video_sha256:
            changed.append(str(path))
    if changed:
        raise ValueError(f"Source video bytes changed: {changed[:3]}")
    # Reuse the already completed independent decoder check only with verified bytes.
    independent = json.loads((source / "independent_video_audit.json").read_text())
    assert all(row["passed"] for row in independent)
    cache = joblib.load(source / "feature_cache.joblib")
    records = {r["sample_id"]: r for r in cache["records"] if r.get("included")}
    for index, sample_id in enumerate(ids):
        assert np.array_equal(np.asarray(records[sample_id]["feature"]), raw[index])
    return {"passed": True, "shape": list(raw.shape), "unique_ids": len(ids),
            "nan_inf": int((~np.isfinite(raw)).sum()),
            "all_zero_rows": int(np.all(raw == 0, axis=1).sum()),
            "cache_records_exactly_equal": True,
            "subject_video_hash_split_disjoint": True,
            "source_videos_hashed": len(unique) - len(missing),
            "missing_videos": missing, "changed_video_bytes": changed,
            "reused_independent_video_audit": str(source / "independent_video_audit.json"),
            "audit_applicable_to_available_identical_videos": not missing,
            "sampling_span_min_max_seconds": [float(manifest.actual_sample_span_seconds.min()),
                                                  float(manifest.actual_sample_span_seconds.max())],
            "repeated_frame_indices": int(manifest.repeated_index_count.sum()),
            "legacy_preprocessing_seconds": cache["preprocessing_seconds"],
            "new_video_decode_seconds": 0.,
            "note": "Existing floating-point frame-difference cache reused; no new extraction."}


def selection_key(row):
    # Declared before any test evaluation; validation only, deterministic final ties.
    return (-row["validation_macro_f1"], -row["validation_accuracy"],
            row["head_alpha"], row.get("sigma", 0),
            row.get("calibration_alpha", 0), row.get("factor_upper", 0), row["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/kth_six_subject_split_v1"))
    parser.add_argument("--hardware-root", type=Path,
                        default=Path("/home/ziyan/SURF2026/rbf_hardware"))
    parser.add_argument("--output", type=Path, default=Path("outputs/kth_joint_rbf256_v1"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    assert args.repeats > 0
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    args.source, args.hardware_root, args.output = [p.resolve() for p in
                                                (args.source, args.hardware_root, args.output)]
    sys.path.insert(0, str(args.hardware_root / "src"))
    from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer, benchmark_kmeans, linear_cka
    from rbf_hardware.modeling.ridge_classifier import RidgeLinearClassifier
    from rbf_hardware.inference.empirical_response import EmpiricalHardwareResponseBank
    from rbf_hardware.inference.preprocessing import DifferentialLevelQuantizer
    api = {"transform": FullJointGaussianTransformer, "ridge": RidgeLinearClassifier}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("models", "search_cache", "figures", "source_code"):
        (args.output / name).mkdir(exist_ok=True)
    bank_path = args.hardware_root / "dataset/empirical/hardware_response_samples.npz"
    levels_path = args.hardware_root / "dataset/csv/differential_levels.csv"
    checkpoint_path = args.hardware_root / "checkpoints/weights.pt"
    source_paths = [args.source / "features_arrays.npz", args.source / "sample_manifest.csv",
                    args.source / "feature_cache.joblib", args.source / "independent_video_audit.json",
                    bank_path, levels_path, checkpoint_path, Path(__file__).resolve()]
    source_paths += [args.hardware_root / "src/rbf_hardware" / s for s in UPSTREAM_FILES]
    fingerprint = {str(p): sha256(p) for p in source_paths}
    signature = {"source_hashes": fingerprint, "seed": 42, "repeats": args.repeats,
                 "versions": {p: importlib.metadata.version(p) for p in
                              ("numpy", "torch", "scikit-learn", "pandas", "scipy")}}
    signature_path = args.output / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("Output belongs to different inputs/code/environment. Use a new --output.")
    else:
        write_json(signature_path, signature)
    if (args.output / "COMPLETE.json").exists() and not args.verify_only:
        print("Conditions unchanged; completed run reused. Use --verify-only for saved-model checks.", flush=True)
        print((args.output / "results_table.csv").read_text(), flush=True)
        return
    archive = np.load(args.source / "features_arrays.npz", allow_pickle=True)
    raw = archive["raw"]
    labels = np.array([CLASSES.index(v) for v in archive["labels"]])
    splits = archive["splits"].astype(str)
    masks = {name: splits == name for name in ("train", "validation", "test")}
    manifest = pd.read_csv(args.source / "sample_manifest.csv")
    quantizer = DifferentialLevelQuantizer.load(levels_path, input_features=16,
                                               expected_levels=256, raw_minimum=0, raw_maximum=100)
    bank = EmpiricalHardwareResponseBank.load(bank_path, expected_groups=16,
                                              expected_levels=quantizer.levels)
    maximum = float(raw[masks["train"]].max())
    denominator = maximum if maximum > 0 else 1.
    normalized = np.clip(raw / denominator, 0., 1.)
    q = quantizer.transform(normalized * 100.)
    response_start = time.perf_counter()
    h = bank.simulate(q, sampling_mode="mean", random_generator=np.random.default_rng(42))
    response_seconds = time.perf_counter() - response_start
    assert h.shape == (len(q), 256) and np.isfinite(h).all()
    if args.verify_only:
        verify_saved(args.output, q, h, labels, splits, api)
        return
    print("Checking cached features, original video hashes, labels and official subject splits...", flush=True)
    checks = inspect_alignment(args.source, archive, manifest)
    write_json(args.output / "input_checks.json", checks)
    manifest.to_csv(args.output / "sample_manifest.csv", index=False)
    manifest.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(
        columns=CLASSES).to_csv(args.output / "class_counts.csv")
    for name in ("exclusions.csv", "official_missing_videos.csv", "run_signature.json",
                 "independent_video_audit.json", "decoder_warning_audit.json", "data_and_cost_stats.json"):
        shutil.copy2(args.source / name, args.output / ("previous_" + name))
    shutil.copy2(__file__, args.output / "source_code" / Path(__file__).name)
    for relative in UPSTREAM_FILES:
        destination = args.output / "source_code/upstream" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.hardware_root / "src/rbf_hardware" / relative, destination)
    write_json(args.output / "scaling_and_adapter.json", {
        "fit_split": "train", "raw_global_maximum": maximum, "denominator": denominator,
        "formula": "q=original_quantizer(100*clip(raw/train_global_max,0,1))",
        "reason": "Original raw-coordinate interface is [0,100]; video scaling is the existing encoder.",
        "levels": quantizer.levels,
        "clipped_value_fractions": {k: float(np.mean((raw[v]/denominator > 1) |
                                                     (raw[v]/denominator < 0))) for k, v in masks.items()},
        "quantization_mean_absolute_error": float(np.abs(q-normalized).mean()),
        "quantization_max_absolute_error": float(np.abs(q-normalized).max()),
        "no_test_or_validation_fit": True})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    defaults = checkpoint["feature_transform"]
    default_cfg = {"family": "hardware", "sigma": float(defaults["joint_sigma"]),
                   "calibration_alpha": float(defaults["calibration_alpha"]),
                   "factor_upper": float(defaults["factor_upper"])}
    assert default_cfg == {"family": "hardware", "sigma": .25, "calibration_alpha": 1., "factor_upper": 1.}
    default_head_alpha = float(checkpoint["ridge_alpha"])
    assert default_head_alpha == 1e-8
    write_json(args.output / "preregistered_search.json", {
        "created_before_test_evaluation": True, "head_alphas": HEAD_ALPHAS,
        "joint_target_sigmas": SIGMAS, "calibration_alphas": CAL_ALPHAS,
        "factor_uppers": FACTOR_UPPERS, "factor_lower": 1e-6,
        "default_from_saved_checkpoint_not_dirty_config": default_cfg,
        "default_head_alpha": default_head_alpha,
        "physical_bank_curves_fixed": True, "centers": "256 training-only original Torch KMeans centers",
        "selection": "highest validation Macro-F1; then accuracy; then smaller head alpha, sigma, calibration alpha, upper; id",
        "train_validation_protocol": "Original KTH subject split; train 8 only. No train+validation refit.",
        "test_status": "Exploratory: this dataset test split was examined in earlier different-model experiments.",
        "precision": "Original transformer float64 then float32; original Ridge head CPU float64.",
        "bank_mean": "400-cycle mean; current upstream mean mode adds NO synthetic noise.",
        "stress_tests": {"frozen_models": True, "repeats": args.repeats,
                         "seeds": list(range(1042, 1042 + args.repeats)),
                         "conditions": {"recorded_cycle_only": [0., 0.],
                                        "recorded_cycle_plus_simulator_noise": [.01, .05]}}})
    center_start = time.perf_counter()
    centers = benchmark_kmeans(q[masks["train"]], 256, 42, 300)
    center_seconds = time.perf_counter() - center_start
    np.save(args.output / "centers.npy", centers)
    np.savez_compressed(args.output / "input_and_bank_mean.npz", raw=raw, scaled=normalized,
                        quantized=q, bank_mean=h, labels=labels, splits=splits,
                        sample_ids=archive["sample_ids"].astype(str))
    print(f"Checks passed. Train/validation/test = {[int(m.sum()) for m in masks.values()]}; fitting 657 candidates.", flush=True)
    rows, refit_count, reuse_count = [], 0, 0
    transform_states = {}
    transform_timing = {}
    configurations = [{"family": name} for name in ("linear", "linear_standardized", "response_direct")]
    configurations += [{"family": "ideal", "sigma": s, "calibration_alpha": 1., "factor_upper": 1.}
                       for s in SIGMAS]
    configurations += [{"family": "hardware", "sigma": s, "calibration_alpha": a, "factor_upper": u}
                       for s in SIGMAS for a in CAL_ALPHAS for u in FACTOR_UPPERS]
    search_start = time.perf_counter()
    for number, config in enumerate(configurations):
        transform_id = key_of(config)
        payload = {"config": config, "fit_split": "train", "hardware_root": str(args.hardware_root)}
        fit_start = time.perf_counter()
        family = config["family"]
        if family == "linear":
            features = q.astype(np.float64)
        elif family in ("linear_standardized", "response_direct"):
            base = q.astype(np.float64) if family == "linear_standardized" else h.astype(np.float64)
            payload["mean"] = base[masks["train"]].mean(axis=0)
            payload["std"] = np.maximum(base[masks["train"]].std(axis=0), 1e-12)
            features = (base - payload["mean"]) / payload["std"]
        else:
            # Upper clipping does not enter calibration fit; reuse an identical fit
            # across the two clipping settings, never across train/val/test fits.
            fit_key = (config["sigma"], config["calibration_alpha"])
            if fit_key not in transform_states:
                transform = make_transform(config, api).fit(h[masks["train"]], q[masks["train"]],
                                                            preset_centers=centers)
                transform_states[fit_key] = transform.state_dict()
            state = dict(transform_states[fit_key], factor_upper=config["factor_upper"])
            transform = api["transform"].from_state_dict(state)
            payload["transform"] = transform.state_dict()
            # Test feature transformation is deterministic and unsupervised; test
            # labels/scores are not accessed until locked configuration evaluation.
            features = (transform.ideal_pc_transform(q) if family == "ideal"
                        else transform.transform(h))
        assert features.shape == (len(q), 16 if family.startswith("linear") else 256)
        assert np.isfinite(features).all()
        transform_timing[transform_id] = time.perf_counter() - fit_start
        for alpha in HEAD_ALPHAS:
            full_config = dict(config, head_alpha=alpha)
            candidate_id = key_of(full_config)
            cache_path = args.output / "search_cache" / (candidate_id + ".joblib")
            if cache_path.exists():
                cached = joblib.load(cache_path)
                assert cached["config"] == full_config
                row = cached["metrics"]
                reuse_count += 1
            else:
                head_start = time.perf_counter()
                head = fit_head(features[masks["train"]], labels[masks["train"]], alpha, api)
                fit_seconds = time.perf_counter() - head_start
                row = dict(full_config, id=candidate_id, output_dimension=features.shape[1],
                           head_fit_seconds=fit_seconds,
                           feature_fit_and_all_transform_seconds=transform_timing[transform_id])
                cached = dict(payload, config=full_config, head=head.cpu_state_dict())
                for split in ("train", "validation"):
                    predictions = head.predict(torch.as_tensor(features[masks[split]])).numpy()
                    row.update({split+"_"+k: v for k, v in score(labels[masks[split]], predictions).items()})
                    cached[split+"_predictions"] = predictions
                cached["metrics"] = row
                joblib.dump(cached, cache_path, compress=3)
                refit_count += 1
            rows.append(row)
        if (number+1) % 5 == 0 or number == len(configurations)-1:
            print(f"Feature settings {number+1}/{len(configurations)}; heads {len(rows)}/657; elapsed {time.perf_counter()-search_start:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(args.output / "validation_search.csv", index=False)
    search_seconds = time.perf_counter() - search_start
    def choose(family, predicate=lambda r: True):
        return min((r for r in rows if r["family"] == family and predicate(r)), key=selection_key)
    def is_default(row):
        return all(row[k] == v for k, v in default_cfg.items())
    hardware_best = choose("hardware")
    chosen = {
        "Linear16": choose("linear"),
        "Linear16_standardized": choose("linear_standardized"),
        "Response256_direct_diagnostic": choose("response_direct"),
        "Ideal_RBF256_default": choose("ideal", lambda r: r["sigma"] == .25 and r["head_alpha"] == 1e-8),
        "Ideal_RBF256_tuned": choose("ideal"),
        "Hardware_joint256_default": choose("hardware", lambda r: is_default(r) and r["head_alpha"] == default_head_alpha),
        "Hardware_joint256_default_head_tuned": choose("hardware", is_default),
        "Hardware_joint256_tuned": hardware_best,
        "Ideal_RBF256_matched_to_hardware": choose("ideal", lambda r: r["sigma"] == hardware_best["sigma"]
                                                   and r["head_alpha"] == hardware_best["head_alpha"]),
    }
    write_json(args.output / "locked_parameters_before_test.json", chosen)
    print("Validation selection locked; now evaluating the original test subjects.", flush=True)
    final_rows, per_sample, per_subject, models, matrices = [], [], [], {}, {}
    for name, row in chosen.items():
        payload = joblib.load(args.output / "search_cache" / (row["id"]+".joblib"))
        feature_start = time.perf_counter()
        features = features_from_payload(payload, q, h, api)
        feature_seconds = time.perf_counter() - feature_start
        # Reuse the selected trained head, not a second fit with identical data.
        state = payload["head"]
        payload["scaling_maximum"] = maximum
        payload["classes"] = CLASSES
        joblib.dump(payload, args.output / "models" / (name+".joblib"), compress=3)
        models[name] = payload
        result = dict(row, scheme=name, train_n=int(masks["train"].sum()),
                      validation_n=int(masks["validation"].sum()), test_n=int(masks["test"].sum()),
                      all_feature_transform_seconds=feature_seconds)
        for split, mask in masks.items():
            predict_start = time.perf_counter()
            scores = numpy_scores(features[mask], state)
            predicted = scores.argmax(axis=1)
            result[split+"_predict_seconds"] = time.perf_counter() - predict_start
            result.update({split+"_"+k: v for k, v in score(labels[mask], predicted).items()})
            part = manifest.loc[mask, ["sample_id", "video_id", "subject", "class", "split"]].copy()
            part["scheme"] = name
            part["predicted"] = [CLASSES[i] for i in predicted]
            part["correct"] = predicted == labels[mask]
            for c, cls in enumerate(CLASSES):
                part["score_"+cls] = scores[:, c]
            per_sample.append(part)
            for person, local in part.groupby("subject"):
                metrics = score([CLASSES.index(v) for v in local["class"]],
                                [CLASSES.index(v) for v in local.predicted])
                per_subject.append(dict(scheme=name, split=split, subject=int(person), n=len(local), **metrics))
            if split == "test":
                matrix = confusion_matrix(labels[mask], predicted, labels=range(6))
                assert np.trace(matrix) == (predicted == labels[mask]).sum()
                matrices[name] = matrix
                pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(args.output/("confusion_"+name+".csv"))
                write_json(args.output/("classification_report_"+name+".json"),
                           classification_report(labels[mask], predicted, labels=range(6),
                                                 target_names=CLASSES, output_dict=True, zero_division=0))
        final_rows.append(result)
        np.savez_compressed(args.output / "models" / (name+"_features.npz"), features=features,
                            sample_ids=archive["sample_ids"].astype(str), splits=splits, labels=labels)
    table = pd.DataFrame(final_rows)
    table.to_csv(args.output / "results_table.csv", index=False)
    predictions = pd.concat(per_sample, ignore_index=True)
    predictions.to_csv(args.output / "sample_predictions.csv", index=False)
    pd.DataFrame(per_subject).to_csv(args.output / "per_subject_metrics.csv", index=False)
    write_json(args.output / "complete_metrics.json", final_rows)
    print(table[["scheme", "train_accuracy", "validation_accuracy", "test_accuracy", "test_macro_f1"]].to_string(index=False), flush=True)
    model_checks = mathematical_checks(models, q, h, masks, api, args.output)
    assert model_checks["passed"]
    robustness = run_stress_tests(models, bank, q[masks["test"]], labels[masks["test"]],
                                  args.repeats, api, args.output)
    paired = subject_bootstrap(predictions)
    write_json(args.output / "paired_subject_bootstrap.json", paired)
    make_figures(args.output, table, pd.DataFrame(rows), matrices, models, q, h, masks, robustness, api)
    verify_saved(args.output, q, h, labels, splits, api)
    assert all(sha256(Path(path)) == digest for path, digest in fingerprint.items())
    costs = {"wall_seconds": time.perf_counter()-start, "search_seconds": search_seconds,
             "bank_mean_response_all_seconds": response_seconds, "centers_fit_seconds": center_seconds,
             "new_head_fits": refit_count, "identical_cached_head_fits": reuse_count,
             "raw_cache_bytes": raw.nbytes, "quantized_input_bytes": q.nbytes,
             "bank_response_bytes": h.nbytes, "one_joint_feature_matrix_bytes": len(q)*256*4,
             "bank_disk_bytes": bank_path.stat().st_size,
             "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             "system_ram_bytes": psutil.virtual_memory().total, "cpu_logical": psutil.cpu_count(),
             "cpu": cpu_name(), "torch_threads": torch.get_num_threads(), "gpu_used": False,
             "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
             "source_preprocessing_seconds_previous_run": checks["legacy_preprocessing_seconds"],
             "video_download_or_reextraction": False,
             "timing_note": "Wall includes hashing, saving, diagnostics and figures. Peak is actual process RSS, not storage."}
    write_json(args.output / "cost_and_environment.json", costs)
    write_json(args.output / "COMPLETE.json", {"completed": True, "source_hashes_unchanged": True,
                                               "total_candidates": len(rows), "wall_seconds": costs["wall_seconds"]})
    print(f"COMPLETE: {args.output} ({costs['wall_seconds']:.1f}s)", flush=True)


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def mathematical_checks(models, q, h, masks, api, output):
    records = {}
    for name in ("Hardware_joint256_default", "Hardware_joint256_tuned"):
        payload = models[name]
        transform = api["transform"].from_state_dict(payload["transform"])
        grouped = h.astype(np.float64).reshape(-1, 16, 16)
        mean = grouped[masks["train"]].mean(axis=0)
        std = np.maximum(grouped[masks["train"]].std(axis=0), 1e-12)
        assert np.array_equal(mean, transform.channel_mean_)
        assert np.array_equal(std, transform.channel_std_)
        selected = np.flatnonzero(masks["train"])[:12]
        product = np.ones((len(selected), 256))
        clipped_low, clipped_high, count = 0, 0, 0
        squared_error, squared_target, factors_count = 0., 0., 0
        diagnostic_rows = []
        for split, mask in masks.items():
            low, high, err, target_sq, total = 0, 0, 0., 0., 0
            for d in range(16):
                z = np.column_stack((np.ones(mask.sum()),
                                     (grouped[mask, d]-mean[d])/std[d]))
                estimated = z @ transform.calibration_weights_[d]
                ideal = np.exp(-(q[mask, d, None]-transform.centers_[None, :, d])**2/
                               (2*transform.joint_sigma**2))
                low += int((estimated < transform.factor_lower).sum())
                high += int((estimated > transform.factor_upper).sum())
                err += float(np.square(estimated-ideal).sum())
                target_sq += float(np.square(ideal).sum())
                total += estimated.size
            calibrated = transform.transform(h[mask])
            ideal_joint = transform.ideal_pc_transform(q[mask])
            # Bounded-memory CKA: feature covariance, never N x N kernel storage.
            from rbf_hardware.modeling.joint_gaussian import linear_cka
            diagnostic_rows.append({"split": split, "factor_rmse_before_clip": np.sqrt(err/total),
                                    "factor_relative_rmse": np.sqrt(err/max(target_sq, 1e-30)),
                                    "low_clip_fraction": low/total, "high_clip_fraction": high/total,
                                    "joint_rmse": float(np.sqrt(np.square(calibrated.astype(float)-ideal_joint).mean())),
                                    "joint_relative_rmse": float(np.linalg.norm(calibrated-ideal_joint)/max(np.linalg.norm(ideal_joint), 1e-30)),
                                    "joint_linear_cka": linear_cka(calibrated, ideal_joint),
                                    "all_zero_joint_rows": int((calibrated == 0).all(axis=1).sum()),
                                    "constant_joint_channels": int((calibrated.std(axis=0) < 1e-12).sum())})
        for d in range(16):
            z = np.column_stack((np.ones(len(selected)), (grouped[selected, d]-mean[d])/std[d]))
            product *= np.clip(z @ transform.calibration_weights_[d],
                               transform.factor_lower, transform.factor_upper)
        actual = transform.transform(h[selected])
        product_error = float(np.abs(product.astype(np.float32)-actual).max())
        assert np.allclose(actual, product, atol=2e-7, rtol=2e-6)
        # Joint factors are dimension-labelled. Only coordinated axis relabelling
        # (hardware, centers, standardizers and calibration together) is invariant.
        permutation = np.random.default_rng(42).permutation(16)
        state = dict(payload["transform"])
        for key in ("channel_mean", "channel_std", "calibration_weights", "calibration_r2"):
            state[key] = np.asarray(state[key])[permutation]
        state["centers"] = np.asarray(state["centers"])[:, permutation]
        relabelled = api["transform"].from_state_dict(state)
        hp = grouped[selected][:, permutation].reshape(-1, 256)
        relabel_error = float(np.abs(relabelled.transform(hp)-actual).max())
        assert np.allclose(relabelled.transform(hp), actual, atol=2e-7, rtol=2e-6)
        permuted_input_h = h[selected].reshape(-1,16,16)[:, permutation].reshape(-1,256)
        input_only_error = float(np.abs(transform.transform(permuted_input_h)-actual).max())
        # Independent Ridge normal equations and exact reconstruction of Torch scores.
        x = transform.transform(h[masks["train"]]).astype(np.float64)
        z = np.column_stack((np.ones(len(x)), x))
        target = np.eye(6)[np.load(output/"input_and_bank_mean.npz")["labels"][masks["train"]]]
        regularizer = np.eye(257)*payload["config"]["head_alpha"]
        regularizer[0, 0] = 0
        weights = np.asarray(payload["head"]["combined_weights"])
        residual = (z.T@z+regularizer)@weights-z.T@target
        normal_eq_relative_residual = float(np.linalg.norm(residual)/max(np.linalg.norm(z.T@target),1e-30))
        assert normal_eq_relative_residual < 1e-8
        records[name] = {"independent_product_max_abs_error": product_error,
                         "coordinated_dimension_relabelling_max_error": relabel_error,
                         "input_only_permutation_max_feature_change": input_only_error,
                         "input_only_invariance_required": False,
                         "tolerance_atol": 2e-7, "tolerance_rtol": 2e-6,
                         "normal_equations_relative_residual": normal_eq_relative_residual,
                         "ridge_is_direct_solve_not_iterative": True,
                         "train_only_means_stds_exact": True,
                         "diagnostics": diagnostic_rows}
    result = {"passed": True, "model": "Original dimension-labelled full joint Gaussian, NOT scalar response sum.",
              "inference_transform_uses_only_hardware_response": True, "checks": records}
    write_json(output / "model_checks.json", result)
    return result


def run_stress_tests(models, bank, q, y, repeats, api, output):
    records, predictions = [], {}
    names = ["Hardware_joint256_default", "Hardware_joint256_default_head_tuned", "Hardware_joint256_tuned"]
    for condition, bounds in (("recorded_cycle_only", (0., 0.)),
                              ("recorded_cycle_plus_simulator_noise", (.01, .05))):
        for repeat in range(repeats):
            seed = 1042+repeat
            h = bank.simulate(q, sampling_mode="empirical", random_generator=np.random.default_rng(seed),
                              minimum_noise_rate=bounds[0], maximum_noise_rate=bounds[1])
            for name in names:
                features = features_from_payload(models[name], q, h, api)
                predicted = numpy_scores(features, models[name]["head"]).argmax(axis=1)
                records.append(dict(condition=condition, repeat=repeat, seed=seed, scheme=name,
                                    n=len(y), **score(y, predicted)))
                predictions[f"{condition}_{repeat}_{name}"] = predicted
    frame = pd.DataFrame(records)
    frame.to_csv(output / "bank_cycle_robustness.csv", index=False)
    frame.groupby(["condition", "scheme"])[["accuracy", "macro_f1"]].agg(["mean", "std", "min", "max"]).to_csv(
        output / "bank_cycle_robustness_summary.csv")
    np.savez_compressed(output / "bank_cycle_predictions.npz", labels=y, **predictions)
    return frame


def subject_bootstrap(predictions):
    subset = predictions[predictions.split == "test"]
    base = subset[subset.scheme == "Linear16_standardized"].set_index("sample_id")
    records = {}
    for name in ("Hardware_joint256_default", "Hardware_joint256_tuned"):
        other = subset[subset.scheme == name].set_index("sample_id").loc[base.index]
        delta = other.correct.astype(int).to_numpy()-base.correct.astype(int).to_numpy()
        subjects = base.subject.to_numpy()
        unique = np.unique(subjects)
        rng = np.random.default_rng(42)
        sums = np.array([delta[subjects == v].sum() for v in unique])
        counts = np.array([(subjects == v).sum() for v in unique])
        draws = rng.integers(0, len(unique), size=(2000, len(unique)))
        differences = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        records[name+"_minus_Linear16_standardized"] = {
            "observed_accuracy_difference": float(delta.mean()),
            "subject_cluster_bootstrap_percentile_95_interval": np.quantile(differences, [.025,.975]),
            "repeats": 2000, "subjects": len(unique), "seed": 42,
            "interpretation": "Exploratory uncertainty across 9 held-out people; not independent new data or a guarantee."}
    return records


def verify_saved(output, q, h, labels, splits, api):
    predicted = pd.read_csv(output / "sample_predictions.csv")
    results = pd.read_csv(output / "results_table.csv")
    checks = []
    for row in results.itertuples():
        payload = joblib.load(output / "models" / (row.scheme+".joblib"))
        features = features_from_payload(payload, q, h, api)
        prediction = numpy_scores(features, payload["head"]).argmax(axis=1)
        saved = predicted[predicted.scheme == row.scheme]
        # Stored rows are grouped by split, so join explicitly rather than assuming
        # original archive order equals output CSV order.
        archive = np.load(output / "input_and_bank_mean.npz")
        by_id = saved.set_index("sample_id")
        assert [CLASSES[i] for i in prediction] == by_id.loc[archive["sample_ids"], "predicted"].tolist()
        for split in ("train", "validation", "test"):
            actual = score(labels[splits == split], prediction[splits == split])
            assert abs(actual["accuracy"]-getattr(row,split+"_accuracy")) < 1e-12
            assert abs(actual["macro_f1"]-getattr(row,split+"_macro_f1")) < 1e-12
        checks.append({"scheme": row.scheme, "predictions_reproduced": True, "all_metrics_reproduced": True})
    write_json(output / "postrun_verification.json", {"passed": True, "checks": checks})
    print("Reloaded saved models: predictions and metrics reproduced exactly.", flush=True)


def make_figures(output, table, search, matrices, models, q, h, masks, robustness, api):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220})
    colors = ["#576d8b", "#d3883c", "#258a88"]
    names = ["Linear16_standardized", "Hardware_joint256_default", "Hardware_joint256_tuned"]
    labels = ["Linear16 (standardized)", "Joint Gaussian default", "Joint Gaussian tuned"]
    figure, ax = plt.subplots(figsize=(11, 4.5))
    chosen = table.set_index("scheme").loc[names]
    positions = np.arange(3)
    for i, split in enumerate(("train", "validation", "test")):
        bars = ax.bar(positions+(i-1)*.24, chosen[split+"_accuracy"]*100,
                      width=.24, color=colors[i], label=split.capitalize())
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    ax.set(ylim=(0,100), ylabel="Accuracy (%)", xticks=positions, xticklabels=labels,
           title="KTH official subject split | original measured-bank joint RBF pipeline")
    ax.legend(ncol=3)
    figure.tight_layout(); figure.savefig(output/"figures/performance.png"); plt.close(figure)
    view = table[["scheme", "output_dimension", "train_accuracy", "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1"]].copy()
    for column in view.columns[2:]:
        view[column] = view[column].map(lambda x:f"{x:.4f}")
    figure, ax = plt.subplots(figsize=(17, 4.6)); ax.axis("off")
    grid = ax.table(cellText=view.values, colLabels=["Scheme", "Dim", "Train Acc", "Val Acc", "Val Macro-F1", "Test Acc", "Test Macro-F1"],
                    colWidths=[.38,.05,.1,.1,.13,.1,.13], cellLoc="center", loc="center")
    grid.auto_set_font_size(False); grid.set_fontsize(9); grid.scale(1,1.65)
    for c in range(7):
        grid[0,c].set_facecolor("#d8ebe9")
    ax.set_title("Software screening: all heads are one-hot Ridge; parameters selected on validation only", pad=12)
    figure.tight_layout(); figure.savefig(output/"results_table.png", bbox_inches="tight"); plt.close(figure)
    for name, matrix in matrices.items():
        figure, ax = plt.subplots(figsize=(8,7))
        confusion_panel(ax, matrix, name)
        figure.tight_layout(); figure.savefig(output/("figures/confusion_"+name+".png")); plt.close(figure)
    hardware = search[search.family == "hardware"]
    best = hardware.sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False).drop_duplicates(["sigma", "calibration_alpha", "factor_upper"])
    figure, axes = plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    low, high = hardware.validation_macro_f1.min(),hardware.validation_macro_f1.max()
    for ax, upper in zip(axes, FACTOR_UPPERS):
        grid = best[best.factor_upper==upper].pivot(index="calibration_alpha",columns="sigma",values="validation_macro_f1").reindex(index=CAL_ALPHAS,columns=SIGMAS)
        im = ax.imshow(grid,aspect="auto",cmap="viridis",vmin=low,vmax=high)
        ax.set(xticks=np.arange(len(SIGMAS)),xticklabels=SIGMAS,yticks=np.arange(3),yticklabels=CAL_ALPHAS,
               xlabel="Target joint sigma (NOT physical curve width)",ylabel="Calibration alpha",title=f"Upper factor clip = {upper}")
        for i in range(3):
            for j in range(len(SIGMAS)):
                ax.text(j,i,f"{grid.iloc[i,j]:.2f}",ha="center",va="center",color="white",fontsize=8)
    figure.colorbar(im,ax=axes.ravel().tolist(),label="Validation Macro-F1",shrink=.8)
    figure.savefig(output/"figures/validation_heatmap.png",bbox_inches="tight");plt.close(figure)
    sample = np.flatnonzero(masks["train"])[0]
    payload = models["Hardware_joint256_tuned"]
    phi = features_from_payload(payload,q[sample:sample+1],h[sample:sample+1],api)[0]
    figure, axes=plt.subplots(2,2,figsize=(13,9))
    axes[0,0].plot(np.arange(1,17),q[sample],"o-",color=colors[0])
    axes[0,0].set(title="16 dimension-labelled video inputs",xlabel="Temporal interval (1-16)",ylabel="Quantized normalized frame change")
    axes[0,1].plot(np.arange(1,257),phi,color=colors[2])
    axes[0,1].set(title="256 reconstructed joint Gaussian features",xlabel="Full-vector center index (1-256)",ylabel="Product across 16 calibrated factors")
    for i,split in enumerate(("validation","test")):
        bars=axes[1,0].bar(positions+(i-.5)*.32,chosen[split+"_accuracy"]*100,width=.32,color=colors[i+1],label=split)
        axes[1,0].bar_label(bars,fmt="%.1f",fontsize=9,padding=2)
    axes[1,0].set(ylim=(0,100),xticks=positions,xticklabels=["Linear16", "G default", "G tuned"],ylabel="Accuracy (%)",title="Official subject holdouts (no invented error bars)")
    axes[1,0].legend()
    confusion_panel(axes[1,1],matrices["Hardware_joint256_tuned"],"Validation-selected Gaussian | test")
    figure.tight_layout();figure.savefig(output/"four_panel_summary.png");plt.close(figure)
    figure,ax=plt.subplots(figsize=(11,5))
    for index,condition in enumerate(robustness.condition.unique()):
        group=robustness[robustness.condition==condition].groupby("scheme").accuracy.agg(["mean","std"])
        ys=[group.loc[n,"mean"]*100 for n in names[1:]]
        errors=[group.loc[n,"std"]*100 for n in names[1:]]
        ax.bar(np.arange(2)+(index-.5)*.32,ys,width=.32,yerr=errors,capsize=4,color=colors[index+1],label=condition)
    clean=table.set_index("scheme").loc[names[1:],"test_accuracy"]*100
    ax.scatter(np.arange(2),clean,color="black",marker="D",label="400-cycle mean response",zorder=4)
    ax.set(ylim=(0,100),xticks=np.arange(2),xticklabels=["G default", "G tuned"],ylabel="Test accuracy (%)",
           title="Frozen model sensitivity | error bars = bank-draw SD, not subject uncertainty")
    ax.legend(fontsize=9);figure.tight_layout();figure.savefig(output/"figures/bank_cycle_sensitivity.png");plt.close(figure)


def confusion_panel(ax, matrix, title):
    ax.imshow(matrix,cmap="Blues")
    ax.set(xticks=np.arange(6),xticklabels=CLASSES,yticks=np.arange(6),yticklabels=CLASSES,
           xlabel="Predicted class",ylabel="True class",title=title)
    ax.tick_params(axis="x",labelrotation=40)
    for i in range(6):
        for j in range(6):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",
                    color="white" if matrix[i,j]>matrix.max()/2 else "#243747",fontsize=9)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
