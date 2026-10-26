#!/usr/bin/env python3
"""G0/G1/G2: fixed-form KTH joint-RBF training with held-out bank indices.

No physical hardware access, video downloads or changed visual encoding.
The original transformer and Ridge head are imported read-only. Cycle holdout
is within the existing processed bank, not an independently acquired device.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import run_kth_joint_rbf as shared

SEED = 42
COPIES = 8
ANCHOR_N = 760
CAL_ALPHA_ANCHOR = .01
HEAD_ALPHA_ANCHOR = .1
SIGMA = .25
REPEATS = 10
SCHEMES = ["A_linear", "I_ideal", "G0_mean", "G1_head_cycle", "G2_calibration_and_head_cycle"]
G_NAMES = SCHEMES[2:]
UPSTREAM = Path("/home/ziyan/SURF2026/rbf_hardware")


def imports(root=UPSTREAM):
    sys.path.insert(0, str(root / "src"))
    from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer, benchmark_kmeans
    from rbf_hardware.modeling.ridge_classifier import RidgeLinearClassifier
    from rbf_hardware.inference.empirical_response import EmpiricalHardwareResponseBank
    from rbf_hardware.inference.preprocessing import DifferentialLevelQuantizer
    return {"transform": FullJointGaussianTransformer, "kmeans": benchmark_kmeans,
            "ridge": RidgeLinearClassifier, "bank": EmpiricalHardwareResponseBank,
            "quantizer": DifferentialLevelQuantizer}


def cycle_partition(seed=SEED):
    permutation = np.random.default_rng(seed).permutation(400)
    return {"train": np.sort(permutation[:240]), "validation": np.sort(permutation[240:320]),
            "test": np.sort(permutation[320:])}


def sum_alpha(anchor_alpha, n_rows):
    """Same average-loss lambda for all row counts, including duplicated rows."""
    return anchor_alpha * n_rows / ANCHOR_N


def prepare_inputs(raw_train, raw_other, quantizer):
    maximum = float(np.max(raw_train))
    denominator = maximum if maximum > 0 else 1.
    train = quantizer.transform(100*np.clip(raw_train/denominator, 0, 1))
    other = quantizer.transform(100*np.clip(raw_other/denominator, 0, 1))
    return train, other, {"fit_split": "train", "maximum": maximum, "denominator": denominator,
                          "other_clipped_value_fraction": float(np.mean((raw_other/denominator > 1) |
                                                                        (raw_other/denominator < 0)))}


def sampled_training_responses(bank, q, pool, copies=COPIES, seed=143):
    """One bank index shared by every dimension/channel of each augmented row."""
    if copies > len(pool):
        raise ValueError("Not enough distinct training cycle indices for the requested copies.")
    rng = np.random.default_rng(seed)
    cycles = np.stack([rng.choice(pool, size=copies, replace=False) for _ in q])
    indices = bank._resolve_level_indices(q)
    response = bank.response_samples[indices[:, None, :], cycles[:, :, None]]
    return response.reshape(len(q)*copies, 256).astype(np.float32), cycles


def mean_response(bank, q, pool):
    indices = bank._resolve_level_indices(q)
    means = bank.response_samples[:, pool].mean(axis=1, dtype=np.float64)
    return means[indices].astype(np.float32).reshape(len(q), 256)


def draw_response(bank, q, pool, seed, references, noise=False):
    # Restrict the original simulator to one partition; no custom drift model.
    subset = type(bank)(bank.levels, bank.response_samples[:, pool], references)
    response = subset.simulate(q, sampling_mode="empirical",
                               random_generator=np.random.default_rng(seed),
                               minimum_noise_rate=.01 if noise else 0.,
                               maximum_noise_rate=.05 if noise else 0.)
    local = np.random.default_rng(seed).integers(0, len(pool), size=(len(q), 1)).ravel()
    return response, pool[local]


def new_transform(api, n_rows):
    return api["transform"](dimensions=16, basis_per_dimension=16, output_features=256,
                            joint_sigma=SIGMA, calibration_alpha=sum_alpha(CAL_ALPHA_ANCHOR, n_rows),
                            factor_lower=1e-6, factor_upper=1., epsilon=1e-12,
                            random_state=SEED, kmeans_max_iter=300)


def fit_head(features, labels, anchor_alpha, api):
    return shared.fit_head(features, labels, sum_alpha(anchor_alpha, len(features)), api)


def model_features(payload, q, h, api):
    if payload["kind"] == "linear":
        return (q-payload["mean"])/payload["std"]
    transformer = api["transform"].from_state_dict(payload["transform"])
    return (transformer.ideal_pc_transform(q) if payload["kind"] == "ideal"
            else transformer.transform(h))


def predict(payload, features):
    scores = shared.numpy_scores(features, payload["head"])
    return scores, scores.argmax(axis=1)


def fit_three(bank, q, labels, pool, centers, api, seed=143, include_controls=True):
    begin = time.perf_counter()
    mean = mean_response(bank, q, pool)
    augmented, cycles = sampled_training_responses(bank, q, pool, seed=seed)
    q_aug = np.repeat(q, COPIES, axis=0)
    labels_aug = np.repeat(labels, COPIES)
    cal_begin = time.perf_counter()
    g0 = new_transform(api, len(q)).fit(mean, q, preset_centers=centers)
    g0_cal_seconds = time.perf_counter()-cal_begin
    cal_begin = time.perf_counter()
    g2 = new_transform(api, len(q_aug)).fit(augmented, q_aug, preset_centers=centers)
    g2_cal_seconds = time.perf_counter()-cal_begin
    phi_mean = g0.transform(mean)
    phi_aug0 = g0.transform(augmented)
    phi_aug2 = g2.transform(augmented)
    training_features = {"G0_mean": phi_mean, "G1_head_cycle": phi_aug0,
                         "G2_calibration_and_head_cycle": phi_aug2}
    models = {}
    timings = []
    for name, transformer, x, targets, cal_seconds in (
        ("G0_mean", g0, phi_mean, labels, g0_cal_seconds),
        ("G1_head_cycle", g0, phi_aug0, labels_aug, 0.),
        ("G2_calibration_and_head_cycle", g2, phi_aug2, labels_aug, g2_cal_seconds)):
        head_begin = time.perf_counter()
        head = fit_head(x, targets, HEAD_ALPHA_ANCHOR, api)
        seconds = time.perf_counter()-head_begin
        state = head.cpu_state_dict()
        prediction = shared.numpy_scores(x, state).argmax(axis=1)
        models[name] = {"kind": "hardware", "transform": transformer.state_dict(), "head": state,
                        "classes": shared.CLASSES, "fit_split": "train", "unique_train_n": len(q),
                        "head_training_rows": len(x), "calibration_training_rows": len(q_aug) if name == G_NAMES[2] else len(q),
                        "average_loss_calibration_lambda": CAL_ALPHA_ANCHOR/ANCHOR_N,
                        "average_loss_head_lambda": HEAD_ALPHA_ANCHOR/ANCHOR_N,
                        "sum_form_head_alpha": sum_alpha(HEAD_ALPHA_ANCHOR, len(x)),
                        "fit_training_row_metrics": shared.score(targets, prediction)}
        timings.append({"scheme": name, "head_fit_seconds": seconds, "calibration_fit_seconds": cal_seconds,
                        "calibration_reused_from": "G0_mean" if name == G_NAMES[1] else "",
                        "unique_train_n": len(q), "head_training_rows": len(x),
                        "sum_form_head_alpha": sum_alpha(HEAD_ALPHA_ANCHOR, len(x)),
                        "sum_form_calibration_alpha": transformer.calibration_alpha})
    if include_controls:
        mean_q = q.astype(float).mean(axis=0)
        std_q = np.maximum(q.astype(float).std(axis=0), 1e-12)
        x_linear = (q-mean_q)/std_q
        x_ideal = g0.ideal_pc_transform(q)
        for name, x, kind, anchor in (("A_linear", x_linear, "linear", 1.),
                                      ("I_ideal", x_ideal, "ideal", HEAD_ALPHA_ANCHOR)):
            head = fit_head(x, labels, anchor, api)
            models[name] = {"kind": kind, "head": head.cpu_state_dict(), "classes": shared.CLASSES,
                            "mean": mean_q, "std": std_q, "transform": g0.state_dict(), "fit_split": "train"}
    return models, {"mean": mean, "augmented": augmented, "cycles": cycles,
                    "features": training_features, "timings": timings,
                    "total_fit_seconds": time.perf_counter()-begin}


def fit_checks(models, train_q, training, labels, api):
    result = {"passed": True, "inference_uses_h_only": True, "checks": {}}
    for name in G_NAMES:
        model = models[name]
        transformer = api["transform"].from_state_dict(model["transform"])
        h = training["augmented"] if name == G_NAMES[2] else training["mean"]
        grouped = h.astype(float).reshape(-1, 16, 16)
        np.testing.assert_array_equal(grouped.mean(axis=0), transformer.channel_mean_)
        np.testing.assert_array_equal(np.maximum(grouped.std(axis=0), 1e-12), transformer.channel_std_)
        h_short = training["mean"][:12]
        grouped = h_short.astype(float).reshape(-1, 16, 16)
        independent = np.ones((len(h_short), 256))
        for d in range(16):
            z = np.column_stack((np.ones(len(h_short)),
                                 (grouped[:, d]-transformer.channel_mean_[d])/transformer.channel_std_[d]))
            independent *= np.clip(z@transformer.calibration_weights_[d], 1e-6, 1.)
        actual = transformer.transform(h_short)
        error = float(np.abs(independent.astype(np.float32)-actual).max())
        np.testing.assert_allclose(actual, independent, rtol=2e-6, atol=2e-7)
        x = training["features"][name].astype(float)
        target_labels = labels if name == G_NAMES[0] else np.repeat(labels, COPIES)
        z = np.column_stack((np.ones(len(x)), x))
        target = np.eye(6)[target_labels]
        penalty = np.eye(257)*model["sum_form_head_alpha"]
        penalty[0, 0] = 0
        weights = np.asarray(model["head"]["combined_weights"])
        residual = (z.T@z+penalty)@weights-z.T@target
        relative_residual = float(np.linalg.norm(residual)/max(np.linalg.norm(z.T@target), 1e-30))
        assert relative_residual < 1e-8
        assert actual.shape == (len(h_short), 256) and np.isfinite(actual).all()
        result["checks"][name] = {"train_only_channel_statistics_exact": True,
                                 "independent_product_max_error": error,
                                 "normal_equation_relative_residual": relative_residual,
                                 "atol": 2e-7, "rtol": 2e-6}
    for key in ("channel_mean", "channel_std", "calibration_weights", "centers"):
        np.testing.assert_array_equal(models[G_NAMES[0]]["transform"][key], models[G_NAMES[1]]["transform"][key])
    result["g1_calibration_exactly_reuses_g0"] = True
    result["g2_changes_training_standardization_and_calibration_together"] = True
    return result


def response_provenance(root, bank):
    metadata_path = root/"dataset/empirical/hardware_response_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["response_layout"] == "level,cycle,group"
    assert (metadata["level_count"], metadata["cycle_count"], metadata["group_count"]) == (256, 400, 16)
    assert bank.response_samples.shape == (256, 400, 16)
    assert np.isfinite(bank.response_samples).all()
    # The raw workbook is not distributed with the repository; do not invent it.
    candidates = [root/metadata["source_workbook"], root/"dataset/origin_xslx"/metadata["source_workbook"]]
    source = next((p for p in candidates if p.is_file()), None)
    return {"array_shape": list(bank.response_samples.shape), "metadata": metadata,
            "metadata_path": str(metadata_path),
            "migration_source": str(root/"src/rbf_hardware/utilities/empirical_response_migration.py"),
            "migration_contract": "Each of 256 sheets must have Cycle=1..400 and Group_1..16; mismatch raises an error.",
            "source_workbook_found_at_documented_locations": source is not None,
            "source_workbook_sha_verified": shared.sha256(source) == metadata["source_sha256"] if source else None,
            "independent_raw_workbook_or_chronology_verification": False,
            "scope": "Disjoint indices of an existing processed response library, under its existing alignment convention.",
            "not_claimed": ["new hardware acquisition", "independent device/batch generalization", "temporal drift extrapolation",
                            "end-to-end holdout of raw measurements before source-workbook normalization"],
            "negative_responses_preserved": bool((bank.response_samples < 0).any())}


def diagnostic(transformer, h, q):
    grouped = h.astype(float).reshape(-1, 16, 16)
    squared_error, total, low, high = 0., 0, 0, 0
    for d in range(16):
        z = np.column_stack((np.ones(len(q)), (grouped[:, d]-transformer.channel_mean_[d])/transformer.channel_std_[d]))
        factors = z@transformer.calibration_weights_[d]
        ideal = np.exp(-(q[:, d, None]-transformer.centers_[None, :, d])**2/(2*SIGMA**2))
        squared_error += float(np.square(factors-ideal).sum())
        total += factors.size
        low += int((factors < 1e-6).sum()); high += int((factors > 1).sum())
    return {"factor_rmse_before_clip": float(np.sqrt(squared_error/total)),
            "factor_low_clip_fraction": low/total, "factor_high_clip_fraction": high/total}


def evaluate(models, bank, q, y, manifest, pool, references, split, condition, repeat, seed, api):
    if condition == "pool_mean":
        h = mean_response(bank, q, pool)
        cycles = np.full(len(q), -1)
    else:
        h, cycles = draw_response(bank, q, pool, seed, references,
                                  noise=condition == "cycle_plus_simulator_noise")
    records, parts, features_by_calibration, arrays, diagnostics = [], [], {}, {}, []
    for name in SCHEMES:
        model = models[name]
        feature_begin = time.perf_counter()
        # G0 and G1 share EXACTLY the same response transformation.
        feature_key = "G0_mean" if name == "G1_head_cycle" else name
        if feature_key not in features_by_calibration:
            features_by_calibration[feature_key] = model_features(model, q, h, api)
        features = features_by_calibration[feature_key]
        transform_seconds = time.perf_counter()-feature_begin
        assert features.shape == (len(q), 16 if name == "A_linear" else 256)
        assert np.isfinite(features).all()
        predict_begin = time.perf_counter()
        scores, prediction = predict(model, features)
        predict_seconds = time.perf_counter()-predict_begin
        metrics = shared.score(y, prediction)
        records.append(dict(scheme=name, split=split, condition=condition, repeat=repeat,
                            seed=seed, unique_n=len(y), **metrics,
                            feature_seconds=transform_seconds, prediction_seconds=predict_seconds))
        part = manifest[["sample_id", "subject", "class", "split"]].copy()
        part["scheme"] = name; part["condition"] = condition; part["repeat"] = repeat
        part["seed"] = seed; part["bank_cycle_index_zero_based"] = cycles
        part["predicted"] = [shared.CLASSES[i] for i in prediction]
        part["correct"] = prediction == y
        for i, cls in enumerate(shared.CLASSES):
            part["score_"+cls] = scores[:, i]
        parts.append(part)
        arrays[name] = {"features": features, "scores": scores, "predictions": prediction}
        if name in G_NAMES:
            ideal = model_features(models["I_ideal"], q, h, api)
            relative = float(np.linalg.norm(features-ideal)/max(np.linalg.norm(ideal), 1e-30))
            entry = {"scheme": name, "split": split, "condition": condition, "repeat": repeat,
                     "relative_joint_feature_rmse": relative,
                     "all_zero_joint_rows": int((features == 0).all(axis=1).sum()),
                     "constant_joint_channels": int((features.std(axis=0) < 1e-12).sum())}
            if condition == "pool_mean" or repeat == 0:
                transformer = api["transform"].from_state_dict(model["transform"])
                entry.update(diagnostic(transformer, h, q))
            diagnostics.append(entry)
    return records, parts, arrays, diagnostics, cycles


def group_cv(raw, labels, manifest, bank, pools, quantizer, references, api, output):
    """Stability audit only; official validation remains the predeclared selector."""
    folds, results, timings, scaling_records = [], [], [], []
    splitter = GroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    for fold, (fit_index, validation_index) in enumerate(splitter.split(raw, labels, manifest.subject)):
        train_manifest, validation_manifest = manifest.iloc[fit_index], manifest.iloc[validation_index]
        assert not set(train_manifest.subject) & set(validation_manifest.subject)
        q_train, q_val, scaling = prepare_inputs(raw[fit_index], raw[validation_index], quantizer)
        centers = api["kmeans"](q_train, 256, SEED, 300)
        models, training = fit_three(bank, q_train, labels[fit_index], pools["train"], centers, api,
                                     seed=143+100*(fold+1), include_controls=True)
        for role, indices in (("train", fit_index), ("validation", validation_index)):
            part = manifest.iloc[indices][["sample_id", "subject", "class"]].copy()
            part["fold"] = fold; part["role"] = role; folds.append(part)
        fold_directory = output/"group_cv"/f"fold_{fold}"
        fold_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_directory/"inputs_and_assignments.npz", centers=centers,
                            fit_sample_ids=train_manifest.sample_id.to_numpy(dtype=str),
                            validation_sample_ids=validation_manifest.sample_id.to_numpy(dtype=str),
                            quantized_fit=q_train, quantized_validation=q_val, training_cycles=training["cycles"])
        for name, model in models.items():
            model["scaling"] = scaling
            joblib.dump(model, fold_directory/(name+".joblib"), compress=3)
        for repeat in range(5):
            seed = 4042+100*fold+repeat
            h, cycles = draw_response(bank, q_val, pools["validation"], seed, references)
            for name in SCHEMES:
                features = model_features(models[name], q_val, h, api)
                _, predicted = predict(models[name], features)
                results.append(dict(fold=fold, repeat=repeat, seed=seed, scheme=name,
                                    validation_n=len(validation_index), train_n=len(fit_index),
                                    **shared.score(labels[validation_index], predicted)))
        timings.extend([dict(fold=fold, **row) for row in training["timings"]])
        scaling_records.append(dict(fold=fold, train_subjects=sorted(set(train_manifest.subject)),
                                    validation_subjects=sorted(set(validation_manifest.subject)), **scaling))
        print(f"Grouped stability fold {fold+1}/4 completed (no official test subjects).", flush=True)
    pd.concat(folds, ignore_index=True).to_csv(output/"group_cv_folds.csv", index=False)
    frame = pd.DataFrame(results)
    frame.to_csv(output/"group_cv_results.csv", index=False)
    fold_means = frame.groupby(["scheme", "fold"])[["accuracy", "macro_f1"]].mean().reset_index()
    fold_means.to_csv(output/"group_cv_fold_means.csv", index=False)
    fold_means.groupby("scheme")[["accuracy", "macro_f1"]].agg(["mean", "std"]).to_csv(output/"group_cv_summary.csv")
    pd.DataFrame(timings).to_csv(output/"group_cv_training_times.csv", index=False)
    shared.write_json(output/"group_cv_scaling_and_subject_audit.json", scaling_records)


def verify_output(output, api):
    signature = json.loads((output/"signature.json").read_text())
    for path, expected in signature["source_hashes"].items():
        assert shared.sha256(path) == expected, f"Changed source: {path}"
    data = np.load(output/"inputs.npz", allow_pickle=False)
    q, y, splits = [data[k] for k in ("quantized", "labels", "splits")]
    assignments = pd.read_csv(output/"cycle_split.csv")
    assert assignments.bank_cycle_index_zero_based.nunique() == 400
    assert assignments.groupby("bank_cycle_index_zero_based").split.nunique().max() == 1
    pools = {name: assignments.loc[assignments.split == name, "bank_cycle_index_zero_based"].to_numpy()
             for name in ("train", "validation", "test")}
    root = Path(signature["hardware_root"])
    quantizer = api["quantizer"].load(root/"dataset/csv/differential_levels.csv", input_features=16,
                                     expected_levels=256, raw_minimum=0, raw_maximum=100)
    bank = api["bank"].load(root/"dataset/empirical/hardware_response_samples.npz", expected_groups=16,
                            expected_levels=quantizer.levels)
    references = np.load(output/"training_noise_magnitude_references.npy")
    expected_references = np.quantile(np.abs(bank.response_samples[:, pools["train"]]), .95, axis=(0,1))
    np.testing.assert_array_equal(references, expected_references)
    recorded = pd.read_csv(output/"sample_predictions.csv")
    reloaded = []
    # Reproduce every saved score/label for all conditions and repetitions, not
    # merely an accuracy that could match despite changed individual predictions.
    for name in SCHEMES:
        model = joblib.load(output/"models"/(name+".joblib"))
        for (split, condition, repeat), frame in recorded[recorded.scheme == name].groupby(["split", "condition", "repeat"]):
            mask = splits == split
            index = pd.Index(data["sample_ids"][mask])
            frame = frame.set_index("sample_id").loc[index]
            if condition == "pool_mean":
                h = mean_response(bank, q[mask], pools[split])
            else:
                h, cycles = draw_response(bank, q[mask], pools[split], int(frame.seed.iloc[0]), references,
                                          noise=condition == "cycle_plus_simulator_noise")
                np.testing.assert_array_equal(cycles, frame.bank_cycle_index_zero_based.to_numpy())
                assert set(cycles) <= set(pools[split])
            features = model_features(model, q[mask], h, api)
            scores, prediction = predict(model, features)
            np.testing.assert_allclose(scores, frame[["score_"+c for c in shared.CLASSES]].to_numpy(), rtol=1e-11, atol=1e-11)
            assert [shared.CLASSES[i] for i in prediction] == frame.predicted.tolist()
        if name in G_NAMES:
            assert model["transform"]["factor_lower"] == 1e-6
            assert model["transform"]["factor_upper"] == 1.
        reloaded.append(name)
    train_cycles = np.load(output/"training_responses.npz")["cycle_indices"]
    assert set(train_cycles.ravel()) <= set(pools["train"])
    assert all(len(set(row)) == COPIES for row in train_cycles)
    assert json.loads((output/"implementation_checks.json").read_text())["passed"]
    shared.write_json(output/"postrun_verification.json", {
        "passed": True, "saved_scores_and_labels_reproduced": reloaded,
        "cycle_pools_disjoint": True, "training_cycle_rows_unique_within_clip": True,
        "noise_magnitude_reference_train_cycles_only": True,
        "score_tolerance_atol": 1e-11, "score_tolerance_rtol": 1e-11,
        "source_hashes_unchanged": True})
    print("Saved models and all draw-level predictions verified; source hashes unchanged.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/kth_joint_rbf256_strict_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/kth_cycle_robustness_v1"))
    parser.add_argument("--hardware-root", type=Path, default=UPSTREAM)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.source, args.output, args.hardware_root = [p.resolve() for p in (args.source, args.output, args.hardware_root)]
    api = imports(args.hardware_root)
    if args.verify_only:
        verify_output(args.output, api)
        return
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    source_paths = [Path(__file__).resolve(), Path(shared.__file__).resolve(),
                    args.source/"input_and_bank_mean.npz", args.source/"sample_manifest.csv", args.source/"centers.npy",
                    args.source/"models/Linear16_standardized.joblib",
                    args.source/"models/Ideal_RBF256_matched_to_hardware.joblib",
                    args.hardware_root/"dataset/empirical/hardware_response_samples.npz",
                    args.hardware_root/"dataset/empirical/hardware_response_metadata.json",
                    args.hardware_root/"dataset/csv/differential_levels.csv"]
    source_paths += [args.hardware_root/"src/rbf_hardware"/s for s in
                     shared.UPSTREAM_FILES+["utilities/empirical_response_migration.py"]]
    signature = {"source_hashes": {str(p): shared.sha256(p) for p in source_paths},
                 "hardware_root": str(args.hardware_root), "seed": SEED, "copies": COPIES, "repeats": REPEATS,
                 "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scikit-learn", "pandas")}}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"signature.json").exists():
        if json.loads((args.output/"signature.json").read_text()) != signature:
            raise ValueError("Output signature differs. Preserve it and use a new --output directory.")
    else:
        shared.write_json(args.output/"signature.json", signature)
    if (args.output/"COMPLETE.json").exists():
        print("Identical completed experiment reused; no training repeated.")
        print(pd.read_csv(args.output/"results_table.csv").to_string(index=False))
        return
    for directory in ("models", "features", "source_code"):
        (args.output/directory).mkdir(exist_ok=True)
    data = np.load(args.source/"input_and_bank_mean.npz", allow_pickle=False)
    raw, labels, splits, ids = [data[k] for k in ("raw", "labels", "splits", "sample_ids")]
    manifest = pd.read_csv(args.source/"sample_manifest.csv")
    assert raw.shape == (2389, 16) and np.isfinite(raw).all() and (raw >= 0).all()
    assert manifest.sample_id.tolist() == ids.tolist() and len(set(ids)) == len(ids)
    assert [shared.CLASSES[i] for i in labels] == manifest["class"].tolist()
    assert manifest.split.tolist() == splits.tolist()
    assert manifest.groupby("subject").split.nunique().max() == 1
    assert manifest.groupby("video_id").split.nunique().max() == 1
    assert manifest.groupby("video_sha256").split.nunique().max() == 1
    masks = {name: splits == name for name in ("train", "validation", "test")}
    assert [int(mask.sum()) for mask in masks.values()] == [760, 767, 862]
    quantizer = api["quantizer"].load(args.hardware_root/"dataset/csv/differential_levels.csv", input_features=16,
                                     expected_levels=256, raw_minimum=0, raw_maximum=100)
    bank = api["bank"].load(args.hardware_root/"dataset/empirical/hardware_response_samples.npz", expected_groups=16,
                            expected_levels=quantizer.levels)
    pools = cycle_partition()
    references = np.quantile(np.abs(bank.response_samples[:, pools["train"]]), .95, axis=(0, 1))
    assert np.isfinite(references).all() and (references > 0).all()
    np.save(args.output/"training_noise_magnitude_references.npy", references)
    pool_rows = [{"bank_cycle_index_zero_based": int(i), "source_cycle_label_one_based": int(i)+1, "split": name}
                 for name, pool in pools.items() for i in pool]
    pd.DataFrame(pool_rows).sort_values("bank_cycle_index_zero_based").to_csv(args.output/"cycle_split.csv", index=False)
    shared.write_json(args.output/"response_bank_provenance.json", response_provenance(args.hardware_root, bank))
    shared.write_json(args.output/"preregistered_protocol.json", {
        "written_before_fitting_and_test_evaluation": True, "seed": SEED, "training_copies_per_clip": COPIES,
        "cycle_pool_counts": {k: len(v) for k, v in pools.items()}, "hyperparameter_search": False,
        "sigma": SIGMA, "factor_clip": [1e-6, 1.], "centers": "Same 256 original training-only KMeans centers",
        "anchor_unique_n": ANCHOR_N, "calibration_anchor_sum_alpha": CAL_ALPHA_ANCHOR,
        "head_anchor_sum_alpha": HEAD_ALPHA_ANCHOR,
        "average_loss_calibration_lambda": CAL_ALPHA_ANCHOR/ANCHOR_N,
        "average_loss_head_lambda": HEAD_ALPHA_ANCHOR/ANCHOR_N,
        "objective": "mean squared error per response row + lambda * squared non-intercept weights",
        "sum_alpha_rule": "anchor_alpha * response_row_count / 760; eight copies -> eight times sum alpha",
        "g1_control": "G0 standardization and calibration frozen; only head sees augmented responses",
        "g2_scope": "Train-response standardization AND calibration AND head refitted on 8 cycles; not a calibration-weights-only intervention",
        "training_cycles_seed": 143, "validation_draw_seeds": list(range(2042, 2052)),
        "test_draw_seeds": list(range(3042, 3052)),
        "main_condition": "Recorded cycle only; one shared library cycle index per clip across all dimensions/channels",
        "secondary_condition": "Original simulator 1-5% multiplicative noise; magnitude references fit train cycles only",
        "mean_reference": "Separate bank-pool mean at each split (240/80/80 reads), not single-read performance",
        "selection": "Among G0/G1/G2: validation cycle Macro-F1 mean desc, accuracy mean desc, Macro-F1 std asc, G0/G1/G2 order",
        "group_cv": "4 folds within original training 8 people; fold-only scaling/centers/stats/fit; five validation-cycle draws; stability audit not selector",
        "test_status": "Exploratory: KTH test split and full processed bank were viewed previously; not independent new-device validation",
        "test_mean_and_noise_do_not_select_models": True})
    _, q_all, scaling = prepare_inputs(raw[masks["train"]], raw, quantizer)
    np.testing.assert_array_equal(q_all, data["quantized"])
    centers = np.load(args.source/"centers.npy")
    assert centers.shape == (256, 16)
    manifest.to_csv(args.output/"sample_manifest.csv", index=False)
    manifest.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(columns=shared.CLASSES).to_csv(args.output/"class_counts.csv")
    shared.write_json(args.output/"scaling.json", dict(scaling, clipped_value_fraction_by_split={
        name: float(np.mean((raw[mask]/scaling["denominator"] > 1) | (raw[mask]/scaling["denominator"] < 0)))
        for name, mask in masks.items()}))
    np.savez_compressed(args.output/"inputs.npz", raw=raw, quantized=q_all, labels=labels, splits=splits,
                        sample_ids=ids, centers=centers, classes=np.asarray(shared.CLASSES))
    for file in ("previous_exclusions.csv", "previous_official_missing_videos.csv", "previous_independent_video_audit.json",
                 "previous_data_and_cost_stats.json"):
        shutil.copy2(args.source/file, args.output/file)
    shutil.copy2(__file__, args.output/"source_code"/Path(__file__).name)
    shutil.copy2(shared.__file__, args.output/"source_code"/Path(shared.__file__).name)
    shutil.copy2("requirements_joint.txt", args.output/"source_code/requirements_joint.txt")
    print("Cycle index pools locked: 240 train / 80 validation / 80 test. Fitting G0/G1/G2 with equal average-loss penalties.", flush=True)
    models, training = fit_three(bank, q_all[masks["train"]], labels[masks["train"]], pools["train"], centers, api,
                                include_controls=False)
    np.savez_compressed(args.output/"training_responses.npz", mean=training["mean"], augmented=training["augmented"],
                        cycle_indices=training["cycles"], sample_ids=ids[masks["train"]])
    train_assignments = manifest.loc[masks["train"], ["sample_id", "subject", "class"]].loc[
        manifest.index[masks["train"]].repeat(COPIES)].reset_index(drop=True)
    train_assignments["copy"] = np.tile(np.arange(COPIES), ANCHOR_N)
    train_assignments["bank_cycle_index_zero_based"] = training["cycles"].ravel()
    train_assignments.to_csv(args.output/"training_cycle_assignments.csv", index=False)
    pd.DataFrame(training["timings"]).to_csv(args.output/"training_times_and_penalties.csv", index=False)
    checks = fit_checks(models, q_all[masks["train"]], training, labels[masks["train"]], api)
    checks.update({"cache_and_manifest_alignment": True, "subjects_videos_hashes_disjoint_between_splits": True,
                   "raw_and_quantized_inputs_exactly_reused": True, "bank_cycle_pools_disjoint": True})
    shared.write_json(args.output/"implementation_checks.json", checks)
    reuse = []
    for name, old_name, kind in (("A_linear", "Linear16_standardized", "linear"),
                                  ("I_ideal", "Ideal_RBF256_matched_to_hardware", "ideal")):
        previous = joblib.load(args.source/"models"/(old_name+".joblib"))
        if kind == "linear":
            np.testing.assert_array_equal(previous["mean"], q_all[masks["train"]].astype(float).mean(axis=0))
            np.testing.assert_array_equal(previous["std"], q_all[masks["train"]].astype(float).std(axis=0))
            assert previous["config"]["head_alpha"] == 1.
        else:
            np.testing.assert_array_equal(previous["transform"]["centers"], centers)
            assert previous["transform"]["joint_sigma"] == SIGMA
            assert previous["config"]["head_alpha"] == HEAD_ALPHA_ANCHOR
        models[name] = dict(previous, kind=kind)
        if kind == "ideal":
            # Ideal features depend on centers/sigma only. Drop the irrelevant old
            # 400-cycle calibration state so saved controls carry no such state.
            models[name]["transform"] = models["G0_mean"]["transform"]
        reuse.append({"scheme": name, "source": str(args.source/"models"/(old_name+".joblib")),
                      "identical_conditions_verified": True, "new_fit": False,
                      "reason": "Same 760 quantized training coordinates, labels, train-only preprocessing, centers/sigma if applicable, head alpha and numerical environment; no cycle dependence."})
    shared.write_json(args.output/"reused_control_models.json", reuse)
    for name, model in models.items():
        model["input_scaling"] = scaling
        joblib.dump(model, args.output/"models"/(name+".joblib"), compress=3)
    for name, features in training["features"].items():
        np.savez_compressed(args.output/"features"/(name+"_head_training.npz"), features=features,
                            labels=labels[masks["train"]] if name == G_NAMES[0] else np.repeat(labels[masks["train"]], COPIES))
    cv_begin = time.perf_counter()
    group_cv(raw[masks["train"]], labels[masks["train"]], manifest.loc[masks["train"]].reset_index(drop=True),
             bank, pools, quantizer, references, api, args.output)
    cv_seconds = time.perf_counter()-cv_begin
    all_metrics, all_predictions, all_diagnostics = [], [], []
    feature_draws = {name: [] for name in G_NAMES}
    assignments = {}
    def consume(split, condition, repeat, seed):
        mask = masks[split]
        records, parts, values, diagnostics, cycles = evaluate(
            models, bank, q_all[mask], labels[mask], manifest.loc[mask], pools[split], references,
            split, condition, repeat, seed, api)
        all_metrics.extend(records); all_predictions.extend(parts); all_diagnostics.extend(diagnostics)
        assignments[f"{split}_{condition}_{repeat}"] = cycles
        np.savez_compressed(args.output/"features"/f"{split}_{condition}_{repeat}.npz",
                            sample_ids=ids[mask], cycle_indices=cycles,
                            **{name: value["features"] for name, value in values.items()})
        if split == "test" and condition == "cycle":
            for name in G_NAMES: feature_draws[name].append(values[name]["features"])
        return values
    # Validation is completed and selector written before any test metric exists.
    consume("train", "pool_mean", -1, -1)
    consume("validation", "pool_mean", -1, -1)
    for repeat in range(REPEATS):
        consume("validation", "cycle", repeat, 2042+repeat)
    validation = pd.DataFrame(all_metrics)
    validation.to_csv(args.output/"validation_before_test.csv", index=False)
    stats = validation[(validation.split == "validation") & (validation.condition == "cycle")].groupby("scheme")[
        ["accuracy", "macro_f1"]].agg(["mean", "std"])
    locked_name = min(G_NAMES, key=lambda n: (-stats.loc[n, ("macro_f1", "mean")],
                                             -stats.loc[n, ("accuracy", "mean")],
                                             stats.loc[n, ("macro_f1", "std")], G_NAMES.index(n)))
    shared.write_json(args.output/"locked_selection_before_test.json", {
        "selected_scheme": locked_name, "selection_uses_validation_cycles_only": True,
        "gaussian_parameters_fixed": True, "validation": [dict(scheme=n,
            accuracy_mean=stats.loc[n, ("accuracy", "mean")], macro_f1_mean=stats.loc[n, ("macro_f1", "mean")],
            macro_f1_std=stats.loc[n, ("macro_f1", "std")]) for n in G_NAMES]})
    print(f"Validation selector locked: {locked_name}. Now evaluating test clips and test-cycle indices.", flush=True)
    test_mean = consume("test", "pool_mean", -1, -1)
    for repeat in range(REPEATS):
        consume("test", "cycle", repeat, 3042+repeat)
        consume("test", "cycle_plus_simulator_noise", repeat, 3042+repeat)
        if repeat in (4, 9):
            print(f"Held-out test-cycle draws completed: {repeat+1}/{REPEATS}; models remain frozen.", flush=True)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(args.output/"all_metrics.csv", index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(args.output/"sample_predictions.csv", index=False)
    pd.DataFrame(all_diagnostics).to_csv(args.output/"feature_diagnostics.csv", index=False)
    np.savez_compressed(args.output/"evaluation_cycle_assignments.npz", **assignments)
    parameters = []
    for name in G_NAMES:
        parameter = models[name]
        parameters.append({"scheme": name, "sigma": SIGMA, "factor_lower": 1e-6, "factor_upper": 1.,
                           "unique_training_clips": ANCHOR_N,
                           "calibration_response_rows": parameter["calibration_training_rows"],
                           "head_response_rows": parameter["head_training_rows"],
                           "calibration_sum_alpha": parameter["transform"]["calibration_alpha"],
                           "head_sum_alpha": parameter["sum_form_head_alpha"],
                           "calibration_average_lambda": CAL_ALPHA_ANCHOR/ANCHOR_N,
                           "head_average_lambda": HEAD_ALPHA_ANCHOR/ANCHOR_N})
    pd.DataFrame(parameters).to_csv(args.output/"model_parameters.csv", index=False)
    summary_rows = []
    for name in SCHEMES:
        row = {"scheme": name, "output_dimension": 16 if name == "A_linear" else 256,
               "train_unique_n": 760, "validation_unique_n": 767, "test_unique_n": 862,
               "selected_by_validation": name == locked_name}
        for split in masks:
            result = metrics[(metrics.scheme == name) & (metrics.split == split) & (metrics.condition == "pool_mean")].iloc[0]
            for key in ("accuracy", "macro_f1"):
                row[split+"_pool_mean_"+key] = float(result[key])
        for split, condition, prefix in (("validation", "cycle", "validation_cycle"),
                                          ("test", "cycle", "test_cycle"),
                                          ("test", "cycle_plus_simulator_noise", "test_noise")):
            subset = metrics[(metrics.scheme == name) & (metrics.split == split) & (metrics.condition == condition)]
            for key in ("accuracy", "macro_f1"):
                row[prefix+"_"+key+"_mean"] = float(subset[key].mean())
                row[prefix+"_"+key+"_std"] = float(subset[key].std(ddof=1))
        row["test_cycle_accuracy_drop_from_pool_mean"] = row["test_pool_mean_accuracy"]-row["test_cycle_accuracy_mean"]
        if name in G_NAMES:
            row.update({"fit_training_rows_accuracy": models[name]["fit_training_row_metrics"]["accuracy"],
                        "fit_training_rows_macro_f1": models[name]["fit_training_row_metrics"]["macro_f1"]})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output/"results_table.csv", index=False)
    per_subject = []
    for keys, group in predictions.groupby(["scheme", "split", "condition", "repeat", "subject"]):
        values = shared.score([shared.CLASSES.index(c) for c in group["class"]],
                              [shared.CLASSES.index(c) for c in group.predicted])
        per_subject.append(dict(zip(("scheme", "split", "condition", "repeat", "subject"), keys), n=len(group), **values))
    pd.DataFrame(per_subject).to_csv(args.output/"per_subject_metrics.csv", index=False)
    for name in SCHEMES:
        for condition in ("pool_mean", "cycle", "cycle_plus_simulator_noise"):
            subset = predictions[(predictions.scheme == name) & (predictions.split == "test") & (predictions.condition == condition)]
            for category, selected in (("all_draws", subset), ("first_draw", subset[subset.repeat == subset.repeat.min()])):
                yy = [shared.CLASSES.index(c) for c in selected["class"]]
                pp = [shared.CLASSES.index(c) for c in selected.predicted]
                matrix = confusion_matrix(yy, pp, labels=range(6))
                assert np.trace(matrix) == selected.correct.sum()
                pd.DataFrame(matrix, index=shared.CLASSES, columns=shared.CLASSES).to_csv(
                    args.output/f"confusion_{name}_{condition}_{category}.csv")
                shared.write_json(args.output/f"classes_{name}_{condition}_{category}.json", classification_report(
                    yy, pp, labels=range(6), target_names=shared.CLASSES, output_dict=True, zero_division=0))
    variability = []
    for name in G_NAMES:
        stacked = np.stack(feature_draws[name])
        for j in range(256):
            variability.append({"scheme": name, "joint_feature": j+1,
                                "mean_within_clip_cycle_std": float(stacked[:, :, j].std(axis=0, ddof=1).mean()),
                                "mean_abs_difference_from_pool_mean": float(np.abs(stacked[:, :, j]-test_mean[name]["features"][None, :, j]).mean())})
    pd.DataFrame(variability).to_csv(args.output/"joint_feature_cycle_variability.csv", index=False)
    # Verify all model files and predictions without fitting anything again.
    verify_output(args.output, api)
    costs = {"wall_seconds": time.perf_counter()-started, "main_fit_seconds": training["total_fit_seconds"],
             "group_cv_seconds": cv_seconds, "new_main_head_fits": 3, "new_main_calibration_fits": 2,
             "reused_main_control_heads": 2, "new_cv_head_fits": 20, "new_cv_calibration_fits": 8,
             "unique_training_clips": 760, "augmented_training_rows": 6080,
             "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             "cpu": shared.cpu_name(), "logical_cpus": psutil.cpu_count(), "torch_threads": torch.get_num_threads(),
             "system_ram_bytes": psutil.virtual_memory().total, "gpu_used": False,
             "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
             "raw_feature_bytes": raw.nbytes, "training_response_augmented_bytes": training["augmented"].nbytes,
             "one_test_joint_feature_matrix_bytes": 862*256*4, "bank_disk_bytes":
             (args.hardware_root/"dataset/empirical/hardware_response_samples.npz").stat().st_size,
             "new_video_download_seconds": 0., "new_video_preprocessing_seconds": 0.,
             "note": "Wall includes audits, CSV/model/feature saves and reload verification; plots generated separately. Storage is not peak memory."}
    shared.write_json(args.output/"cost_and_environment.json", costs)
    shared.write_json(args.output/"COMPLETE.json", {"completed": True, "selected_by_validation": locked_name,
                                                    "wall_seconds": costs["wall_seconds"], "old_results_preserved": True,
                                                    "source_hashes_unchanged": True})
    print(summary[["scheme", "validation_cycle_macro_f1_mean", "test_pool_mean_accuracy",
                   "test_cycle_accuracy_mean", "test_cycle_macro_f1_mean", "test_noise_accuracy_mean"]].to_string(index=False), flush=True)
    print(f"COMPLETE: {args.output}; {costs['wall_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
