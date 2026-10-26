#!/usr/bin/env python3
"""Finite, validation-only tuning of the fixed KTH region4/time4 device pipeline.

No video decoding, downloads, physical curve changes, new centers, or hardware IO.
Import the original joint transformer/Ridge read-only and reuse matching old fits.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
import resource
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

import run_kth_spatiotemporal as previous

cycle = previous.cycle
shared = cycle.shared
SIGMAS = [.125, .25, .5]
CAL_ALPHAS = [.001, .01, .1]
HEAD_ALPHAS = [.01, .1, 1., 10.]
SCHEMES = previous.SCHEMES
BASE = Path("outputs/kth_spatiotemporal_v1")
OUT = Path("outputs/kth_spatiotemporal_tuning_v1")


def config(scheme, head_alpha, sigma=None, calibration_alpha=None):
    result = dict(scheme=scheme, head_alpha=float(head_alpha), sigma=sigma,
                  calibration_alpha=calibration_alpha)
    result["id"] = shared.key_of(result)
    return result


def configurations():
    rows = [config("A_linear", a) for a in HEAD_ALPHAS]
    for sigma, cal, head, name in itertools.product(SIGMAS, CAL_ALPHAS, HEAD_ALPHAS,
                                                  ["G0_mean", "G1_head_cycle"]):
        rows.append(config(name, head, sigma, cal))
    rows += [config("I_ideal", head, sigma, .01) for sigma, head in
             itertools.product(SIGMAS, HEAD_ALPHAS)]
    return rows


def is_default(row):
    if row["scheme"] == "A_linear":
        return row["head_alpha"] == 1.
    return (row["head_alpha"] == .1 and row["sigma"] == .25 and
            row["calibration_alpha"] == .01)


def selection_key(row):
    """Only the named validation fields influence selection, never test fields."""
    return (-row["validation_macro_f1_mean"], -row["validation_accuracy_mean"],
            row["validation_macro_f1_std"], not is_default(row),
            SCHEMES.index(row["scheme"]),
            abs(np.log2(row["sigma"]/.25)) if row["sigma"] else 0.,
            abs(np.log10(row["calibration_alpha"]/.01)) if row["calibration_alpha"] else 0.,
            -row["head_alpha"], row["id"])


def select(rows):
    chosen = {name: min((r for r in rows if r["scheme"] == name), key=selection_key)
              for name in SCHEMES}
    return {"per_scheme": chosen,
            "gaussian": min((chosen[n] for n in ["G0_mean", "G1_head_cycle"]), key=selection_key),
            "selection_uses_validation_only": True,
            "ideal_is_diagnostic_not_hardware_candidate": True}


def transformer(cfg, mean_h, q, centers, api):
    return api["transform"](dimensions=16, basis_per_dimension=16, output_features=256,
        joint_sigma=cfg["sigma"], calibration_alpha=cycle.sum_alpha(cfg["calibration_alpha"], len(q)),
        factor_lower=1e-6, factor_upper=1., epsilon=1e-12,
        random_state=42, kmeans_max_iter=300).fit(mean_h, q, preset_centers=centers)


def payload(cfg, transform, q, features, y, api):
    head = cycle.fit_head(features, y, cfg["head_alpha"], api)
    return {"kind": "linear" if cfg["scheme"] == "A_linear" else
                    "ideal" if cfg["scheme"] == "I_ideal" else "hardware",
            "head": head.cpu_state_dict(), "transform": transform.state_dict(),
            "mean": q.astype(float).mean(axis=0), "std": np.maximum(q.astype(float).std(axis=0), 1e-12),
            "classes": shared.CLASSES, "config": cfg, "fit_split": "train",
            "unique_train_n": len(q), "head_training_rows": len(features),
            "sum_form_head_alpha": cycle.sum_alpha(cfg["head_alpha"], len(features)),
            "average_loss_head_lambda": cfg["head_alpha"]/760.}


def metrics_for(model, x, y):
    return shared.score(y, cycle.predict(model, x)[1])


def load_context(source, api):
    sig = json.loads((source/"signature.json").read_text())
    for path, digest in sig["source_hashes"].items():
        assert shared.sha256(path) == digest, f"Previous experiment changed: {path}"
    assert {k: importlib.metadata.version(k) for k in sig["versions"]} == sig["versions"]
    assert json.loads((source/"independent_protocol_verification.json").read_text())["passed"]
    arrays = dict(np.load(source/"new_inputs.npz", allow_pickle=False))
    manifest = pd.read_csv(source/"sample_manifest.csv")
    assert manifest.sample_id.tolist() == arrays["sample_ids"].tolist()
    assert manifest.split.tolist() == arrays["splits"].tolist()
    assert manifest["class"].tolist() == [shared.CLASSES[i] for i in arrays["labels"]]
    assert arrays["raw"].shape == (2389, 16) and np.isfinite(arrays["raw"]).all()
    assert not manifest.sample_id.duplicated().any()
    assert manifest.groupby("subject").split.nunique().max() == 1
    assert manifest.groupby("video_sha256").split.nunique().max() == 1
    masks = {name: arrays["splits"] == name for name in ["train", "validation", "test"]}
    assert [int(m.sum()) for m in masks.values()] == [760, 767, 862]
    quantizer = api["quantizer"].load(cycle.UPSTREAM/"dataset/csv/differential_levels.csv",
        input_features=16, expected_levels=256, raw_minimum=0, raw_maximum=100)
    bank = api["bank"].load(cycle.UPSTREAM/"dataset/empirical/hardware_response_samples.npz",
                           expected_groups=16, expected_levels=quantizer.levels)
    _, q, scaling = cycle.prepare_inputs(arrays["raw"][masks["train"]], arrays["raw"], quantizer)
    np.testing.assert_array_equal(q, arrays["quantized"])
    training = dict(np.load(source/"new_training_responses.npz", allow_pickle=False))
    np.testing.assert_array_equal(training["sample_ids"], arrays["sample_ids"][masks["train"]])
    pools = cycle.cycle_partition()
    saved = pd.read_csv(source/"cycle_split.csv")
    for name in pools:
        np.testing.assert_array_equal(pools[name], saved.loc[saved.split == name, "bank_cycle_index_zero_based"])
    np.testing.assert_array_equal(training["mean"], cycle.mean_response(bank, q[masks["train"]], pools["train"]))
    indices = bank._resolve_level_indices(q[masks["train"]])
    np.testing.assert_array_equal(training["augmented"],
        bank.response_samples[indices[:, None, :], training["cycle_indices"][:, :, None]].reshape(-1, 256))
    assert set(training["cycle_indices"].ravel()) <= set(pools["train"])
    original_source = Path(sig["old_results"])
    expected = dict(np.load(original_source/"evaluation_cycle_assignments.npz", allow_pickle=False))
    refs = np.load(source/"training_noise_magnitude_references.npy")
    np.testing.assert_array_equal(refs, np.quantile(np.abs(bank.response_samples[:, pools["train"]]), .95, axis=(0, 1)))
    models = {n: joblib.load(source/"models"/previous.NEW/(n+".joblib")) for n in SCHEMES}
    np.testing.assert_array_equal(models["G0_mean"]["transform"]["centers"], arrays["centers"])
    return dict(arrays=arrays, manifest=manifest, masks=masks, training=training, pools=pools,
                quantizer=quantizer, bank=bank, refs=refs, models=models, expected=expected, scaling=scaling)


def response(context, split, condition, repeat):
    mask = context["masks"][split]; q = context["arrays"]["quantized"][mask]
    seed = -1 if condition == "pool_mean" else (2042 if split == "validation" else 3042)+repeat
    if condition == "pool_mean":
        h = cycle.mean_response(context["bank"], q, context["pools"][split]); indices = np.full(len(q), -1)
    else:
        h, indices = cycle.draw_response(context["bank"], q, context["pools"][split], seed,
            context["refs"], noise=condition == "cycle_plus_simulator_noise")
    np.testing.assert_array_equal(indices, context["expected"][f"{split}_{condition}_{repeat}"])
    return q, h, indices, seed


def search(source, root, context, api):
    arrays = context["arrays"]; train = context["masks"]["train"]; val = context["masks"]["validation"]
    qt, qv = arrays["quantized"][train], arrays["quantized"][val]
    yt, yv = arrays["labels"][train], arrays["labels"][val]
    mean_h, aug_h = context["training"]["mean"], context["training"]["augmented"]
    val_h = [response(context, "validation", "cycle", i)[1] for i in range(10)]
    old_metrics = pd.read_csv(source/"all_metrics.csv")
    old_metrics = old_metrics[old_metrics.representation == previous.NEW]
    rows, details, timings = [], [], []
    counters = {"calibrations_fitted": 0, "calibrations_reused": 0, "heads_fitted": 0, "heads_reused": 0}
    groups = [(None, None)] + list(itertools.product(SIGMAS, CAL_ALPHAS))
    configs = configurations()
    for sigma, cal in groups:
        group = [c for c in configs if c["sigma"] == sigma and c["calibration_alpha"] == cal]
        start = time.perf_counter()
        if sigma is None or (sigma == .25 and cal == .01):
            transform = api["transform"].from_state_dict(context["models"]["G0_mean"]["transform"])
            if sigma is not None: counters["calibrations_reused"] += 1
        else:
            transform = transformer(group[0], mean_h, qt, arrays["centers"], api)
            counters["calibrations_fitted"] += 1
        calibration_seconds = time.perf_counter()-start
        if sigma is None:
            mean_features = (qt-context["models"]["A_linear"]["mean"])/context["models"]["A_linear"]["std"]
            train_features = {"A_linear": mean_features}
            validation_features = {"A_linear": [(qv-context["models"]["A_linear"]["mean"])/context["models"]["A_linear"]["std"]]*10}
        else:
            if sigma == .25 and cal == .01:
                mean_features = np.load(source/"features/G0_mean_head_training.npz")["features"]
                aug_features = np.load(source/"features/G1_head_cycle_head_training.npz")["features"]
                validation_phi = [np.load(source/"features"/f"validation_cycle_{i}.npz")["G0_mean"] for i in range(10)]
            else:
                mean_features = transform.transform(mean_h); aug_features = transform.transform(aug_h)
                validation_phi = [transform.transform(h) for h in val_h]
            train_features = {"G0_mean": mean_features, "G1_head_cycle": aug_features}
            validation_features = {"G0_mean": validation_phi, "G1_head_cycle": validation_phi}
            if cal == .01:
                train_features["I_ideal"] = transform.ideal_pc_transform(qt)
                validation_features["I_ideal"] = [transform.ideal_pc_transform(qv)]*10
        for cfg in group:
            start = time.perf_counter(); name = cfg["scheme"]; x = train_features[name]
            target = np.repeat(yt, 8) if name == "G1_head_cycle" else yt
            if is_default(cfg):
                model = dict(context["models"][name]); model["config"] = cfg
                counters["heads_reused"] += 1
            else:
                model = payload(cfg, transform, qt, x, target, api); counters["heads_fitted"] += 1
            head_seconds = time.perf_counter()-start
            model["input_scaling"] = context["models"]["A_linear"]["input_scaling"]
            model["representation"] = previous.NEW
            joblib.dump(model, root/"candidate_models"/(cfg["id"]+".joblib"), compress=3)
            if name == "I_ideal": train_mean_x = x
            elif name == "A_linear": train_mean_x = x
            else: train_mean_x = mean_features
            train_scores = metrics_for(model, train_mean_x, yt)
            actual_scores = metrics_for(model, x, target)
            fold_scores = []
            for repeat, feature in enumerate(validation_features[name]):
                if is_default(cfg):
                    old = old_metrics[(old_metrics.scheme == name)&(old_metrics.split == "validation")&
                        (old_metrics.condition == "cycle")&(old_metrics.repeat == repeat)].iloc[0]
                    score = {key: float(old[key]) for key in ["accuracy", "macro_f1"]}
                    # Independently check one saved default prediction without fitting it again.
                    if repeat == 0:
                        actual = metrics_for(model, feature, yv)
                        assert all(np.isclose(score[k], actual[k], atol=1e-14, rtol=0) for k in score)
                else: score = metrics_for(model, feature, yv)
                detail = dict(**cfg, repeat=repeat, seed=2042+repeat, unique_n=len(yv), **score)
                details.append(detail); fold_scores.append(score)
            means = {key: float(np.mean([r[key] for r in fold_scores])) for key in ["accuracy", "macro_f1"]}
            stds = {key: float(np.std([r[key] for r in fold_scores], ddof=1)) for key in ["accuracy", "macro_f1"]}
            rows.append(dict(**cfg, output_dimension=16 if name == "A_linear" else 256,
                default=is_default(cfg), fit_reused=is_default(cfg),
                train_mean_accuracy=train_scores["accuracy"], train_mean_macro_f1=train_scores["macro_f1"],
                fit_row_accuracy=actual_scores["accuracy"], fit_row_macro_f1=actual_scores["macro_f1"],
                validation_accuracy_mean=means["accuracy"], validation_accuracy_std=stds["accuracy"],
                validation_macro_f1_mean=means["macro_f1"], validation_macro_f1_std=stds["macro_f1"]))
            timings.append(dict(id=cfg["id"], scheme=name, head_fit_or_load_seconds=head_seconds,
                                head_fit_reused=is_default(cfg), head_rows=len(x)))
        print(f"Validation completed: sigma={sigma}, calibration_alpha={cal}; candidates={len(rows)}/88", flush=True)
        timings.append(dict(id=f"transform_{sigma}_{cal}", calibration_fit_or_load_seconds=calibration_seconds))
    assert len(rows) == 88 and len(details) == 880
    pd.DataFrame(rows).to_csv(root/"validation_candidates.csv", index=False)
    pd.DataFrame(details).to_csv(root/"validation_draw_metrics.csv", index=False)
    pd.DataFrame(timings).to_csv(root/"search_timings.csv", index=False)
    shared.write_json(root/"reuse_audit.json", counters)
    return rows


def evaluate_locked(source, root, context, selected, api):
    """Called only after the validation lock has been written to disk."""
    lock = json.loads((root/"locked_selection_before_test.json").read_text())
    assert lock == selected
    ids = {r["id"] for r in selected["per_scheme"].values()}
    defaults = [c for c in configurations() if is_default(c)]
    configs = {c["id"]: c for c in defaults}
    configs.update({r["id"]: {k: r[k] for k in ["id", "scheme", "head_alpha", "sigma", "calibration_alpha"]}
                    for r in selected["per_scheme"].values()})
    old_metrics = pd.read_csv(source/"all_metrics.csv"); old_pred = pd.read_csv(source/"sample_predictions.csv")
    old_metrics = old_metrics[old_metrics.representation == previous.NEW]
    old_pred = old_pred[old_pred.representation == previous.NEW]
    metrics, predictions, diagnostics = [], [], []
    for identifier, cfg in configs.items():
        model = joblib.load(root/"candidate_models"/(identifier+".joblib"))
        joblib.dump(model, root/"models"/(identifier+".joblib"), compress=3)
        if is_default(cfg):
            m = old_metrics[old_metrics.scheme == cfg["scheme"]].copy(); m["id"] = identifier
            p = old_pred[old_pred.scheme == cfg["scheme"]].copy(); p["id"] = identifier
            metrics.append(m); predictions.append(p)
            continue
        for split, condition, repeats in [("train", "pool_mean", [-1]), ("validation", "pool_mean", [-1]),
                ("validation", "cycle", range(10)), ("test", "pool_mean", [-1]),
                ("test", "cycle", range(10)), ("test", "cycle_plus_simulator_noise", range(10))]:
            mask = context["masks"][split]; y = context["arrays"]["labels"][mask]
            for repeat in repeats:
                q, h, indices, seed = response(context, split, condition, repeat)
                begin = time.perf_counter(); feature = cycle.model_features(model, q, h, api)
                feature_seconds = time.perf_counter()-begin
                assert feature.shape == (len(y), 16 if cfg["scheme"] == "A_linear" else 256)
                assert np.isfinite(feature).all()
                begin = time.perf_counter(); scores, pred = cycle.predict(model, feature)
                prediction_seconds = time.perf_counter()-begin
                metrics.append(pd.DataFrame([dict(id=identifier, scheme=cfg["scheme"], split=split,
                    condition=condition, repeat=repeat, seed=seed, unique_n=len(y), **shared.score(y, pred),
                    feature_seconds=feature_seconds, prediction_seconds=prediction_seconds)]))
                part = context["manifest"].loc[mask, ["sample_id", "subject", "scenario", "class", "split"]].copy()
                part["id"] = identifier; part["scheme"] = cfg["scheme"]; part["condition"] = condition
                part["repeat"] = repeat; part["seed"] = seed; part["bank_cycle_index_zero_based"] = indices
                part["predicted"] = [shared.CLASSES[i] for i in pred]; part["correct"] = pred == y
                for j, cls in enumerate(shared.CLASSES): part["score_"+cls] = scores[:, j]
                predictions.append(part)
                np.savez_compressed(root/"features"/f"{identifier}_{split}_{condition}_{repeat}.npz",
                    sample_ids=context["arrays"]["sample_ids"][mask], features=feature)
                if model["kind"] == "hardware":
                    transform = api["transform"].from_state_dict(model["transform"])
                    ideal = transform.ideal_pc_transform(q)
                    diagnostics.append(dict(id=identifier, split=split, condition=condition, repeat=repeat,
                        relative_joint_feature_rmse=float(np.linalg.norm(feature-ideal)/max(np.linalg.norm(ideal), 1e-30))))
    all_metrics = pd.concat(metrics, ignore_index=True); all_pred = pd.concat(predictions, ignore_index=True)
    all_metrics.to_csv(root/"all_metrics.csv", index=False); all_pred.to_csv(root/"sample_predictions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(root/"joint_feature_diagnostics.csv", index=False)
    table = []
    for name in SCHEMES:
        for stage, cfg in [("default", next(c for c in defaults if c["scheme"] == name)),
                           ("adjusted", selected["per_scheme"][name])]:
            subset = all_metrics[all_metrics.id == cfg["id"]]
            row = dict(stage=stage, scheme=name, id=cfg["id"], output_dimension=16 if name == "A_linear" else 256,
                       sigma=cfg["sigma"], calibration_alpha=cfg["calibration_alpha"], head_alpha=cfg["head_alpha"],
                       unique_train_n=760, unique_validation_n=767, unique_test_n=862)
            for split in ["train", "validation", "test"]:
                mean = subset[(subset.split == split)&(subset.condition == "pool_mean")].iloc[0]
                for metric in ["accuracy", "macro_f1"]: row[split+"_mean_"+metric] = float(mean[metric])
            for split, cond, prefix in [("validation", "cycle", "validation"), ("test", "cycle", "test"),
                                        ("test", "cycle_plus_simulator_noise", "test_noise")]:
                part = subset[(subset.split == split)&(subset.condition == cond)]
                assert len(part) == 10
                for metric in ["accuracy", "macro_f1"]:
                    row[prefix+"_"+metric+"_mean"] = float(part[metric].mean())
                    row[prefix+"_"+metric+"_std"] = float(part[metric].std(ddof=1))
            table.append(row)
    table = pd.DataFrame(table); table.to_csv(root/"results_table.csv", index=False)
    # Fixed first draw, not the best draw; aggregate counts are separately labelled.
    for identifier, part in all_pred[(all_pred.split == "test")&(all_pred.condition == "cycle")].groupby("id"):
        for label, frame in [("first_draw", part[part.repeat == 0]), ("all_draws", part)]:
            matrix = confusion_matrix(frame["class"], frame.predicted, labels=shared.CLASSES)
            pd.DataFrame(matrix, index=shared.CLASSES, columns=shared.CLASSES).to_csv(root/f"confusion_{identifier}_{label}.csv")
    return table


def cross_validate(source, root, context, selected, api):
    """Post-selection subject audit, NOT an additional search or test selector."""
    results, audits, fit_times = [], [], []
    lookup = pd.Index(context["arrays"]["sample_ids"])
    old_cv = pd.read_csv(source/"group_cv_results.csv")
    old_cv = old_cv[old_cv.representation == previous.NEW].copy(); old_cv["stage"] = "default"
    for fold in range(4):
        base = source/"group_cv"/previous.NEW/f"fold_{fold}"
        saved = np.load(base/"inputs_and_assignments.npz", allow_pickle=False)
        fit = lookup.get_indexer(saved["fit_sample_ids"]); val = lookup.get_indexer(saved["validation_sample_ids"])
        assert (fit >= 0).all() and (val >= 0).all()
        assert set(context["arrays"]["splits"][fit]) == {"train"}
        assert set(context["arrays"]["splits"][val]) == {"train"}
        assert not set(context["manifest"].iloc[fit].subject)&set(context["manifest"].iloc[val].subject)
        qt, qv, scaling = cycle.prepare_inputs(context["arrays"]["raw"][fit], context["arrays"]["raw"][val], context["quantizer"])
        np.testing.assert_array_equal(qt, saved["quantized_fit"]); np.testing.assert_array_equal(qv, saved["quantized_validation"])
        centers = saved["centers"]; yt = context["arrays"]["labels"][fit]; yv = context["arrays"]["labels"][val]
        mean_h = cycle.mean_response(context["bank"], qt, context["pools"]["train"])
        indices = context["bank"]._resolve_level_indices(qt)
        aug_h = context["bank"].response_samples[indices[:, None, :], saved["training_cycles"][:, :, None]].reshape(-1, 256)
        destination = root/"group_cv"/f"fold_{fold}"; destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/"inputs_and_assignments.npz", destination/"inputs_and_assignments.npz")
        transforms = {}; models = {}
        for name, chosen in selected["per_scheme"].items():
            cfg = {k: chosen[k] for k in ["id", "scheme", "head_alpha", "sigma", "calibration_alpha"]}
            begin = time.perf_counter()
            if is_default(cfg): model = dict(joblib.load(base/(name+".joblib"))); model["config"] = cfg
            else:
                key = (cfg["sigma"], cfg["calibration_alpha"])
                if key not in transforms:
                    if cfg["sigma"] is None or key == (.25, .01):
                        transform = api["transform"].from_state_dict(joblib.load(base/"G0_mean.joblib")["transform"])
                    else: transform = transformer(cfg, mean_h, qt, centers, api)
                    transforms[key] = transform
                transform = transforms[key]
                if name == "A_linear": x = (qt-qt.astype(float).mean(axis=0))/np.maximum(qt.astype(float).std(axis=0), 1e-12)
                elif name == "I_ideal": x = transform.ideal_pc_transform(qt)
                else: x = transform.transform(aug_h if name == "G1_head_cycle" else mean_h)
                target = np.repeat(yt, 8) if name == "G1_head_cycle" else yt
                model = payload(cfg, transform, qt, x, target, api)
            model["input_scaling"] = scaling; models[name] = model
            joblib.dump(model, destination/(name+".joblib"), compress=3)
            fit_times.append(dict(fold=fold, scheme=name, fit_reused=is_default(cfg), seconds=time.perf_counter()-begin))
        for repeat in range(5):
            seed = 4042+100*fold+repeat
            h, bank_cycles = cycle.draw_response(context["bank"], qv, context["pools"]["validation"], seed, context["refs"])
            for name, model in models.items():
                pred = cycle.predict(model, cycle.model_features(model, qv, h, api))[1]
                results.append(dict(stage="adjusted", fold=fold, repeat=repeat, seed=seed, scheme=name,
                                    train_n=len(fit), validation_n=len(val), **shared.score(yv, pred)))
        audits.append(dict(fold=fold, fit_sample_n=len(fit), validation_sample_n=len(val), **scaling,
                           participants_disjoint=True, train_only_centers_and_statistics=True))
        print(f"Locked-config participant audit: fold {fold+1}/4", flush=True)
    combined = pd.concat([old_cv, pd.DataFrame(results)], ignore_index=True)
    combined.to_csv(root/"group_cv_results.csv", index=False)
    combined.groupby(["stage", "scheme", "fold"])[["accuracy", "macro_f1"]].mean().reset_index().to_csv(root/"group_cv_fold_means.csv", index=False)
    pd.DataFrame(fit_times).to_csv(root/"group_cv_fit_times.csv", index=False)
    shared.write_json(root/"group_cv_checks.json", audits)


def verify(root, source, context, api):
    """Replay final predictions and audit train-only fits without re-training."""
    checks = []
    targets = context["arrays"]["labels"][context["masks"]["train"]]
    def audit_model(model, q, y, mean_h, aug_h, expected_centers):
        cfg = model["config"]; name = cfg["scheme"]
        if name == "A_linear":
            np.testing.assert_array_equal(model["mean"], q.astype(float).mean(axis=0))
            np.testing.assert_array_equal(model["std"], np.maximum(q.astype(float).std(axis=0), 1e-12))
        else:
            transform = api["transform"].from_state_dict(model["transform"])
            np.testing.assert_array_equal(transform.centers_, expected_centers)
            assert transform.joint_sigma == cfg["sigma"] and transform.factor_upper == 1. and transform.factor_lower == 1e-6
            if name != "I_ideal":
                grouped = mean_h.astype(float).reshape(-1, 16, 16)
                np.testing.assert_array_equal(transform.channel_mean_, grouped.mean(axis=0))
                np.testing.assert_array_equal(transform.channel_std_, np.maximum(grouped.std(axis=0), 1e-12))
                assert np.isclose(transform.calibration_alpha, cycle.sum_alpha(cfg["calibration_alpha"], len(q)))
                independent = np.ones((12, 256))
                for d in range(16):
                    z = np.column_stack((np.ones(len(q)), (grouped[:, d]-transform.channel_mean_[d])/transform.channel_std_[d]))
                    target = np.exp(-(q.astype(float)[:, d, None]-expected_centers[None, :, d])**2/(2*cfg["sigma"]**2))
                    reg = np.eye(17)*transform.calibration_alpha; reg[0, 0] = 0
                    residual = (z.T@z+reg)@transform.calibration_weights_[d]-z.T@target
                    assert np.linalg.norm(residual)/max(np.linalg.norm(z.T@target), 1e-30) < 1e-8
                    independent *= np.clip(z[:12]@transform.calibration_weights_[d], 1e-6, 1.)
                np.testing.assert_allclose(transform.transform(mean_h[:12]), independent, atol=2e-7, rtol=2e-6)
        if name == "G1_head_cycle": x = cycle.model_features(model, np.repeat(q, 8, axis=0), aug_h, api); y = np.repeat(y, 8)
        else: x = cycle.model_features(model, q, mean_h, api)
        z = np.column_stack((np.ones(len(x)), x.astype(float))); target = np.eye(6)[y]
        alpha = cycle.sum_alpha(cfg["head_alpha"], len(x)); reg = np.eye(z.shape[1])*alpha; reg[0, 0] = 0
        residual = (z.T@z+reg)@np.asarray(model["head"]["combined_weights"])-z.T@target
        relative = float(np.linalg.norm(residual)/max(np.linalg.norm(z.T@target), 1e-30))
        assert relative < 1e-8
        return relative
    predictions = pd.read_csv(root/"sample_predictions.csv")
    all_metrics = pd.read_csv(root/"all_metrics.csv")
    for identifier, frame in predictions.groupby("id"):
        model = joblib.load(root/"models"/(identifier+".joblib"))
        residual = audit_model(model, context["arrays"]["quantized"][context["masks"]["train"]], targets,
            context["training"]["mean"], context["training"]["augmented"], context["arrays"]["centers"])
        max_error = 0.
        for (split, condition, repeat), part in frame.groupby(["split", "condition", "repeat"]):
            mask = context["masks"][split]; q, h, indices, _ = response(context, split, condition, int(repeat))
            part = part.set_index("sample_id").loc[context["arrays"]["sample_ids"][mask]]
            np.testing.assert_array_equal(indices, part.bank_cycle_index_zero_based.to_numpy())
            feature = cycle.model_features(model, q, h, api); scores, pred = cycle.predict(model, feature)
            saved_scores = part[["score_"+c for c in shared.CLASSES]].to_numpy()
            np.testing.assert_allclose(scores, saved_scores, rtol=1e-11, atol=1e-11)
            max_error = max(max_error, float(np.max(np.abs(scores-saved_scores))))
            assert [shared.CLASSES[i] for i in pred] == part.predicted.tolist()
            actual = shared.score(context["arrays"]["labels"][mask], pred)
            old = all_metrics[(all_metrics.id == identifier)&(all_metrics.split == split)&
                (all_metrics.condition == condition)&(all_metrics.repeat == repeat)].iloc[0]
            for key in actual: assert np.isclose(actual[key], old[key], atol=1e-14)
        checks.append(dict(id=identifier, normal_equation_relative_residual=residual,
                           maximum_saved_score_error=max_error, all_predictions_reproduced=True))
    lookup = pd.Index(context["arrays"]["sample_ids"])
    for fold in range(4):
        directory = root/"group_cv"/f"fold_{fold}"; saved = np.load(directory/"inputs_and_assignments.npz")
        fit = lookup.get_indexer(saved["fit_sample_ids"])
        q = saved["quantized_fit"]; y = context["arrays"]["labels"][fit]
        mean = cycle.mean_response(context["bank"], q, context["pools"]["train"])
        indices = context["bank"]._resolve_level_indices(q)
        augmented = context["bank"].response_samples[indices[:, None, :], saved["training_cycles"][:, :, None]].reshape(-1, 256)
        for name in SCHEMES:
            audit_model(joblib.load(directory/(name+".joblib")), q, y, mean, augmented, saved["centers"])
    # Independently reconstruct selection from saved per-draw validation scores.
    validation = pd.read_csv(root/"validation_draw_metrics.csv")
    reconstructed = []
    for cfg in configurations():
        part = validation[validation.id == cfg["id"]]
        reconstructed.append(dict(**cfg, validation_macro_f1_mean=float(part.macro_f1.mean()),
            validation_accuracy_mean=float(part.accuracy.mean()), validation_macro_f1_std=float(part.macro_f1.std(ddof=1))))
    lock = json.loads((root/"locked_selection_before_test.json").read_text())
    repeated = select(reconstructed)
    assert {n: r["id"] for n, r in repeated["per_scheme"].items()} == {n: r["id"] for n, r in lock["per_scheme"].items()}
    assert repeated["gaussian"]["id"] == lock["gaussian"]["id"]
    shared.write_json(root/"verification.json", dict(passed=True, main_models=checks,
        four_fold_train_only_normal_equations_verified=True, original_centers_fixed=True,
        no_test_scores_in_selection=True, selected_ids_reconstructed_from_validation_only=True,
        repeated_cycle_pairing_exact=True, score_atol=1e-11, score_rtol=1e-11))
    print("PASS: all final scores, train-only fits, four-fold equations and validation-only selection verified.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(); source = args.source.resolve(); root = args.output.resolve()
    started = time.perf_counter(); torch.set_num_threads(1); torch.manual_seed(42)
    api = cycle.imports(); context = load_context(source, api)
    previous_sig = json.loads((source/"signature.json").read_text())
    paths = [Path(__file__).resolve(), source/"signature.json", source/"new_inputs.npz", source/"sample_manifest.csv",
        source/"new_training_responses.npz", source/"all_metrics.csv", source/"sample_predictions.csv",
        source/"group_cv_results.csv", source/"independent_protocol_verification.json"]
    paths += list((source/"models"/previous.NEW).glob("*.joblib"))
    paths += list((source/"group_cv"/previous.NEW).rglob("*.joblib"))
    paths += list((source/"group_cv"/previous.NEW).rglob("inputs_and_assignments.npz"))
    paths += list((source/"features").glob("validation_cycle_*.npz"))
    paths += [source/"features/G0_mean_head_training.npz", source/"features/G1_head_cycle_head_training.npz"]
    paths += [Path(path) for path in previous_sig["source_hashes"]]
    signature = dict(source=str(source), hashes={str(p): shared.sha256(p) for p in set(paths)},
        versions=previous_sig["versions"], seed=42, sigmas=SIGMAS, calibration_alphas=CAL_ALPHAS,
        head_alphas=HEAD_ALPHAS, configurations=configurations())
    root.mkdir(parents=True, exist_ok=True)
    if (root/"signature.json").exists():
        assert json.loads((root/"signature.json").read_text()) == signature, "Changed conditions: preserve old output and use a new --output."
    else: shared.write_json(root/"signature.json", signature)
    if args.verify_only: verify(root, source, context, api); return
    if (root/"COMPLETE.json").exists():
        print("Matching completed tuning reused; no search, fits or test evaluation repeated."); return
    for name in ["candidate_models", "models", "features", "source_code"]: (root/name).mkdir(exist_ok=True)
    protocol = dict(fixed_input="region4_time4; same 17 cached frames; no extraction", seed=42,
        train_validation_test_unique_n=[760, 767, 862], centers="same 256 train-only KMeans centers, no center search",
        primary_candidates=76, ideal_diagnostic_candidates=12, sigmas=SIGMAS,
        calibration_anchor_alphas=CAL_ALPHAS, head_anchor_alphas=HEAD_ALPHAS,
        anchor_n=760, sum_form_alpha="anchor_alpha * number_of_rows / 760; intercept unpenalized",
        gaussian_clip=[1e-6, 1.], physical_bank_unchanged=True, pulse_or_circuit_simulation=False,
        formula="H[N,16,16] -> train channel standardization -> per-dimension 17x256 Ridge -> clip factors -> exp(sum(log(factors))) [N,256] -> Ridge scores",
        schemes="G0: training-pool mean calibration and head; G1: same calibration, head trained on same eight cycles/video",
        selection="10 official validation Cycle draws: Macro-F1 mean desc, accuracy mean desc, F1 SD asc, default preferred, scheme order, width/calibration distance to default, larger head alpha, ID",
        test_lock_required=True, test_status="Exploratory: old official test set already examined; no new independent confirmation",
        group_cv="Same 4 participant folds, fitted only after configuration lock, audit only; no additional search",
        limitation="Existing processed response bank index holdout; source workbook unavailable, not independent new-device or raw-measurement validation")
    shared.write_json(root/"preregistered_protocol.json", protocol)
    for filename in ["sample_manifest.csv", "class_counts.csv", "new_inputs.npz", "new_input_scaling.json",
                     "cycle_split.csv", "response_bank_provenance.json", "training_noise_magnitude_references.npy"]:
        shutil.copy2(source/filename, root/filename)
    begin = time.perf_counter(); candidates = search(source, root, context, api)
    search_seconds = time.perf_counter()-begin
    locked = select(candidates); shared.write_json(root/"locked_selection_before_test.json", locked)
    print("LOCKED before new test scoring:", json.dumps({n: {k: r[k] for k in ["sigma", "calibration_alpha", "head_alpha"]}
          for n, r in locked["per_scheme"].items()}), flush=True)
    begin = time.perf_counter(); cross_validate(source, root, context, locked, api)
    cv_seconds = time.perf_counter()-begin
    begin = time.perf_counter(); table = evaluate_locked(source, root, context, locked, api)
    evaluation_seconds = time.perf_counter()-begin
    verify(root, source, context, api)
    costs = dict(wall_seconds=time.perf_counter()-started, search_seconds=search_seconds,
        locked_config_group_cv_seconds=cv_seconds, locked_config_evaluation_seconds=evaluation_seconds,
        cpu=platform.processor() or cycle.shared.platform.processor(), logical_cpus=psutil.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False), total_ram_bytes=psutil.virtual_memory().total,
        gpu_used=False, torch_threads=torch.get_num_threads(), blas_threads=1,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        video_decode_seconds=0., download_seconds=0., no_hardware_energy_or_latency_measurement=True)
    costs["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                               if line.startswith("model name")), "unavailable")
    shared.write_json(root/"cost_and_environment.json", costs)
    for path in [Path(__file__), Path(previous.__file__), Path(cycle.__file__), Path(shared.__file__), Path("requirements_joint.txt")]:
        shutil.copy2(path, root/"source_code"/path.name)
    for path, digest in signature["hashes"].items(): assert shared.sha256(path) == digest
    shared.write_json(root/"COMPLETE.json", dict(completed=True, old_results_preserved=True,
        selection_locked_before_new_test_evaluation=True, selected_gaussian_id=locked["gaussian"]["id"],
        source_hashes_unchanged=True))
    print(table[["stage", "scheme", "sigma", "calibration_alpha", "head_alpha", "validation_macro_f1_mean", "test_accuracy_mean", "test_macro_f1_mean"]].to_string(index=False))
    print(f"COMPLETE: {root}; {costs['wall_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1): main()
