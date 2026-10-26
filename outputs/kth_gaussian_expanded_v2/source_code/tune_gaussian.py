#!/usr/bin/env python3
"""Source-grouped, training-only Gaussian parameter comparison on real HMDB51."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import resource
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_experiment import CLASSES, SEED, gaussian_sum, plot_confusion

SIGMAS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
CENTER_MODES = ["uniform_full", "power2", "power3", "uniform_p95", "uniform_p99", "quantiles"]
CENTER_LABELS = {
    "uniform_full": "Uniform [0, 1]",
    "power2": "Power 2 (denser near 0)",
    "power3": "Power 3 (denser near 0)",
    "uniform_p95": "Uniform [0, training P95]",
    "uniform_p99": "Uniform [0, training P99]",
    "quantiles": "Training empirical quantiles",
}
CLASSIFIER_SETTINGS = {"C": 1.0, "solver": "lbfgs", "max_iter": 5000, "tol": 1e-6,
                       "random_state": SEED}


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_group(cls: str, filename: str) -> str:
    """Conservative source-title heuristic, shared across action classes."""
    source, marker, suffix = Path(filename).stem.rpartition(f"_{cls}_")
    if not marker or not re.match(r"[a-z]+_(cm|nm)_", suffix):
        raise ValueError(f"Cannot infer source group: {filename}")
    group = re.sub(r"[^a-z0-9]+", "", source.casefold())
    if not group:
        raise ValueError(f"Empty source group: {filename}")
    return group


def scale_fit_and_eval(raw_fit, raw_eval):
    maximum = float(np.max(raw_fit))
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("Invalid training maximum")
    # All-zero training features retain a defined, train-only scale.
    divisor = maximum if maximum > 0 else 1.0
    fit = raw_fit / divisor
    before_clip = raw_eval / divisor
    mask = (before_clip < 0) | (before_clip > 1)
    evaluation = np.clip(before_clip, 0, 1)
    return fit, evaluation, {"training_global_max": maximum, "divisor": divisor,
        "zero_max_fallback": maximum == 0, "eval_clipped_values": int(mask.sum()),
        "eval_total_values": int(mask.size), "eval_clipped_ratio": float(mask.mean())}


def make_centers(mode, fit_scaled):
    t = np.linspace(0.0, 1.0, 16)
    if mode == "uniform_full":
        centers = t
    elif mode in ("power2", "power3"):
        centers = t ** int(mode[-1])
    elif mode in ("uniform_p95", "uniform_p99"):
        q = 0.95 if mode.endswith("95") else 0.99
        centers = t * np.quantile(fit_scaled, q)
    elif mode == "quantiles":
        centers = np.quantile(fit_scaled, t)
    else:
        raise ValueError(mode)
    assert centers.shape == (16,) and np.isfinite(centers).all()
    assert (np.diff(centers) >= 0).all() and centers.min() >= 0 and centers.max() <= 1
    return centers


def make_model():
    return Pipeline([("standard_scaler", StandardScaler()),
                     ("logistic_regression", LogisticRegression(**CLASSIFIER_SETTINGS))])


def fit_evaluate(fit, evaluation, y_fit, y_eval):
    model = make_model()
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(fit, y_fit)
    train_seconds = time.perf_counter() - start
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("A classifier did not converge; refusing to rank incomplete fits")
    start = time.perf_counter()
    prediction = model.predict(evaluation)
    predict_seconds = time.perf_counter() - start
    stats = {"accuracy": float(accuracy_score(y_eval, prediction)),
        "macro_f1": float(f1_score(y_eval, prediction, average="macro", labels=CLASSES, zero_division=0)),
        "classifier_training_seconds": train_seconds, "prediction_seconds": predict_seconds,
        "n_iter": int(model.named_steps["logistic_regression"].n_iter_.max()),
        "train_samples": len(fit), "validation_samples": len(evaluation)}
    assert model.named_steps["standard_scaler"].n_samples_seen_ == len(fit)
    return model, prediction, stats


def config_id(mode, sigma):
    return f"{mode}__sigma_{sigma:g}"


def rank_results(frame):
    return frame.sort_values(["mean_validation_accuracy", "mean_validation_macro_f1",
                              "std_validation_accuracy", "config_id"],
                             ascending=[False, False, True, True]).reset_index(drop=True)


def run_validation(training, out, folds, seed):
    raw = np.asarray(training.feature.tolist(), dtype=np.float64)
    y = training["class"].to_numpy()
    groups = training.source_group.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    results, predictions, audit, per_class_counts, centers_records = [], [], [], [], []
    fold_manifest = training[["sample_id", "class", "filename", "source_group"]].copy()
    fold_manifest["validation_fold"] = -1
    centers_path = out / "fold_features"
    centers_path.mkdir(exist_ok=True)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(raw, y, groups), 1):
        fit_groups, val_groups = set(groups[fit_idx]), set(groups[val_idx])
        assert not fit_groups.intersection(val_groups)
        assert set(y[fit_idx]) == set(CLASSES) and set(y[val_idx]) == set(CLASSES)
        fold_manifest.loc[val_idx, "validation_fold"] = fold
        fit, val, scaling = scale_fit_and_eval(raw[fit_idx], raw[val_idx])
        audit.append({"fold": fold, "training_groups": len(fit_groups), "validation_groups": len(val_groups),
                      "group_overlap": 0, "training_samples": len(fit_idx), "validation_samples": len(val_idx),
                      **scaling})
        for cls in CLASSES:
            per_class_counts.append({"fold": fold, "class": cls,
                "train": int(np.sum(y[fit_idx] == cls)), "validation": int(np.sum(y[val_idx] == cls))})
        np.savez_compressed(centers_path / f"fold_{fold}_inputs.npz", raw_train=raw[fit_idx],
            raw_validation=raw[val_idx], scaled_train=fit, scaled_validation=val,
            training_sample_ids=training.iloc[fit_idx].sample_id.to_numpy(),
            validation_sample_ids=training.iloc[val_idx].sample_id.to_numpy(),
            training_labels=y[fit_idx], validation_labels=y[val_idx])
        baseline_model, baseline_prediction, stats = fit_evaluate(fit, val, y[fit_idx], y[val_idx])
        results.append({"fold": fold, "config_id": "A_linear", "center_mode": "none", "sigma": None,
                        "feature_computation_seconds": 0.0, **stats})
        for idx, pred in zip(val_idx, baseline_prediction):
            predictions.append({"fold": fold, "config_id": "A_linear", "sample_id": int(training.iloc[idx].sample_id),
                                "true_class": y[idx], "predicted_class": pred})
        outputs = {}
        for mode in CENTER_MODES:
            centers = make_centers(mode, fit)
            centers_records.append({"fold": fold, "center_mode": mode, "centers": centers.tolist()})
            for sigma in SIGMAS:
                cid = config_id(mode, sigma)
                start = time.perf_counter()
                mapped_fit, mapped_val = gaussian_sum(fit, sigma, centers), gaussian_sum(val, sigma, centers)
                transform_seconds = time.perf_counter() - start
                assert mapped_fit.shape == fit.shape and mapped_val.shape == val.shape
                outputs[f"{cid}_train"] = mapped_fit
                outputs[f"{cid}_validation"] = mapped_val
                model, prediction, stats = fit_evaluate(mapped_fit, mapped_val, y[fit_idx], y[val_idx])
                results.append({"fold": fold, "config_id": cid, "center_mode": mode, "sigma": sigma,
                                "feature_computation_seconds": transform_seconds, **stats})
                for idx, pred in zip(val_idx, prediction):
                    predictions.append({"fold": fold, "config_id": cid,
                        "sample_id": int(training.iloc[idx].sample_id), "true_class": y[idx], "predicted_class": pred})
        np.savez_compressed(centers_path / f"fold_{fold}_gaussian_outputs.npz", **outputs)
        print(f"Validation fold {fold}/{folds}: train={len(fit_idx)}, val={len(val_idx)}, source overlap=0", flush=True)
    assert (fold_manifest.validation_fold > 0).all()
    fold_manifest.to_csv(out / "training_validation_folds.csv", index=False)
    pd.DataFrame(per_class_counts).to_csv(out / "fold_class_counts.csv", index=False)
    write_json(out / "fold_scaling_and_group_audit.json", audit)
    write_json(out / "fold_centers.json", centers_records)
    detail = pd.DataFrame(results)
    detail.to_csv(out / "validation_fold_metrics.csv", index=False)
    oof = pd.DataFrame(predictions)
    oof.to_csv(out / "validation_predictions.csv", index=False)
    summary_rows = []
    for cid, values in detail.groupby("config_id", sort=False):
        pred = oof[oof.config_id == cid]
        assert len(pred) == len(training) and pred.sample_id.nunique() == len(training)
        summary_rows.append({"config_id": cid, "center_mode": values.iloc[0].center_mode,
            "sigma": None if cid == "A_linear" else float(values.iloc[0].sigma),
            "mean_validation_accuracy": float(values.accuracy.mean()),
            "std_validation_accuracy": float(values.accuracy.std(ddof=1)),
            "mean_validation_macro_f1": float(values.macro_f1.mean()),
            "std_validation_macro_f1": float(values.macro_f1.std(ddof=1)),
            "pooled_validation_accuracy": float(accuracy_score(pred.true_class, pred.predicted_class)),
            "pooled_validation_macro_f1": float(f1_score(pred.true_class, pred.predicted_class, average="macro", labels=CLASSES, zero_division=0)),
            "classifier_training_seconds_sum": float(values.classifier_training_seconds.sum()),
            "feature_computation_seconds_sum": float(values.feature_computation_seconds.sum()),
            "maximum_classifier_iterations": int(values.n_iter.max())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "validation_summary.csv", index=False)
    ranked = rank_results(summary[summary.center_mode != "none"])
    ranked.insert(0, "validation_rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(out / "gaussian_validation_ranking.csv", index=False)
    return summary, ranked


def plot_sensitivity(summary, out):
    gaussian = summary[summary.center_mode != "none"]
    grid = gaussian.pivot(index="center_mode", columns="sigma", values="mean_validation_accuracy").loc[CENTER_MODES, SIGMAS]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=200)
    values = grid.to_numpy() * 100
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(SIGMAS)), [f"{s:g}" for s in SIGMAS])
    ax.set_yticks(range(len(CENTER_MODES)), [CENTER_LABELS[m] for m in CENTER_MODES])
    ax.set(xlabel="Gaussian width (sigma)", ylabel="16-center placement",
           title="Training-only, source-grouped 5-fold validation accuracy (%)")
    for i in range(len(CENTER_MODES)):
        for j in range(len(SIGMAS)):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center",
                    color="white" if values[i,j] > np.mean(values) else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Mean validation accuracy (%)")
    fig.tight_layout(); fig.savefig(out / "validation_accuracy_heatmap.png", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    for mode in CENTER_MODES:
        subset = gaussian[gaussian.center_mode == mode].sort_values("sigma")
        for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
            ax.plot(subset.sigma, subset[f"mean_validation_{metric}"] * 100, marker="o", ms=4,
                    label=CENTER_LABELS[mode])
    for ax, label in zip(axes, ["Accuracy (%)", "Macro-F1 x 100"]):
        ax.set(xscale="log", xlabel="Gaussian width (sigma), log scale", ylabel=label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Gaussian parameter sensitivity on grouped validation folds")
    fig.tight_layout(); fig.savefig(out / "sigma_sensitivity.png", bbox_inches="tight"); plt.close(fig)


def run_final(records, selection, out):
    included = records[records.included].copy().reset_index(drop=True)
    included["source_group"] = [source_group(c, f) for c, f in zip(included["class"], included.filename)]
    raw = np.asarray(included.feature.tolist(), dtype=np.float64)
    labels = included["class"].to_numpy()
    train, test = included.split.to_numpy() == "train", included.split.to_numpy() == "test"
    fit, evaluation, scaling = scale_fit_and_eval(raw[train], raw[test])
    write_json(out / "final_scaling.json", scaling)
    write_json(out / "training_value_distribution.json", {
        "source": "official training only; after full-training global-max scaling",
        "quantile_probabilities": [0, .25, .5, .75, .95, .99, 1],
        "quantile_values": np.quantile(fit, [0, .25, .5, .75, .95, .99, 1]).tolist(),
    })
    full_features = {"raw": raw, "official_splits": included.split.to_numpy(), "labels": labels,
                     "sample_ids": included.sample_id.to_numpy()}
    included[["sample_id", "class", "filename", "split", "path", "source_group"]].to_csv(out / "sample_manifest.csv", index=False)
    counts = included.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(CLASSES)
    counts["validation"] = 0
    counts.to_csv(out / "final_counts_by_class.csv")
    plans = [("A_linear", None), ("B_default", {"center_mode": "uniform_full", "sigma": 0.1}),
             ("B_width_only_selected", selection["best_uniform"]), ("B_centers_and_width_selected", selection["best_overall"])]
    metrics, centers_records, plot_centers = [], [], {}
    test_predictions = included.loc[test, ["sample_id", "class", "filename", "path", "source_group"]].copy()
    final_models = out / "models"
    final_models.mkdir(exist_ok=True)
    for name, config in plans:
        start = time.perf_counter()
        centers = None if config is None else make_centers(config["center_mode"], fit)
        features_fit = fit if config is None else gaussian_sum(fit, config["sigma"], centers)
        features_test = evaluation if config is None else gaussian_sum(evaluation, config["sigma"], centers)
        transform_seconds = time.perf_counter() - start
        model, prediction, stats = fit_evaluate(features_fit, features_test, labels[train], labels[test])
        stats["test_samples"] = stats.pop("validation_samples")
        metrics.append({"method": name, "center_mode": "none" if config is None else config["center_mode"],
                        "sigma": None if config is None else config["sigma"], "feature_computation_seconds": transform_seconds, **stats})
        features = np.empty_like(raw)
        features[train], features[test] = features_fit, features_test
        full_features[name] = features
        test_predictions[f"prediction_{name}"] = prediction
        test_predictions[f"correct_{name}"] = prediction == labels[test]
        joblib.dump({"classifier": model, "scale": scaling, "centers": centers,
                     "sigma": None if config is None else config["sigma"], "classes": CLASSES}, final_models / f"{name}.joblib")
        cm = confusion_matrix(labels[test], prediction, labels=CLASSES)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f"confusion_{name}.csv")
        plot_confusion(cm, f"{name}: official split 1 test", out / f"confusion_{name}.png", False)
        if config is not None:
            centers_records.append({"method": name, "center_mode": config["center_mode"], "sigma": config["sigma"], "centers": centers.tolist()})
            plot_centers[name] = (centers, config["sigma"])
    write_json(out / "final_gaussian_parameters.json", centers_records)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "final_test_metrics.csv", index=False)
    write_json(out / "final_test_metrics.json", metrics.astype(object).where(pd.notna(metrics), None).to_dict("records"))
    test_predictions.to_csv(out / "final_test_predictions.csv", index=False)
    np.savez_compressed(out / "final_features.npz", **full_features)
    source_overlap = sorted(set(included.loc[train, "source_group"]) & set(included.loc[test, "source_group"]))
    write_json(out / "official_split_source_audit.json", {"source_group_overlap_count": len(source_overlap),
        "overlapping_source_groups": source_overlap,
        "note": "Official split kept unchanged; filename-derived groups are a heuristic, not verified source IDs."})
    fig, axes = plt.subplots(len(plot_centers), 1, figsize=(10, 8), dpi=180, sharex=True)
    x = np.linspace(0, 1, 1000)
    for ax, (name, (centers, sigma)) in zip(axes, plot_centers.items()):
        for center in centers:
            ax.plot(x, np.exp(-((x - center) ** 2) / (2 * sigma ** 2)), lw=1)
        ax.scatter(centers, np.zeros(16), marker="|", c="black")
        ax.set(title=f"{name}: sigma={sigma:g}", ylabel="Gaussian response", ylim=(-.05, 1.1))
    axes[-1].set_xlabel("Scaled input value")
    fig.tight_layout(); fig.savefig(out / "gaussian_parameter_examples.png", bbox_inches="tight"); plt.close(fig)
    return metrics


def write_report(summary, ranked, metrics, selection, cost, out):
    uniform = summary[summary.center_mode == "uniform_full"].sort_values("sigma")
    fixed_width = summary[(summary.center_mode != "none") & (summary.sigma == .1)]
    rows = ["| 中心分布 | σ | 分组验证准确率（均值 ± 折间SD） | 验证 Macro-F1 |",
            "|---|---:|---:|---:|"]
    for _, r in ranked.head(10).iterrows():
        rows.append(f"| {r.center_mode} | {r.sigma:g} | {r.mean_validation_accuracy:.2%} ± {r.std_validation_accuracy:.2%} | {r.mean_validation_macro_f1:.4f} |")
    tests = ["| 方法 | 中心分布 | σ | 测试准确率 | 测试 Macro-F1 |", "|---|---|---:|---:|---:|"]
    for _, r in metrics.iterrows():
        width = "—" if pd.isna(r.sigma) else f"{r.sigma:g}"
        tests.append(f"| {r.method} | {r.center_mode} | {width} | {r.accuracy:.2%} | {r.macro_f1:.4f} |")
    default = metrics[metrics.method == "B_default"].iloc[0]
    best = metrics[metrics.method == "B_centers_and_width_selected"].iloc[0]
    distribution = json.loads((out / "training_value_distribution.json").read_text(encoding="utf-8"))
    median, p95 = distribution["quantile_values"][2], distribution["quantile_values"][4]
    text = f"""# Gaussian 参数影响实验（真实 HMDB51 六类）

本轮复用原始视频特征缓存，不重新采样。官方训练集成功读取 412 个样本，官方测试集 176 个。只在 412 个训练样本内部做 5 折分组验证；每次用 4 折训练、1 折验证，每个训练样本恰好作为验证样本一次。按文件名动作标签之前的来源标题分组并归一化大小写/标点，同一推断来源不会跨该折训练和验证。它仍是启发式来源识别，不能保证发现所有实际同源视频。

随机种子 42；16 个输出保持不变，Gaussian 峰值为 1，所有 Gaussian 参数在训练分类器时固定。分类器设置与原始 A/B 一致：StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=5000, tol=1e-6)。为稳定记录耗时，本次数值库线程限制为 1，未使用 GPU。

## 参数范围与中心定义

- σ：{', '.join(str(s) for s in SIGMAS)}，统一在全局最大值缩放后的坐标上定义。
- `uniform_full`：16 个中心在 [0,1] 等距（原默认）。
- `power2`、`power3`：中心分别是 linspace(0,1,16) 的平方/立方，保留两个端点并在低值区加密。
- `uniform_p95`、`uniform_p99`：16 个中心均匀分布于 [0,该折训练值的95%/99%分位数]；这是对原默认中心范围的显式扩展。
- `quantiles`：16 个中心位于该折训练值的均匀分位数；相同分位值允许产生重合中心，不额外抖动。

每一折先从该折训练集拟合全局最大值（零值时除数设为1），再仅用训练值拟合中心。验证超界值截断并单独记录。每个方案的 StandardScaler 只拟合对应训练 Gaussian 特征，输出严格为 N×16。

## 参数是否有影响

固定默认中心、仅改变宽度时，验证平均准确率从 {uniform.mean_validation_accuracy.min():.2%} 到 {uniform.mean_validation_accuracy.max():.2%}；固定 σ=0.1、仅改变中心策略时，从 {fixed_width.mean_validation_accuracy.min():.2%} 到 {fixed_width.mean_validation_accuracy.max():.2%}。这些差异显示本次实验对参数有敏感性，但折间波动和48组搜索造成的选择偏差也需要考虑，不能只凭最高点断言普遍提升。

全部官方训练值缩放后的中位数是 {median:.4f}，95% 分位数是 {p95:.4f}，而默认均匀中心间距为 0.0667。数值集中在低值区是测试中心加密的依据；该统计仅用于解释已完成的参数对比，每折的中心仍由各自训练部分拟合。

选择规则预先固定为：验证平均准确率最高，其次平均 Macro-F1，再其次准确率折间SD较小，最后配置ID。测试数据不参与排名。参数在 `selection_frozen_before_test.json` 保存后才计算最终测试指标。

验证前10组：

{chr(10).join(rows)}

## 冻结参数后的测试

先固定全部结果再报告；不会因测试结果继续修改参数。选中的仅宽度方案：{selection['best_uniform']}；选中的宽度+中心方案：{selection['best_overall']}。随后用全部 412 个官方训练样本重新拟合缩放、数据相关中心和分类器。

{chr(10).join(tests)}

联合选参相对默认 B 的测试准确率变化 {(best.accuracy-default.accuracy)*100:+.2f} 个百分点，Macro-F1 变化 {best.macro_f1-default.macro_f1:+.4f}。验证最优并不保证测试最优，测试集在前一轮已报告过，因此这是探索性后续实验，不是全新盲测。

## 成本与文件

复用缓存，不新增下载/解压/视频采样成本。共训练 {cost['classifier_fits']} 个分类器；总耗时 {cost['total_seconds']:.2f} 秒，进程峰值 RSS {cost['peak_process_rss_bytes']/1024**2:.1f} MiB。

`validation_summary.csv` 是所有配置的验证指标；`gaussian_validation_ranking.csv` 是完整排名；`validation_fold_metrics.csv` 是每折指标和耗时；`training_validation_folds.csv`、`fold_class_counts.csv`、`fold_scaling_and_group_audit.json` 可核对划分及训练/验证样本数。`fold_centers.json` 记录实际中心，`fold_features/` 保存每折缩放输入与全部 Gaussian 输出；`final_features.npz` 和 `models/` 保存最终特征及可预测模型。排除记录沿用 `exclusions.csv`。

后续应结合 σ 曲线及训练分布检查：中心是否覆盖有效输入区、过宽时响应是否趋同、过窄时响应是否过于稀疏。当前输入只描述平均帧差；Gaussian 求和还会丢失16个时间位置的顺序，因此即使参数变化有效，也不能据此认为输入已足以区分所有动作。
"""
    (out / "参数实验说明.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/hmdb51_six_split1/feature_cache.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hmdb51_gaussian_tuning_v1"))
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cache_bytes = args.cache.read_bytes()
    records = pd.DataFrame(joblib.load(args.cache)["records"])
    records["sample_id"] = np.arange(len(records))
    assert records.filename.is_unique
    assert set(records.split) == {"train", "test"}
    records[~records.included].to_csv(out / "exclusions.csv", index=False)
    training = records[records.included & (records.split == "train")].copy().reset_index(drop=True)
    training["source_group"] = [source_group(c, f) for c, f in zip(training["class"], training.filename)]
    config = {"seed": SEED, "folds": 5, "sigmas": SIGMAS, "center_modes": CENTER_MODES,
              "number_gaussian_configurations": len(SIGMAS) * len(CENTER_MODES),
              "classifier": CLASSIFIER_SETTINGS, "cache_path": str(args.cache.resolve()),
              "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
              "selection_rule": "mean accuracy desc, mean macro-F1 desc, accuracy SD asc, config_id asc",
              "prior_test_results_already_observed": True, "blas_threads": 1}
    write_json(out / "experiment_config.json", config)
    with threadpool_limits(limits=1):
        summary, ranked = run_validation(training, out, 5, SEED)
        best_uniform = rank_results(summary[summary.center_mode == "uniform_full"]).iloc[0]
        best_overall = ranked.iloc[0]
        selection = {"best_uniform": {"center_mode": "uniform_full", "sigma": float(best_uniform.sigma)},
                     "best_overall": {"center_mode": best_overall.center_mode, "sigma": float(best_overall.sigma)},
                     "selected_by": config["selection_rule"], "test_used_for_selection": False,
                     "freeze_time_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
        write_json(out / "selection_frozen_before_test.json", selection)
        print("Frozen training-only selection:", selection, flush=True)
        metrics = run_final(records, selection, out)
    plot_sensitivity(summary, out)
    counts = pd.read_csv(out / "fold_class_counts.csv")
    audit = {"feature_cache_reused": True, "official_training_samples": len(training),
             "training_source_groups": int(training.source_group.nunique()),
             "validation_folds": 5, "gaussian_output_dimensions": 16,
             "source_overlap_train_validation_every_fold": 0,
             "test_used_for_selection": False, "all_fits_converged": True,
             "per_fold_training_samples": counts.groupby("fold")["train"].sum().tolist(),
             "per_fold_validation_samples": counts.groupby("fold")["validation"].sum().tolist()}
    write_json(out / "audit.json", audit)
    cost = {"total_seconds": time.perf_counter() - start,
            "classifier_fits": 5 * (len(SIGMAS)*len(CENTER_MODES) + 1) + 4,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "memory_total_bytes": psutil.virtual_memory().total, "logical_cpu_count": psutil.cpu_count(),
            "gpu_used": False, "python": platform.python_version(), "sklearn": sklearn.__version__,
            "blas_threads": 1, "preprocessing_cache_reused": True,
            "note": "End-to-end time through computation/plotting, excluding final report serialization."}
    write_json(out / "runtime.json", cost)
    write_report(summary, ranked, metrics, selection, cost, out)
    print("\nTop validation configurations:\n", ranked.head(8).to_string(index=False))
    print("\nFrozen-parameter test results:\n", metrics.to_string(index=False))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
