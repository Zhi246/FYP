#!/usr/bin/env python3
"""Render the completed finite search and paired diagnostics; never fit models."""
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
from sklearn.metrics import confusion_matrix

import tune_kth_spatiotemporal as task

LABELS = {"A_linear": "Linear16", "G0_mean": "G0 mean training",
          "G1_head_cycle": "G1 Cycle-trained head", "I_ideal": "Ideal RBF (diagnostic)"}
COLORS = ["#64748b", "#248c87"]


def read(path):
    return pd.read_csv(path, low_memory=False, dtype={"id": str, "sample_id": str})


def paired_bootstrap(table, predictions):
    data = predictions[(predictions.split == "test")&(predictions.condition == "cycle")]
    arrays = {}; baseline_ids = None
    for identifier, part in data.groupby("id"):
        part = part.sort_values(["repeat", "sample_id"])
        if baseline_ids is None:
            baseline_ids = part.sample_id.to_numpy()
            subjects = part[part.repeat == 0].subject.to_numpy()
        np.testing.assert_array_equal(part.sample_id.to_numpy(), baseline_ids)
        arrays[identifier] = part.correct.to_numpy(dtype=float).reshape(10, 862)
    people = np.unique(subjects); counts = np.array([(subjects == p).sum() for p in people])
    rng = np.random.default_rng(42)
    participant_draws = rng.integers(0, len(people), (2000, len(people)))
    repeat_draws = rng.integers(0, 10, (2000, 10))
    records = []
    cases = [(("adjusted", name), ("default", name), "adjusted_minus_default") for name in task.SCHEMES]
    cases += [(("adjusted", name), ("adjusted", "A_linear"), "adjusted_G_minus_adjusted_Linear")
              for name in ["G0_mean", "G1_head_cycle"]]
    for first, second, label in cases:
        differences = arrays[table.loc[first, "id"]]-arrays[table.loc[second, "id"]]
        sums = np.column_stack([differences[:, subjects == p].sum(axis=1) for p in people])
        boot = sums[repeat_draws[:, :, None], participant_draws[:, None, :]].sum(axis=(1, 2))/(10*counts[participant_draws].sum(axis=1))
        low, high = np.quantile(boot, [.025, .975])
        records.append(dict(comparison=label, first_stage=first[0], first_scheme=first[1],
            second_stage=second[0], second_scheme=second[1], accuracy_difference=float(differences.mean()),
            paired_95_low=float(low), paired_95_high=float(high), participants=9,
            cycle_repeats=10, bootstrap_repeats=2000, seed=42, exploratory=True))
    return pd.DataFrame(records)


def class_rates(table, predictions):
    rows = []
    for (stage, name), item in table.iterrows():
        part = predictions[(predictions.id == item.id)&(predictions.split == "test")&(predictions.condition == "cycle")]
        for actual in task.shared.CLASSES:
            subset = part[part["class"] == actual]
            for predicted in task.shared.CLASSES:
                count = int((subset.predicted == predicted).sum())
                rows.append(dict(stage=stage, scheme=name, true_class=actual, predicted_class=predicted,
                    count=count, repeated_predictions=len(subset), fraction=count/len(subset),
                    unique_clips=int(subset.sample_id.nunique())))
    return pd.DataFrame(rows)


def performance(ax, table):
    for j, stage in enumerate(["default", "adjusted"]):
        values = table.loc[stage].loc[task.previous.PRIMARY]
        bars = ax.bar(np.arange(3)+(j-.5)*.35, values.test_accuracy_mean*100, .35,
            yerr=values.test_accuracy_std*100, capsize=3, color=COLORS[j], label=stage.title())
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set(ylim=(0, 100), xticks=np.arange(3), xticklabels=["Linear16", "G0", "G1"],
        ylabel="Test accuracy (%)", title="Same input, clips and 10 Cycle draws\nError bars: bank-draw SD, not subject uncertainty")
    ax.legend()


def search_panel(ax, candidates):
    subset = candidates[candidates.scheme == "G1_head_cycle"]
    heat = subset.groupby(["sigma", "calibration_alpha"]).validation_macro_f1_mean.max().unstack()
    image = ax.imshow(heat.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set(xticks=range(3), xticklabels=[f"{v:g}" for v in heat.columns],
        yticks=range(3), yticklabels=[f"{v:g}" for v in heat.index], xlabel="Calibration anchor alpha",
        ylabel="Joint Gaussian target sigma", title="G1 validation Macro-F1\nBest of four head alphas in each cell; no test ranking")
    lo, hi = heat.min().min(), heat.max().max()
    for r in range(3):
        for c in range(3):
            value = heat.iloc[r, c]
            ax.text(c, r, f"{value:.4f}", ha="center", va="center",
                    color="white" if value > (lo+hi)/2 else "#152a36", fontsize=12)
    ax.figure.colorbar(image, ax=ax, shrink=.75)


def recall_panel(ax, rates):
    values = rates[(rates.scheme == "G1_head_cycle")&(rates.true_class == rates.predicted_class)]
    for j, stage in enumerate(["default", "adjusted"]):
        recalls = values[values.stage == stage].set_index("true_class").loc[task.shared.CLASSES, "fraction"]
        bars = ax.bar(np.arange(6)+(j-.5)*.35, recalls*100, .35, color=COLORS[j], label=stage.title())
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set(ylim=(0, 100), ylabel="Test recall (%)", xticks=np.arange(6),
        xticklabels=["walk", "jog", "run", "box", "wave", "clap"],
        title="G1 per-class recall across the same ten draws\nRepeated draws are not additional independent videos")
    ax.legend()


def matrix_panel(ax, root, selected):
    matrix = pd.read_csv(root/f"confusion_{selected['id']}_first_draw.csv", index_col=0).to_numpy()
    task.shared.confusion_panel(ax, matrix,
        "Validation-selected Gaussian: "+selected["scheme"]+"\nFirst fixed draw seed=3042, 862 clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=task.OUT)
    args = parser.parse_args(); root = args.output.resolve(); begin = time.perf_counter()
    assert json.loads((root/"COMPLETE.json").read_text())["completed"]
    table = read(root/"results_table.csv").set_index(["stage", "scheme"])
    candidates = read(root/"validation_candidates.csv")
    ablations = []
    g1 = candidates[candidates.scheme == "G1_head_cycle"]
    for label, subset in [
        ("head_only", g1[(g1.sigma == .25)&(g1.calibration_alpha == .01)]),
        ("calibration_only", g1[(g1.sigma == .25)&(g1.head_alpha == .1)]),
        ("width_only", g1[(g1.calibration_alpha == .01)&(g1.head_alpha == .1)])]:
        part = subset.copy(); part["ablation"] = label; ablations.append(part)
    pd.concat(ablations, ignore_index=True).to_csv(root/"validation_single_parameter_ablations.csv", index=False)
    source = Path(json.loads((root/"signature.json").read_text())["source"])
    old_diagnostics = read(source/"new_feature_diagnostics.csv"); old_diagnostics["stage"] = "default"
    new_diagnostics = read(root/"joint_feature_diagnostics.csv"); new_diagnostics["stage"] = "adjusted"
    id_to_name = table.reset_index().set_index("id").scheme.to_dict()
    new_diagnostics["scheme"] = new_diagnostics.id.map(id_to_name)
    diagnostic_summary = pd.concat([old_diagnostics, new_diagnostics], ignore_index=True)
    diagnostic_summary.groupby(["stage", "scheme", "split", "condition"]).relative_joint_feature_rmse.agg(["mean", "std"]).to_csv(root/"calibration_error_summary.csv")
    predictions = read(root/"sample_predictions.csv")
    manifest = read(root/"sample_manifest.csv")
    # Reused default predictions lack scenario; restore metadata by sample ID only.
    predictions["scenario"] = predictions.sample_id.map(manifest.set_index("sample_id").scenario)
    pairs = paired_bootstrap(table, predictions); pairs.to_csv(root/"paired_comparisons.csv", index=False)
    rates = class_rates(table, predictions); rates.to_csv(root/"class_confusion_rates.csv", index=False)
    rows = []
    for (stage, name), row in table.iterrows():
        part = predictions[(predictions.id == row.id)&(predictions.split == "test")&(predictions.condition == "cycle")]
        for (subject, repeat), group in part.groupby(["subject", "repeat"]):
            rows.append(dict(stage=stage, scheme=name, subject=subject, repeat=repeat,
                             accuracy=float(group.correct.mean()), unique_n=len(group)))
    pd.DataFrame(rows).to_csv(root/"per_subject_cycle_metrics.csv", index=False)
    scenarios = []
    for (stage, name), row in table.iterrows():
        part = predictions[(predictions.id == row.id)&(predictions.split == "test")&(predictions.condition == "cycle")]
        for scenario, group in part.groupby("scenario"):
            scenarios.append(dict(stage=stage, scheme=name, scenario=scenario, accuracy=float(group.correct.mean()),
                                  unique_n=int(group.sample_id.nunique()), repeated_predictions=len(group)))
    pd.DataFrame(scenarios).to_csv(root/"scenario_metrics.csv", index=False)
    selected = json.loads((root/"locked_selection_before_test.json").read_text())["gaussian"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 240,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure_dir = root/"figures"; figure_dir.mkdir(exist_ok=True)
    panels = [("performance", lambda ax: performance(ax, table)),
              ("validation_search", lambda ax: search_panel(ax, candidates)),
              ("class_recall", lambda ax: recall_panel(ax, rates)),
              ("confusion", lambda ax: matrix_panel(ax, root, selected))]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for (_, fn), ax in zip(panels, axes.ravel()): fn(ax)
    fig.tight_layout(); fig.savefig(root/"four_panel_summary.png"); plt.close(fig)
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(9, 6)); fn(ax); fig.tight_layout()
        fig.savefig(figure_dir/(name+".png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(16, 5)); ax.axis("off")
    cells = []
    for name in task.previous.PRIMARY:
        for stage in ["default", "adjusted"]:
            row = table.loc[(stage, name)]
            params = f"head={row.head_alpha:g}" if name == "A_linear" else f"s={row.sigma:g}; cal={row.calibration_alpha:g}; head={row.head_alpha:g}"
            cells.append([LABELS[name], stage.title(), f"{row.train_mean_accuracy*100:.2f}%",
                f"{row.validation_macro_f1_mean:.4f}", f"{row.test_accuracy_mean*100:.2f} +/- {row.test_accuracy_std*100:.2f}%",
                f"{row.test_macro_f1_mean:.4f}", params])
    drawing = ax.table(cellText=cells, colLabels=["Path", "Configuration", "Train mean Acc", "Val F1",
        "Test Cycle Acc", "Test Cycle F1", "Anchor parameters"], loc="center", cellLoc="center",
        colWidths=[.18, .10, .12, .09, .16, .12, .25])
    drawing.auto_set_font_size(False); drawing.set_fontsize(10); drawing.scale(1, 2.1)
    for c in range(7): drawing[0, c].set_facecolor("#d8ebe9")
    for r in [2, 4, 6]:
        for c in range(7): drawing[r, c].set_facecolor("#edf7f5")
    ax.set_title("KTH fixed region4 x time4 inputs | 760 / 767 / 862 clips\n10 paired Cycle draws; head sum-alpha = anchor x rows / 760; exploratory test", pad=18)
    fig.tight_layout(); fig.savefig(root/"results_table.png", bbox_inches="tight"); plt.close(fig)
    folds = read(root/"group_cv_fold_means.csv")
    cv = folds.groupby(["stage", "scheme"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    cv.to_csv(root/"group_cv_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, stage in enumerate(["default", "adjusted"]):
        names = task.previous.PRIMARY
        means = [cv.loc[(stage, n), ("accuracy", "mean")]*100 for n in names]
        std = [cv.loc[(stage, n), ("accuracy", "std")]*100 for n in names]
        bars = axes[0].bar(np.arange(3)+(j-.5)*.35, means, .35, yerr=std, capsize=3, color=COLORS[j], label=stage.title())
        axes[0].bar_label(bars, fmt="%.2f", padding=7, fontsize=9)
    axes[0].set(ylim=(0, 100), ylabel="Accuracy (%)", xticks=range(3), xticklabels=["Linear", "G0", "G1"],
                title="Four training-participant folds\nFixed selected parameters; error bars are fold SD")
    axes[0].legend()
    names = ["A_linear", "G1_head_cycle", "I_ideal"]
    for j, stage in enumerate(["default", "adjusted"]):
        values = [table.loc[(stage, n)].test_accuracy_mean*100 for n in names]
        bars = axes[1].bar(np.arange(3)+(j-.5)*.35, values, .35, color=COLORS[j], label=stage.title())
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set(ylim=(0, 100), ylabel="Test accuracy (%)", xticks=range(3), xticklabels=["Linear", "Bank G1", "Ideal (software)"],
                title="Bank path versus ideal software reference\nIdeal is not a hardware score or guaranteed upper bound")
    axes[1].legend(); fig.tight_layout(); fig.savefig(figure_dir/"group_cv_and_ideal_reference.png"); plt.close(fig)
    summary = dict(selected_gaussian=selected, paired_comparisons=pairs.to_dict("records"),
        g1_recall=rates[(rates.scheme == "G1_head_cycle")&(rates.true_class == rates.predicted_class)].to_dict("records"),
        boxing_to_clapping=rates[(rates.true_class == "boxing")&(rates.predicted_class == "handclapping")].to_dict("records"),
        render_and_analysis_wall_seconds=time.perf_counter()-begin,
        render_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        figure_captions_zh={
            "performance": "相同输入与测试抽样下的调参前后准确率；误差线仅为响应库抽样标准差。",
            "validation_search": "G1各宽度与校准正则化组合的验证Macro-F1，每格取四个分类头参数中的最佳值，不用测试排名。",
            "class_recall": "各类G1召回率揭示调参是否只改善总分而牺牲某些动作；十轮抽样并非十倍独立视频。",
            "confusion": "按验证预选的Gaussian配置，在固定首轮seed=3042的测试混淆矩阵；纵轴真实，横轴预测。"})
    task.shared.write_json(root/"analysis_summary.json", summary)
    for path in [Path(__file__), Path("test_tune_kth_spatiotemporal.py")]:
        shutil.copy2(path, root/"source_code"/path.name)
    print(pairs.to_string(index=False))
    print("Figures and paired diagnostics saved:", root)


if __name__ == "__main__": main()
