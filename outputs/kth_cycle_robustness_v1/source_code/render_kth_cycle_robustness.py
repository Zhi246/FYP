#!/usr/bin/env python3
"""Postprocess saved Cycle results; no training or model selection."""
from __future__ import annotations

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

import run_kth_cycle_robustness as experiment

DISPLAY = {"A_linear": "Linear16 reference", "I_ideal": "Ideal joint RBF reference",
           "G0_mean": "G0 mean training", "G1_head_cycle": "G1 head-cycle training",
           "G2_calibration_and_head_cycle": "G2 calibration + head cycles"}
SHORT = ["G0 mean", "G1 head cycles", "G2 cal + head cycles"]
COLORS = ["#64748b", "#d28b43", "#278f8a"]


def paired_analysis(predictions, root):
    """Exploratory paired subject/draw bootstrap, not new independent data."""
    records = []
    draws = predictions[(predictions.split == "test") & (predictions.condition == "cycle")]
    names = ["G1_head_cycle", "G2_calibration_and_head_cycle", "A_linear", "I_ideal"]
    base = draws[draws.scheme == "G0_mean"].sort_values(["repeat", "sample_id"])
    expected_ids = base[base.repeat == 0].sample_id.to_numpy()
    subjects = base[base.repeat == 0].subject.to_numpy()
    unique = np.unique(subjects)
    base_correct = base.correct.to_numpy().reshape(10, len(expected_ids)).astype(float)
    counts = np.array([(subjects == person).sum() for person in unique])
    for name in names:
        other = draws[draws.scheme == name].sort_values(["repeat", "sample_id"])
        np.testing.assert_array_equal(other.sample_id.to_numpy(), base.sample_id.to_numpy())
        difference = other.correct.to_numpy().reshape(10, len(expected_ids)).astype(float)-base_correct
        sums = np.column_stack([difference[:, subjects == person].sum(axis=1) for person in unique])
        rng = np.random.default_rng(42)
        person_draws = rng.integers(0, len(unique), size=(2000, len(unique)))
        cycle_draws = rng.integers(0, 10, size=(2000, 10))
        boot = sums[cycle_draws[:, :, None], person_draws[:, None, :]].sum(axis=(1,2)) / (
            10*counts[person_draws].sum(axis=1))
        low, high = np.quantile(boot, [.025, .975])
        records.append({"comparison": name+" minus G0_mean", "accuracy_difference": difference.mean(),
                        "paired_subject_and_draw_bootstrap_95_low": low,
                        "paired_subject_and_draw_bootstrap_95_high": high,
                        "test_subjects": len(unique), "draw_repeats": 10, "bootstrap_repeats": 2000,
                        "seed": 42, "independent_new_data": False})
    frame = pd.DataFrame(records)
    frame.to_csv(root/"paired_comparisons.csv", index=False)
    return frame


def class_confusions(root):
    rows = []
    for name in experiment.G_NAMES:
        matrix = pd.read_csv(root/f"confusion_{name}_cycle_all_draws.csv", index_col=0)
        for true in experiment.shared.CLASSES:
            support = int(matrix.loc[true].sum())
            for predicted in experiment.shared.CLASSES:
                rows.append({"scheme": name, "true_class": true, "predicted_class": predicted,
                             "prediction_count_over_10_draws": int(matrix.loc[true, predicted]),
                             "prediction_count_denominator_over_10_draws": support,
                             "unique_clips_in_true_class": support//10,
                             "fraction": float(matrix.loc[true, predicted]/support)})
    frame = pd.DataFrame(rows)
    frame.to_csv(root/"class_confusion_rates.csv", index=False)
    return frame


def accuracy_panel(ax, table):
    x = np.arange(3)
    columns = ["test_pool_mean_accuracy", "test_cycle_accuracy_mean", "test_noise_accuracy_mean"]
    errors = [None, "test_cycle_accuracy_std", "test_noise_accuracy_std"]
    for i, (column, error, color, label) in enumerate(zip(columns, errors, COLORS,
                                                         ["Pool mean reference", "One held-out Cycle", "Cycle + simulator noise"])):
        values = table.loc[experiment.G_NAMES, column].to_numpy()*100
        error_values = table.loc[experiment.G_NAMES, error].to_numpy()*100 if error else None
        bars = ax.bar(x+(i-1)*.25, values, .25, color=color, yerr=error_values, capsize=3, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set(ylim=(0,100), ylabel="Test accuracy (%)", xticks=x, xticklabels=SHORT,
           title="Same input and joint formula | different response training")
    ax.legend(fontsize=8, loc="upper right")


def feature_panel(ax, diagnostics):
    mean = diagnostics[(diagnostics.split == "test") & (diagnostics.condition == "pool_mean")].set_index("scheme")
    cycle = diagnostics[(diagnostics.split == "test") & (diagnostics.condition == "cycle")].groupby("scheme").relative_joint_feature_rmse.agg(["mean", "std"])
    x = np.arange(3)
    for i, (values, errors, color, label) in enumerate([
        (mean.loc[experiment.G_NAMES, "relative_joint_feature_rmse"].to_numpy(), None, COLORS[0], "Pool mean reference"),
        (cycle.loc[experiment.G_NAMES, "mean"].to_numpy(), cycle.loc[experiment.G_NAMES, "std"].to_numpy(), COLORS[2], "Held-out Cycle, 10 draws")]):
        bars = ax.bar(x+(i-.5)*.32, values*100, .32, color=color,
                      yerr=None if errors is None else errors*100, capsize=3, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(ylabel="Joint-feature relative RMSE (%)", xticks=x, xticklabels=SHORT,
           title="Reconstruction error relative to matched ideal RBF\nG0 and G1 have identical Gaussian features")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(cycle["mean"].max()*140, 22))


def cv_panel(ax, folds):
    for name, color in zip(experiment.G_NAMES, COLORS):
        subset = folds[folds.scheme == name].sort_values("fold")
        ax.plot(subset.fold+1, subset.accuracy*100, "o-", color=color, label=DISPLAY[name])
    ax.set(ylim=(0,100), xticks=[1,2,3,4], xlabel="Subject-grouped fold", ylabel="Validation accuracy (%)",
           title="Four-fold audit within the 8 training people\nEach point averages 5 bank draws; not official validation")
    ax.legend(fontsize=8, loc="upper left")


def confusion_panel(ax, root, chosen):
    matrix = pd.read_csv(root/f"confusion_{chosen}_cycle_first_draw.csv", index_col=0).to_numpy()
    experiment.shared.confusion_panel(ax, matrix, "Validation-selected G1 | first fixed test draw\nseed=3042; 862 clips, not the best draw")


def draw_table(root, table):
    rows = []
    for name in experiment.SCHEMES:
        row = table.loc[name]
        rows.append([DISPLAY[name], f"{row.validation_cycle_macro_f1_mean:.4f}",
                     f"{row.test_pool_mean_accuracy*100:.2f}%",
                     f"{row.test_cycle_accuracy_mean*100:.2f} +/- {row.test_cycle_accuracy_std*100:.2f}%",
                     f"{row.test_cycle_macro_f1_mean:.4f}",
                     f"{row.test_noise_accuracy_mean*100:.2f} +/- {row.test_noise_accuracy_std*100:.2f}%",
                     f"{row.test_cycle_accuracy_drop_from_pool_mean*100:+.2f}"])
    fig, ax = plt.subplots(figsize=(17,4.7)); ax.axis("off")
    cells = ax.table(cellText=rows,
        colLabels=["Scheme", "Val Cycle F1", "Test pool mean", "Test Cycle Acc", "Test Cycle F1", "Cycle + noise Acc", "Mean-to-Cycle\ndrop (pp)"],
        colWidths=[.27,.10,.11,.14,.11,.16,.10], loc="center", cellLoc="center")
    cells.auto_set_font_size(False); cells.set_fontsize(10); cells.scale(1,2.2)
    for column in range(7):
        cells[0,column].set_facecolor("#d8ebe9")
        cells[4,column].set_facecolor("#edf7f5")
    ax.set_title("KTH | 760 / 767 / 862 clips | 240 / 80 / 80 bank indices | G1 selected on validation\n+/- = SD over 10 bank draws of the same clips, not cross-subject uncertainty", pad=12)
    fig.tight_layout(); fig.savefig(root/"results_table.png", bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/kth_cycle_robustness_v1"))
    args = parser.parse_args()
    root = args.output.resolve(); started = time.perf_counter()
    assert json.loads((root/"COMPLETE.json").read_text())["completed"]
    chosen = json.loads((root/"locked_selection_before_test.json").read_text())["selected_scheme"]
    table = pd.read_csv(root/"results_table.csv").set_index("scheme")
    predictions = pd.read_csv(root/"sample_predictions.csv")
    diagnostics = pd.read_csv(root/"feature_diagnostics.csv")
    folds = pd.read_csv(root/"group_cv_fold_means.csv")
    paired = paired_analysis(predictions, root)
    confusions = class_confusions(root)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "savefig.dpi":240,
                         "axes.spines.top":False, "axes.spines.right":False})
    (root/"figures").mkdir(exist_ok=True)
    draw_table(root, table)
    functions = [("accuracy", lambda ax: accuracy_panel(ax, table)),
                 ("feature_error", lambda ax: feature_panel(ax, diagnostics)),
                 ("group_cv", lambda ax: cv_panel(ax, folds)),
                 ("confusion", lambda ax: confusion_panel(ax, root, chosen))]
    fig, axes = plt.subplots(2,2,figsize=(15,11))
    for (_, function), ax in zip(functions, axes.ravel()): function(ax)
    fig.tight_layout(); fig.savefig(root/"four_panel_summary.png"); plt.close(fig)
    for name, function in functions:
        fig, ax = plt.subplots(figsize=(9,6)); function(ax); fig.tight_layout()
        fig.savefig(root/"figures"/("panel_"+name+".png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    values = table.loc[experiment.G_NAMES, "test_cycle_accuracy_drop_from_pool_mean"].to_numpy()*100
    bars = ax.bar(SHORT, values, color=COLORS)
    ax.bar_label(bars, fmt="%+.2f", padding=4)
    ax.axhline(0,color="black",linewidth=.8)
    ax.set(ylabel="Pool mean minus single Cycle accuracy (pp)", title="A small drop alone does not imply good recognition\nG2 has lower mean-response accuracy")
    fig.tight_layout(); fig.savefig(root/"figures/mean_to_cycle_drop.png"); plt.close(fig)
    for name in experiment.G_NAMES:
        fig, ax = plt.subplots(figsize=(8,7))
        matrix = pd.read_csv(root/f"confusion_{name}_cycle_first_draw.csv", index_col=0).to_numpy()
        experiment.shared.confusion_panel(ax,matrix,DISPLAY[name]+" | first test draw, seed=3042")
        fig.tight_layout(); fig.savefig(root/"figures"/("confusion_"+name+".png")); plt.close(fig)
    # Numerical report material is derived from saved predictions, never a new fit.
    summary = {"chosen_before_test": chosen, "g1_minus_g0_cycle_accuracy_pp": float(
        (table.loc["G1_head_cycle","test_cycle_accuracy_mean"]-table.loc["G0_mean","test_cycle_accuracy_mean"])*100),
        "g1_minus_g0_cycle_macro_f1": float(table.loc["G1_head_cycle","test_cycle_macro_f1_mean"]-table.loc["G0_mean","test_cycle_macro_f1_mean"]),
        "g1_minus_linear_cycle_accuracy_pp": float((table.loc["G1_head_cycle","test_cycle_accuracy_mean"]-table.loc["A_linear","test_cycle_accuracy_mean"])*100),
        "paired_bootstrap": paired.to_dict("records"),
        "group_cv_rank_disagrees_with_official_validation": bool(folds.groupby("scheme").macro_f1.mean().loc[experiment.G_NAMES].idxmax() != chosen),
        "pooled_classification_f1_is_not_the_same_as_mean_of_draw_f1": True,
        "render_wall_seconds": time.perf_counter()-started,
        "render_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    experiment.shared.write_json(root/"analysis_summary.json", summary)
    shutil.copy2(__file__, root/"source_code"/Path(__file__).name)
    shutil.copy2("test_kth_cycle_robustness.py", root/"source_code/test_kth_cycle_robustness.py")
    print("Table, panels, paired analysis and source snapshots saved:",root)
    print(paired.to_string(index=False))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
