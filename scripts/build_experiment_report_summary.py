#!/usr/bin/env python3
"""Build report tables from completed CSV records, without rerunning experiments."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
ROWS = []
SOURCES = {}


def load(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(row, key):
    value = row.get(key)
    return float(value) if value not in [None, ""] else None


def add(phase, row, model, source, representation, response_protocol, *,
        train_n=760, validation_n=767, test_n=862, dimension=256,
        train="train_accuracy", validation="validation_accuracy", validation_f1="validation_macro_f1",
        test="test_accuracy", test_f1="test_macro_f1", test_std=None):
    ROWS.append(dict(phase=phase, model=model, input_representation=representation,
        response_protocol=response_protocol, output_dimension=dimension,
        train_unique_n=train_n, validation_unique_n=validation_n, test_unique_n=test_n,
        train_accuracy=numeric(row, train), validation_accuracy=numeric(row, validation),
        validation_macro_f1=numeric(row, validation_f1), test_accuracy=numeric(row, test),
        test_accuracy_std=numeric(row, test_std) if test_std else None,
        test_macro_f1=numeric(row, test_f1), source_csv=source))


def main():
    source = "outputs/kth_six_subject_split_v1/results_table.csv"
    for row in load(source):
        add("KTH_sum_initial", row, row["method"], source, "temporal16", "scalar Gaussian sum, no response bank", dimension=16)
    source = "outputs/kth_gaussian_expanded_v2/comparison_previous_vs_expanded.csv"
    for row in load(source):
        if row["version"] == "expanded":
            add("KTH_sum_expanded", row, row["method"], source, "temporal16", "scalar Gaussian sum, no response bank", dimension=16)
    source = "outputs/kth_joint_rbf256_strict_v1/results_table.csv"
    wanted = ["Linear16_standardized", "Hardware_joint256_default", "Hardware_joint256_default_head_tuned", "Hardware_joint256_tuned"]
    for row in load(source):
        if row["scheme"] in wanted:
            add("KTH_joint_400mean", row, row["scheme"], source, "temporal16", "all 400 Cycles mean; not held-out indices",
                dimension=int(row["output_dimension"]))
    source = "outputs/kth_cycle_robustness_v1/results_table.csv"
    for row in load(source):
        add("KTH_joint_cycle_holdout", row, row["scheme"], source, "temporal16", "240/80/80 Cycle pools; test single-Cycle 10 draws",
            dimension=int(row["output_dimension"]), train="train_pool_mean_accuracy",
            validation="validation_cycle_accuracy_mean", validation_f1="validation_cycle_macro_f1_mean",
            test="test_cycle_accuracy_mean", test_std="test_cycle_accuracy_std", test_f1="test_cycle_macro_f1_mean")
    source = "outputs/kth_spatiotemporal_v1/results_table.csv"
    for row in load(source):
        if row["representation"] == "region4_time4":
            add("KTH_joint_region4_time4", row, row["scheme"], source, "region4_time4", "240/80/80 Cycle pools; test single-Cycle 10 draws",
                dimension=int(row["output_dimension"]), train="train_pool_mean_accuracy",
                validation="validation_cycle_accuracy_mean", validation_f1="validation_cycle_macro_f1_mean",
                test="test_cycle_accuracy_mean", test_std="test_cycle_accuracy_std", test_f1="test_cycle_macro_f1_mean")
    source = "outputs/kth_spatiotemporal_tuning_v1/results_table.csv"
    for row in load(source):
        add("KTH_joint_region4_time4_tuning_"+row["stage"], row, row["scheme"], source, "region4_time4", "240/80/80 Cycle pools; test single-Cycle 10 draws",
            dimension=int(row["output_dimension"]), train="train_mean_accuracy",
            validation="validation_accuracy_mean", validation_f1="validation_macro_f1_mean",
            test="test_accuracy_mean", test_std="test_accuracy_std", test_f1="test_macro_f1_mean")
    assert len(ROWS) == 27
    for row in ROWS:
        for key in ["train_accuracy", "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1"]:
            assert row[key] is not None and 0 <= row[key] <= 1
    OUT.mkdir(exist_ok=True)
    with (OUT/"experiment_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(ROWS)
    record = dict(source_csv_sha256=SOURCES, row_count=len(ROWS), rows=ROWS,
        no_training_or_test_re_evaluation=True,
        interpretation="Scores from different model definitions and Cycle protocols are NOT one controlled experiment. I_ideal is software only.")
    (OUT/"experiment_summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Built {len(ROWS)} report rows from {len(SOURCES)} completed source CSVs; no experiments rerun.")


if __name__ == "__main__":
    main()
