#!/usr/bin/env python3
"""Read-only verification of saved KTH arrays, preprocessing, classifiers and metrics."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_kth import CLASSES, METHODS, SUBJECTS, centers_from_train, score
from run_experiment import gaussian_sum
from tune_gaussian import scale_fit_and_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/kth_six_subject_split_v1"))
    args = parser.parse_args()
    out = args.output
    # This is our own trusted local artifact. Pandas string columns were saved as
    # object arrays; as with the local joblib models, do not load untrusted files.
    data = np.load(out / "features_arrays.npz", allow_pickle=True)
    samples = pd.read_csv(out / "sample_manifest.csv")
    predictions = pd.read_csv(out / "sample_predictions.csv")
    metrics = json.loads((out / "complete_metrics.json").read_text())
    frozen = json.loads((out / "parameters_frozen_before_test.json").read_text())
    assert not frozen["test_used_for_selection"] and not frozen["refit_train_plus_validation"]
    assert data["raw"].shape == (len(samples), 16)
    assert np.isfinite(data["raw"]).all()
    assert samples.sample_id.tolist() == data["sample_ids"].tolist()
    assert samples["class"].tolist() == data["labels"].tolist()
    assert samples.split.tolist() == data["splits"].tolist()
    for split in SUBJECTS:
        assert set(samples.loc[samples.split == split, "subject"]) == set(SUBJECTS[split])
    train = samples.split.to_numpy() == "train"
    fit, _, _ = scale_fit_and_eval(data["raw"][train], data["raw"][train])
    np.testing.assert_array_equal(fit, data["scaled"][train])
    for split in ["validation", "test"]:
        mask = samples.split.to_numpy() == split
        _, scaled, _ = scale_fit_and_eval(data["raw"][train], data["raw"][mask])
        np.testing.assert_array_equal(scaled, data["scaled"][mask])
    for method in METHODS:
        bundle = joblib.load(out / "models" / (method + ".joblib"))
        row = next(r for r in metrics if r["method"] == method)
        if method == "A":
            expected = data["scaled"]
        else:
            centers = centers_from_train(bundle["configuration"]["center_mode"], fit)
            np.testing.assert_array_equal(centers, bundle["centers"])
            expected = gaussian_sum(data["scaled"], bundle["sigma"], centers)
        np.testing.assert_array_equal(expected, data[method])
        model = bundle["classifier"]
        scaler = model.named_steps["standard_scaler"]
        assert scaler.n_samples_seen_ == train.sum()
        np.testing.assert_allclose(scaler.mean_, expected[train].mean(axis=0), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scaler.var_, expected[train].var(axis=0), atol=1e-12, rtol=1e-12)
        for split in SUBJECTS:
            mask = samples.split.to_numpy() == split
            saved = predictions[(predictions.method == method) & (predictions.split == split)]
            assert saved.sample_id.tolist() == samples.loc[mask, "sample_id"].tolist()
            assert saved.true_class.tolist() == samples.loc[mask, "class"].tolist()
            predicted = model.predict(expected[mask])
            np.testing.assert_array_equal(predicted, saved.predicted_class.to_numpy())
            np.testing.assert_allclose(model.predict_proba(expected[mask]),
                                       saved[[f"prob_{c}" for c in model.classes_]].to_numpy(), atol=1e-12, rtol=1e-10)
            for metric, value in score(saved.true_class, predicted).items():
                assert np.isclose(value, row[f"{split}_{metric}"])
        cm = pd.read_csv(out / f"confusion_{method}.csv", index_col=0)
        assert cm.index.tolist() == CLASSES and cm.columns.tolist() == CLASSES
        assert cm.to_numpy().sum() == (samples.split == "test").sum()
        assert np.isclose(np.trace(cm) / cm.to_numpy().sum(), row["test_accuracy"])
    summary = json.loads((out / "order_independence_checks.json").read_text())
    assert all(s["passed"] and s["prediction_changes"] == 0 for s in summary.values())
    print(f"PASSED: {len(samples)} real KTH subsequences; arrays, train-only scalers/centers, saved models, predictions and metrics agree. No model was fitted.")


if __name__ == "__main__":
    main()
