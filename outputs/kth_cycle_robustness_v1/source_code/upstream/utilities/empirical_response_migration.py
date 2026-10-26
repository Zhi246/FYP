"""Build an empirical response bank from the 256-sheet hardware workbook."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from .calibration_migration import _read_sheet_rows, load_differential_levels
from .xlsx_conversion import worksheet_paths


SUMMARY_HEADER = (
    "level_index",
    "differential_level",
    "group_index",
    "sample_count",
    "mean",
    "std_dev",
    "minimum",
    "q01",
    "q05",
    "q25",
    "median",
    "q75",
    "q95",
    "q99",
    "maximum",
)


def migrate_empirical_response_workbook(
    workbook_path: Path,
    differential_levels_path: Path,
    response_bank_path: Path,
    summary_path: Path,
    metadata_path: Path,
    *,
    expected_groups: int = 16,
    expected_cycles: int = 400,
) -> tuple[int, int, int]:
    source = workbook_path.expanduser().resolve()
    level_strings = load_differential_levels(differential_levels_path)
    levels = np.asarray([float(value) for value in level_strings], dtype=np.float64)
    samples = np.empty(
        (len(levels), expected_cycles, expected_groups),
        dtype=np.float32,
    )
    expected_header = (
        "Cycle",
        *(f"Group_{index}" for index in range(1, expected_groups + 1)),
    )

    with zipfile.ZipFile(source) as workbook:
        sheets = worksheet_paths(workbook)
        expected_names = [str(index) for index in range(len(levels))]
        actual_names = [name for name, _path in sheets]
        if actual_names != expected_names:
            raise ValueError(
                f"Expected sheets {expected_names[0]}..{expected_names[-1]}; "
                f"found {actual_names[:3]}..{actual_names[-3:]}."
            )
        for level_index, (sheet_name, sheet_path) in enumerate(sheets):
            rows = _read_sheet_rows(workbook, sheet_path)
            if not rows or tuple(rows[0]) != expected_header:
                raise ValueError(
                    f"Sheet {sheet_name} must use columns {list(expected_header)}."
                )
            data_rows = rows[1:]
            if len(data_rows) != expected_cycles:
                raise ValueError(
                    f"Sheet {sheet_name} contains {len(data_rows)} cycles; "
                    f"expected {expected_cycles}."
                )
            for cycle_index, row in enumerate(data_rows):
                if len(row) != expected_groups + 1:
                    raise ValueError(
                        f"Sheet {sheet_name}, cycle {cycle_index + 1} has "
                        f"{len(row)} columns; expected {expected_groups + 1}."
                    )
                if int(row[0]) != cycle_index + 1:
                    raise ValueError(
                        f"Sheet {sheet_name} has unexpected Cycle value {row[0]} "
                        f"at row {cycle_index + 2}."
                    )
                samples[level_index, cycle_index] = [
                    float(value) for value in row[1:]
                ]

    if not np.isfinite(samples).all():
        raise ValueError("Hardware response workbook contains NaN or infinity.")

    response_bank_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_bank = response_bank_path.with_suffix(response_bank_path.suffix + ".tmp")
    with temporary_bank.open("wb") as handle:
        group_magnitude_references = np.quantile(
            np.abs(samples),
            0.95,
            axis=(0, 1),
        )
        np.savez_compressed(
            handle,
            differential_levels=levels,
            response_samples=samples,
            group_magnitude_references=group_magnitude_references,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary_bank.replace(response_bank_path)

    quantile_probabilities = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    quantiles = np.quantile(samples, quantile_probabilities, axis=1)
    means = samples.mean(axis=1, dtype=np.float64)
    standard_deviations = samples.std(axis=1, dtype=np.float64)
    minimums = samples.min(axis=1)
    maximums = samples.max(axis=1)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUMMARY_HEADER)
        for level_index, differential_level in enumerate(levels):
            for group_index in range(expected_groups):
                writer.writerow(
                    (
                        level_index,
                        differential_level,
                        group_index + 1,
                        expected_cycles,
                        means[level_index, group_index],
                        standard_deviations[level_index, group_index],
                        minimums[level_index, group_index],
                        quantiles[0, level_index, group_index],
                        quantiles[1, level_index, group_index],
                        quantiles[2, level_index, group_index],
                        quantiles[3, level_index, group_index],
                        quantiles[4, level_index, group_index],
                        quantiles[5, level_index, group_index],
                        quantiles[6, level_index, group_index],
                        maximums[level_index, group_index],
                    )
                )
        handle.flush()
        os.fsync(handle.fileno())
    temporary_summary.replace(summary_path)

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "format_version": 1,
        "source_workbook": source.name,
        "source_sha256": source_hash,
        "level_count": len(levels),
        "cycle_count": expected_cycles,
        "group_count": expected_groups,
        "response_layout": "level,cycle,group",
        "group_magnitude_reference": "95th percentile of abs(response) per Group",
        "sampling_rule": (
            "one shared physical cycle per saved digit; exact 16-Group row "
            "lookup for each quantized level, followed by independent "
            "magnitude-adaptive multiplicative jitter"
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temporary_metadata.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_metadata.replace(metadata_path)
    return len(levels), expected_cycles, expected_groups
