"""Pen Digits normalization and hardware differential-level quantization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DifferentialLevelQuantizer:
    levels: np.ndarray
    input_features: int
    raw_minimum: float
    raw_maximum: float

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        input_features: int,
        expected_levels: int,
        raw_minimum: float,
        raw_maximum: float,
    ) -> "DifferentialLevelQuantizer":
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Differential-level CSV does not exist: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["differential_level"]:
                raise ValueError(
                    f"{source} must contain exactly one 'differential_level' column."
                )
            try:
                levels = np.asarray(
                    [float(row["differential_level"]) for row in reader],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source} contains a non-numeric differential level.") from error

        if len(levels) != expected_levels:
            raise ValueError(
                f"{source} contains {len(levels)} levels; expected {expected_levels}."
            )
        if not np.isfinite(levels).all():
            raise ValueError(f"{source} contains NaN or infinity.")
        if not np.all(np.diff(levels) > 0):
            raise ValueError(f"{source} levels must be strictly increasing and unique.")
        if not np.isclose(levels[0], 0.0) or not np.isclose(levels[-1], 1.0):
            raise ValueError(f"{source} levels must span 0 through 1.")
        if input_features <= 0 or raw_maximum <= raw_minimum:
            raise ValueError("Invalid input feature count or raw normalization bounds.")
        return cls(
            levels=levels,
            input_features=input_features,
            raw_minimum=float(raw_minimum),
            raw_maximum=float(raw_maximum),
        )

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_features, dtype=np.float64)
        values = np.atleast_2d(values)
        if values.shape[1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} raw features, got shape {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Raw Pen Digits features contain NaN or infinity.")
        if (values < self.raw_minimum).any() or (values > self.raw_maximum).any():
            raise ValueError(
                f"Raw Pen Digits features must be within "
                f"[{self.raw_minimum}, {self.raw_maximum}]."
            )

        normalized = (values - self.raw_minimum) / (
            self.raw_maximum - self.raw_minimum
        )
        upper = np.searchsorted(self.levels, normalized, side="left")
        upper = np.clip(upper, 0, len(self.levels) - 1)
        lower = np.maximum(upper - 1, 0)
        lower_distance = np.abs(normalized - self.levels[lower])
        upper_distance = np.abs(self.levels[upper] - normalized)
        # Prefer the upper level on exact midpoint ties, matching the existing
        # train_in/test_in hardware quantization convention.
        indices = np.where(upper_distance <= lower_distance, upper, lower)
        return self.levels[indices].astype(np.float32)
