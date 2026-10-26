"""Empirical simulation backed by repeated physical-hardware measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmpiricalHardwareResponseBank:
    """Sample complete 16-channel responses from aligned hardware cycles."""

    levels: np.ndarray
    response_samples: np.ndarray
    group_magnitude_references: np.ndarray

    @property
    def group_count(self) -> int:
        return int(self.response_samples.shape[2])

    @property
    def cycle_count(self) -> int:
        return int(self.response_samples.shape[1])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_groups: int,
        expected_levels: np.ndarray,
    ) -> "EmpiricalHardwareResponseBank":
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"Empirical hardware-response bank does not exist: {source}"
            )
        with np.load(source, allow_pickle=False) as archive:
            required_arrays = {
                "differential_levels",
                "response_samples",
                "group_magnitude_references",
            }
            missing_arrays = required_arrays - set(archive.files)
            if missing_arrays:
                raise ValueError(
                    f"{source} is missing arrays: {', '.join(sorted(missing_arrays))}."
                )
            levels = np.asarray(archive["differential_levels"], dtype=np.float64)
            response_samples = np.asarray(
                archive["response_samples"],
                dtype=np.float32,
            )
            group_magnitude_references = np.asarray(
                archive["group_magnitude_references"],
                dtype=np.float64,
            )

        expected_levels_array = np.asarray(expected_levels, dtype=np.float64)
        if levels.shape != expected_levels_array.shape or not np.allclose(
            levels,
            expected_levels_array,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                f"{source} differential levels do not match differential_levels.csv."
            )
        expected_shape_prefix = (len(expected_levels_array),)
        if (
            response_samples.ndim != 3
            or response_samples.shape[0:1] != expected_shape_prefix
            or response_samples.shape[2] != expected_groups
        ):
            raise ValueError(
                f"{source} response_samples has shape {response_samples.shape}; "
                f"expected ({len(expected_levels_array)}, cycles, {expected_groups})."
            )
        if response_samples.shape[1] < 2:
            raise ValueError("Empirical response bank must contain at least two cycles.")
        if not np.isfinite(response_samples).all():
            raise ValueError("Empirical response bank contains NaN or infinity.")
        if (
            group_magnitude_references.shape != (expected_groups,)
            or not np.isfinite(group_magnitude_references).all()
            or np.any(group_magnitude_references <= 0)
        ):
            raise ValueError(
                "Empirical response bank must contain one positive finite "
                "magnitude reference per Group."
            )
        return cls(
            levels=expected_levels_array,
            response_samples=response_samples,
            group_magnitude_references=group_magnitude_references,
        )

    def simulate(
        self,
        differential_features: np.ndarray,
        *,
        sampling_mode: str,
        random_generator: np.random.Generator,
        minimum_noise_rate: float = 0.01,
        maximum_noise_rate: float = 0.05,
    ) -> np.ndarray:
        features = np.asarray(differential_features, dtype=np.float64)
        features = np.atleast_2d(features)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError(
                "Differential features must be a finite two-dimensional matrix."
            )

        level_indices = self._resolve_level_indices(features)
        if sampling_mode == "mean":
            response_means = self.response_samples.mean(axis=1, dtype=np.float64)
            simulated = response_means[level_indices]
        elif sampling_mode == "empirical":
            # One saved digit uses one physical cycle across all 16 dimensions.
            # This preserves cycle drift and the measured correlation of 16 Groups.
            cycle_indices = random_generator.integers(
                0,
                self.cycle_count,
                size=(len(features), 1),
            )
            simulated = self.response_samples[level_indices, cycle_indices]
            simulated = self._apply_magnitude_adaptive_noise(
                simulated,
                random_generator=random_generator,
                minimum_noise_rate=minimum_noise_rate,
                maximum_noise_rate=maximum_noise_rate,
            )
        else:
            raise ValueError("sampling_mode must be 'empirical' or 'mean'.")

        # Dimension-major layout: [d1_group1..16, d2_group1..16, ...].
        return np.asarray(simulated, dtype=np.float32).reshape(len(features), -1)

    def _apply_magnitude_adaptive_noise(
        self,
        responses: np.ndarray,
        *,
        random_generator: np.random.Generator,
        minimum_noise_rate: float,
        maximum_noise_rate: float,
    ) -> np.ndarray:
        if minimum_noise_rate < 0:
            raise ValueError("minimum_noise_rate must be non-negative.")
        if maximum_noise_rate < minimum_noise_rate:
            raise ValueError(
                "maximum_noise_rate must be at least minimum_noise_rate."
            )
        magnitude_ratios = np.clip(
            np.abs(responses) / self.group_magnitude_references,
            0.0,
            1.0,
        )
        noise_rates = maximum_noise_rate - (
            (maximum_noise_rate - minimum_noise_rate) * magnitude_ratios
        )
        relative_noise = random_generator.uniform(
            -noise_rates,
            noise_rates,
        )
        return responses * (1.0 + relative_noise)

    def _resolve_level_indices(self, features: np.ndarray) -> np.ndarray:
        upper_indices = np.searchsorted(self.levels, features, side="left")
        upper_indices = np.clip(upper_indices, 0, len(self.levels) - 1)
        lower_indices = np.maximum(upper_indices - 1, 0)
        lower_distance = np.abs(features - self.levels[lower_indices])
        upper_distance = np.abs(self.levels[upper_indices] - features)
        level_indices = np.where(
            upper_distance <= lower_distance,
            upper_indices,
            lower_indices,
        )
        selected_levels = self.levels[level_indices]
        if not np.allclose(selected_levels, features, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "Differential features must already be quantized to "
                "differential_levels.csv."
            )
        return level_indices
