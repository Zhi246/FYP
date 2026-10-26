"""Train-fitted full-dimensional joint Gaussian transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def benchmark_kmeans(
    inputs: np.ndarray,
    centers: int,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    """Reproduce the PC benchmark's deterministic Torch K-means implementation."""
    values = torch.as_tensor(inputs, dtype=torch.float32, device="cpu")
    if values.ndim != 2:
        raise ValueError("K-means inputs must be a two-dimensional matrix.")
    if centers > len(values):
        raise ValueError("The number of joint centers cannot exceed the training rows.")

    generator = torch.Generator(device="cpu").manual_seed(random_state)
    selected = torch.randperm(len(values), generator=generator)[:centers]
    current = values[selected].clone()
    for _ in range(max_iter):
        labels = torch.cdist(values, current).argmin(dim=1)
        counts = torch.bincount(labels, minlength=centers)
        updated = torch.zeros_like(current).index_add_(0, labels, values)
        updated = updated / counts.clamp_min(1).unsqueeze(1)
        updated[counts == 0] = current[counts == 0]
        if torch.allclose(updated, current, atol=1.0e-5, rtol=0):
            current = updated
            break
        current = updated
    return current.cpu().numpy().astype(np.float64, copy=False)


@dataclass
class FullJointGaussianTransformer:
    dimensions: int
    basis_per_dimension: int
    output_features: int
    joint_sigma: float
    calibration_alpha: float
    factor_lower: float
    factor_upper: float
    epsilon: float
    random_state: int
    kmeans_max_iter: int

    channel_mean_: np.ndarray | None = None
    channel_std_: np.ndarray | None = None
    centers_: np.ndarray | None = None
    calibration_weights_: np.ndarray | None = None
    calibration_r2_: np.ndarray | None = None

    @property
    def hardware_features(self) -> int:
        return self.dimensions * self.basis_per_dimension

    def _reshape_hardware(self, hardware: np.ndarray) -> np.ndarray:
        values = np.asarray(hardware, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.hardware_features:
            raise ValueError(
                f"Expected hardware matrix [N,{self.hardware_features}], got {values.shape}."
            )
        return values.reshape(-1, self.dimensions, self.basis_per_dimension)

    def _validate_reference(self, reference_inputs: np.ndarray, rows: int) -> np.ndarray:
        values = np.asarray(reference_inputs, dtype=np.float64)
        if values.shape != (rows, self.dimensions):
            raise ValueError(
                f"Expected paired reference inputs [{rows},{self.dimensions}], got {values.shape}."
            )
        return values

    def fit(
        self,
        hardware: np.ndarray,
        reference_inputs: np.ndarray,
        *,
        preset_centers: np.ndarray | None = None,
    ) -> "FullJointGaussianTransformer":
        grouped = self._reshape_hardware(hardware)
        reference = self._validate_reference(reference_inputs, len(grouped))
        self.channel_mean_ = grouped.mean(axis=0)
        self.channel_std_ = grouped.std(axis=0)
        self.channel_std_ = np.maximum(self.channel_std_, self.epsilon)
        standardized = (grouped - self.channel_mean_) / self.channel_std_

        if preset_centers is None:
            self.centers_ = benchmark_kmeans(
                reference,
                centers=self.output_features,
                random_state=self.random_state,
                max_iter=self.kmeans_max_iter,
            )
        else:
            centers = np.asarray(preset_centers, dtype=np.float64)
            expected = (self.output_features, self.dimensions)
            if centers.shape != expected:
                raise ValueError(f"Expected preset centers {expected}, got {centers.shape}.")
            self.centers_ = centers.copy()

        weights = np.empty(
            (self.dimensions, self.basis_per_dimension + 1, self.output_features),
            dtype=np.float64,
        )
        fit_r2 = np.empty((self.dimensions, self.output_features), dtype=np.float64)
        regularizer = np.eye(self.basis_per_dimension + 1, dtype=np.float64)
        regularizer *= self.calibration_alpha
        regularizer[0, 0] = 0.0
        for dimension in range(self.dimensions):
            design = np.column_stack(
                (np.ones(len(grouped), dtype=np.float64), standardized[:, dimension])
            )
            targets = self._ideal_factors(reference[:, dimension], dimension)
            solved = np.linalg.solve(
                design.T @ design + regularizer,
                design.T @ targets,
            )
            weights[dimension] = solved
            predictions = design @ solved
            residual = np.square(targets - predictions).sum(axis=0)
            centered = targets - targets.mean(axis=0, keepdims=True)
            total = np.square(centered).sum(axis=0)
            fit_r2[dimension] = 1.0 - np.divide(
                residual,
                total,
                out=np.zeros_like(residual),
                where=total > self.epsilon,
            )
        self.calibration_weights_ = weights
        self.calibration_r2_ = fit_r2
        return self

    def _check_fitted(self) -> None:
        required = (
            self.channel_mean_,
            self.channel_std_,
            self.centers_,
            self.calibration_weights_,
        )
        if any(value is None for value in required):
            raise RuntimeError("FullJointGaussianTransformer must be fitted before transformation.")

    def _ideal_factors(self, coordinate: np.ndarray, dimension: int) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Joint centers are not available.")
        difference = coordinate[:, None] - self.centers_[None, :, dimension]
        return np.exp(-np.square(difference) / (2.0 * self.joint_sigma**2))

    def transform(self, hardware: np.ndarray) -> np.ndarray:
        """Create 256 full-dimensional joint features using hardware outputs only."""
        self._check_fitted()
        grouped = self._reshape_hardware(hardware)
        standardized = (grouped - self.channel_mean_) / self.channel_std_
        log_joint = np.zeros((len(grouped), self.output_features), dtype=np.float64)
        for dimension in range(self.dimensions):
            design = np.column_stack(
                (np.ones(len(grouped), dtype=np.float64), standardized[:, dimension])
            )
            factors = np.clip(
                design @ self.calibration_weights_[dimension],
                self.factor_lower,
                self.factor_upper,
            )
            log_joint += np.log(factors)
        return np.exp(log_joint).astype(np.float32)

    def ideal_pc_transform(self, reference_inputs: np.ndarray) -> np.ndarray:
        """Direct PC Gaussian for an architecture-matched comparison only."""
        self._check_fitted()
        reference = self._validate_reference(reference_inputs, len(reference_inputs))
        squared_distance = np.square(reference[:, None, :] - self.centers_[None, :, :]).sum(
            axis=2
        )
        return np.exp(-squared_distance / (2.0 * self.joint_sigma**2)).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "type": "full_joint_gaussian",
            "hardware_gaussian_source": True,
            "uses_reference_at_inference": False,
            "dimensions": self.dimensions,
            "basis_per_dimension": self.basis_per_dimension,
            "output_features": self.output_features,
            "layout": "dimension_major",
            "joint_sigma": self.joint_sigma,
            "calibration_alpha": self.calibration_alpha,
            "factor_lower": self.factor_lower,
            "factor_upper": self.factor_upper,
            "epsilon": self.epsilon,
            "random_state": self.random_state,
            "kmeans_max_iter": self.kmeans_max_iter,
            "channel_mean": self.channel_mean_,
            "channel_std": self.channel_std_,
            "centers": self.centers_,
            "calibration_weights": self.calibration_weights_,
            "calibration_r2": self.calibration_r2_,
            "fit_split": "train",
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FullJointGaussianTransformer":
        if state.get("type") != "full_joint_gaussian":
            raise ValueError("Checkpoint does not contain a full_joint_gaussian transformer.")

        def array(key: str) -> np.ndarray:
            value = state[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)

        transformer = cls(
            dimensions=int(state["dimensions"]),
            basis_per_dimension=int(state["basis_per_dimension"]),
            output_features=int(state["output_features"]),
            joint_sigma=float(state["joint_sigma"]),
            calibration_alpha=float(state["calibration_alpha"]),
            factor_lower=float(state["factor_lower"]),
            factor_upper=float(state["factor_upper"]),
            epsilon=float(state["epsilon"]),
            random_state=int(state["random_state"]),
            kmeans_max_iter=int(state["kmeans_max_iter"]),
        )
        transformer.channel_mean_ = array("channel_mean")
        transformer.channel_std_ = array("channel_std")
        transformer.centers_ = array("centers")
        transformer.calibration_weights_ = array("calibration_weights")
        transformer.calibration_r2_ = array("calibration_r2")
        transformer._check_fitted()
        return transformer


def linear_cka(first: np.ndarray, second: np.ndarray, epsilon: float = 1.0e-30) -> float:
    x = np.array(first, dtype=np.float64, copy=True)
    y = np.array(second, dtype=np.float64, copy=True)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.sqrt(np.square(xx).sum() * np.square(yy).sum())
    return float(np.square(xy).sum() / max(float(denominator), epsilon))
