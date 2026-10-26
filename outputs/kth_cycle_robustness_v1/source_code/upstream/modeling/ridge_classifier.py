"""Ridge-regression classifier used after Gaussian feature generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RidgeLinearClassifier:
    input_features: int
    classes: list[int]
    ridge_alpha: float
    fit_intercept: bool
    regularize_intercept: bool
    device: torch.device
    dtype: torch.dtype

    def _design_matrix(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_features:
            raise ValueError(
                f"Expected a 2-D tensor with {self.input_features} features, got {tuple(features.shape)}."
            )
        if not self.fit_intercept:
            return features
        ones = torch.ones((len(features), 1), dtype=features.dtype, device=features.device)
        return torch.cat((ones, features), dim=1)

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> "RidgeLinearClassifier":
        features = features.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=torch.long)
        classes = torch.as_tensor(self.classes, dtype=torch.long, device=self.device)
        class_indices = torch.searchsorted(classes, labels)
        valid = (class_indices < len(classes)) & (classes[class_indices.clamp_max(len(classes) - 1)] == labels)
        if not bool(valid.all()):
            raise ValueError("Training labels contain values outside the configured class list.")

        design = self._design_matrix(features)
        targets = torch.nn.functional.one_hot(class_indices, num_classes=len(classes)).to(self.dtype)
        regularizer = torch.eye(design.shape[1], dtype=self.dtype, device=self.device) * self.ridge_alpha
        if self.fit_intercept and not self.regularize_intercept:
            regularizer[0, 0] = 0
        self.combined_weights_ = torch.linalg.solve(
            design.T @ design + regularizer,
            design.T @ targets,
        )
        self.classes_ = classes
        if self.fit_intercept:
            self.bias_ = self.combined_weights_[0]
            self.weight_ = self.combined_weights_[1:].T
        else:
            self.bias_ = torch.zeros(len(classes), dtype=self.dtype, device=self.device)
            self.weight_ = self.combined_weights_.T
        return self

    @torch.no_grad()
    def scores(self, features: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "combined_weights_"):
            raise RuntimeError("The classifier must be fitted before prediction.")
        features = features.to(device=self.device, dtype=self.dtype)
        return self._design_matrix(features) @ self.combined_weights_

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.classes_[self.scores(features).argmax(dim=1)]

    def add_class_score_bias(self, class_bias: torch.Tensor) -> None:
        if not hasattr(self, "combined_weights_"):
            raise RuntimeError("The classifier must be fitted before applying class bias.")
        if not self.fit_intercept:
            raise RuntimeError("Class-score bias calibration requires fit_intercept: true.")
        bias = class_bias.to(device=self.device, dtype=self.dtype)
        if bias.shape != self.bias_.shape:
            raise ValueError(f"Expected class bias shape {tuple(self.bias_.shape)}, got {tuple(bias.shape)}.")
        self.combined_weights_[0] += bias
        # bias_ is a view of the intercept row; rebind it instead of adding twice.
        self.bias_ = self.combined_weights_[0]

    def cpu_state_dict(self) -> dict[str, torch.Tensor]:
        if not hasattr(self, "combined_weights_"):
            raise RuntimeError("The classifier has not been fitted.")
        return {
            "weight": self.weight_.detach().cpu(),
            "bias": self.bias_.detach().cpu(),
            "combined_weights": self.combined_weights_.detach().cpu(),
            "classes": self.classes_.detach().cpu(),
        }
