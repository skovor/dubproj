"""Deterministic logistic/Platt calibration; production format is JSON only."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable
from .features import FEATURE_SCHEMA_VERSION, NORMALIZATION_VERSION, FeatureRow
from .lid_features import LID_FEATURE_SCHEMA_VERSION

class CalibrationError(ValueError): pass

@dataclass(frozen=True)
class CalibrationArtifact:
    kind: str
    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    normalization: tuple[tuple[float, float], ...]
    sample_count: int
    group_count: int
    metrics: dict[str, float]
    dataset_sha256: str
    status: str = "DRAFT"
    schema: str = "platt-calibrator-v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    normalization_version: str = NORMALIZATION_VERSION

    def to_dict(self) -> dict:
        return {"schema": self.schema, "kind": self.kind, "feature_schema_version": self.feature_schema_version, "normalization_version": self.normalization_version, "features": list(self.features), "coefficients": list(self.coefficients), "intercept": self.intercept, "normalization": [{"mean": mean, "scale": scale} for mean, scale in self.normalization], "sample_count": self.sample_count, "group_count": self.group_count, "metrics": dict(self.metrics), "dataset_sha256": self.dataset_sha256, "status": self.status}


def train_calibrator(rows: Iterable[FeatureRow], *, kind: str, features: tuple[str, ...], dataset_sha256: str, epochs: int = 600, learning_rate: float = .08) -> CalibrationArtifact:
    rows = list(rows)
    if not rows: raise CalibrationError("no human-labelled rows supplied")
    if any(row.split != "calibration" for row in rows): raise CalibrationError("only the calibration split may train a calibrator")
    if len({row.label for row in rows}) < 2: raise CalibrationError("both positive and negative human labels are required")
    groups = {row.split_group for row in rows}; matrix = [[float(row.features[name]) for name in features] for row in rows]; labels = [float(row.label) for row in rows]
    means = [sum(row[i] for row in matrix) / len(matrix) for i in range(len(features))]
    scales = [max(1e-9, (sum((row[i] - means[i]) ** 2 for row in matrix) / len(matrix)) ** .5) for i in range(len(features))]
    x = [[(row[i] - means[i]) / scales[i] for i in range(len(features))] for row in matrix]; weights = [0.0] * len(features); intercept = 0.0
    for _ in range(max(1, epochs)):
        grad_w = [0.0] * len(features); grad_b = 0.0
        for vector, label in zip(x, labels):
            probability = _sigmoid(intercept + sum(a * b for a, b in zip(weights, vector))); error = probability - label
            grad_b += error
            for index, value in enumerate(vector): grad_w[index] += error * value
        count = float(len(x)); intercept -= learning_rate * grad_b / count
        weights = [value - learning_rate * grad / count for value, grad in zip(weights, grad_w)]
    probabilities = [_sigmoid(intercept + sum(weights[i] * ((value - means[i]) / scales[i]) for i, value in enumerate(row))) for row in matrix]
    brier = sum((p - label) ** 2 for p, label in zip(probabilities, labels)) / len(labels)
    if kind == "final_anchor":
        artifact_schema = "final-anchor-v1"
    elif kind == "lid":
        artifact_schema = LID_FEATURE_SCHEMA_VERSION
    elif kind == "target":
        artifact_schema = FEATURE_SCHEMA_VERSION
    else:
        raise CalibrationError(f"unknown calibrator kind: {kind}")
    return CalibrationArtifact(kind, features, tuple(weights), intercept, tuple(zip(means, scales)), len(rows), len(groups), {"brier_score_training": brier}, dataset_sha256, feature_schema_version=artifact_schema)


def _sigmoid(value: float) -> float:
    if value >= 0: z = math.exp(-min(value, 700.0)); return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0)); return z / (1.0 + z)


__all__ = ["CalibrationArtifact", "CalibrationError", "train_calibrator"]
