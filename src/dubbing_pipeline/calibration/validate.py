"""Validation metrics and one-shot hidden-test evaluation."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Mapping
from .features import FeatureRow
from .train import CalibrationArtifact

@dataclass(frozen=True)
class ValidationReport:
    split: str
    count: int
    brier_score: float
    expected_calibration_error: float
    false_pass_count: int
    false_fail_count: int
    predictions: tuple[dict, ...]
    run_id: str = ""

    def to_dict(self) -> dict:
        return {"split": self.split, "count": self.count, "brier_score": self.brier_score, "expected_calibration_error": self.expected_calibration_error, "false_pass_count": self.false_pass_count, "false_fail_count": self.false_fail_count, "predictions": [dict(row) for row in self.predictions], "run_id": self.run_id}


def predict_artifact(artifact: CalibrationArtifact | Mapping, features: Mapping[str, float]) -> float:
    coefficients = tuple(float(x) for x in artifact.coefficients) if isinstance(artifact, CalibrationArtifact) else tuple(float(x) for x in artifact.get("coefficients", ()))
    intercept = float(artifact.intercept) if isinstance(artifact, CalibrationArtifact) else float(artifact.get("intercept", 0.0))
    normal = artifact.normalization if isinstance(artifact, CalibrationArtifact) else tuple((float(row.get("mean", 0.0)), float(row.get("scale", 1.0))) for row in artifact.get("normalization", ()))
    names = artifact.features if isinstance(artifact, CalibrationArtifact) else tuple(artifact.get("features", ()))
    if len(names) != len(coefficients) or len(normal) != len(names): raise ValueError("incomplete calibration artifact")
    logit = intercept
    for index, name in enumerate(names):
        value = float(features[name]); mean, scale = normal[index]; logit += coefficients[index] * ((value - mean) / max(abs(scale), 1e-9))
    if not math.isfinite(logit): raise ValueError("non-finite calibration logit")
    return _sigmoid(logit)


def evaluate(artifact: CalibrationArtifact | Mapping, rows: Iterable[FeatureRow], *, split: str, pass_probability: float = .8, fail_probability: float = .2, run_id: str = "") -> ValidationReport:
    rows = [row for row in rows if row.split == split]
    if not rows: raise ValueError(f"no rows for {split}")
    predictions = tuple({"clip_id": row.clip_id, "label": row.label, "probability": predict_artifact(artifact, row.features)} for row in rows)
    brier = sum((item["probability"] - item["label"]) ** 2 for item in predictions) / len(predictions)
    false_pass = sum(1 for item in predictions if item["label"] == 0 and item["probability"] >= pass_probability)
    false_fail = sum(1 for item in predictions if item["label"] == 1 and item["probability"] <= fail_probability)
    ece = _ece(predictions)
    return ValidationReport(split, len(rows), brier, ece, false_pass, false_fail, predictions, run_id)


def _ece(rows: Iterable[Mapping]) -> float:
    rows = list(rows)
    if not rows: return 0.0
    bins = []
    for low in (0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9):
        high = low + .1; group = [row for row in rows if low <= row["probability"] < high or (high == 1.0 and row["probability"] <= high)]
        if group: bins.append(len(group) / len(rows) * abs(sum(row["probability"] for row in group) / len(group) - sum(row["label"] for row in group) / len(group)))
    return sum(bins)


def _sigmoid(value: float) -> float:
    if value >= 0: z = math.exp(-min(value, 700.0)); return 1 / (1 + z)
    z = math.exp(max(value, -700.0)); return z / (1 + z)

__all__ = ["ValidationReport", "predict_artifact", "evaluate"]
