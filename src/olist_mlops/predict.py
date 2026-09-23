"""Prediction service: raw order -> engineered features -> label + probability.

Wraps the frozen preprocessor/model (olist_mlops.artifacts) and the
notebook-parity feature engineering (olist_mlops.features). Never fits
anything; only .transform() and .predict_proba().
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from olist_mlops.artifacts import ModelArtifacts
from olist_mlops.features import build_prediction_features


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float
    model_version: str


class PredictionService:
    def __init__(self, artifacts: ModelArtifacts, config: dict):
        self._artifacts = artifacts
        self._config = config
        self._contract = config["feature_contract"]
        self._model_cfg = config["model"]

        expected = self._model_cfg["expected_feature_count"]
        actual = len(artifacts.feature_list)
        if actual != expected:
            raise ValueError(
                f"Loaded feature list has {actual} features, "
                f"config expects {expected} (model.expected_feature_count)"
            )

    @property
    def model_version(self) -> str:
        return self._artifacts.model_version

    def predict_batch(self, frame: pd.DataFrame) -> list[Prediction]:
        raw_features = build_prediction_features(frame, self._contract)
        transformed = self._artifacts.preprocessor.transform(raw_features)

        X = pd.DataFrame(
            transformed,
            columns=self._artifacts.feature_list,
            index=frame.index,
        ).astype("float32")

        probabilities = self._artifacts.model.predict_proba(X)[:, 1]
        threshold = self._model_cfg["decision_threshold"]
        positive_label = self._model_cfg["positive_label"]
        negative_label = self._model_cfg["negative_label"]

        return [
            Prediction(
                label=positive_label if p >= threshold else negative_label,
                probability=float(p),
                model_version=self._artifacts.model_version,
            )
            for p in probabilities
        ]

    def predict_one(self, order: dict) -> Prediction:
        frame = pd.DataFrame([order])
        return self.predict_batch(frame)[0]
