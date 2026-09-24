"""Pydantic request/response models for the API.

The single-order request model is built dynamically from
config.feature_contract, rather than hand-listing all 40 raw predictor
fields twice — one mismatch between config and a hand-written schema
would silently reject or corrupt valid requests.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, create_model


class _ForbidExtraBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_order_request_model(contract: dict) -> type[BaseModel]:
    fields: dict[str, tuple] = {}

    for column in contract["numeric_features"]:
        fields[column] = (float | None, None)
    for column in contract["categorical_features"]:
        fields[column] = (str | None, None)
    for column in contract["timestamp_features"]:
        fields[column] = (datetime | None, None)

    return create_model(  # type: ignore[call-overload,no-any-return]
        "OrderRequest",
        __base__=_ForbidExtraBase,
        **fields,
    )


class PredictionResponse(BaseModel):
    label: str
    probability: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    model_version: str
    artifact_source: str
    decision_threshold: float
    expected_feature_count: int
