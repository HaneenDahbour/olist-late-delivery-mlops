"""Integration tests for the FastAPI routes, end to end through TestClient.

Requires local artifacts (the app loads the model at startup) — see
tests/conftest.py::requires_local_artifacts.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from olist_mlops.config import find_repo_root
from tests.conftest import requires_local_artifacts, requires_local_test_data


@pytest.fixture(scope="module")
def client(config):
    from app.main import create_app

    return TestClient(create_app(config))


@pytest.fixture(scope="module")
def sample_order(config):
    test_df = pd.read_parquet(find_repo_root() / "data/processed/test.parquet")
    row = test_df.iloc[0]
    contract = config["feature_contract"]
    order = {}
    for column in contract["numeric_features"]:
        value = row[column]
        order[column] = None if pd.isna(value) else float(value)
    for column in contract["categorical_features"]:
        value = row[column]
        order[column] = None if pd.isna(value) else str(value)
    for column in contract["timestamp_features"]:
        value = row[column]
        order[column] = None if pd.isna(value) else pd.Timestamp(value).isoformat()
    return order


@requires_local_artifacts
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@requires_local_artifacts
def test_model_info(client, config):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["expected_feature_count"] == config["model"]["expected_feature_count"]
    assert body["decision_threshold"] == config["model"]["decision_threshold"]


@requires_local_artifacts
@requires_local_test_data
def test_predict_returns_label_and_probability(client, sample_order):
    response = client.post("/predict", json=sample_order)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("late", "on_time")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"]


@requires_local_artifacts
@requires_local_test_data
def test_predict_rejects_forbidden_column(client, sample_order):
    bad_order = dict(sample_order, is_late=1)
    response = client.post("/predict", json=bad_order)
    assert response.status_code == 422


@requires_local_artifacts
@requires_local_test_data
def test_predict_rejects_null_required_field(client, sample_order):
    bad_order = dict(sample_order, item_count=None)
    response = client.post("/predict", json=bad_order)
    assert response.status_code == 422


@requires_local_artifacts
@requires_local_test_data
def test_predict_batch(client, sample_order):
    response = client.post("/predict/batch", json=[sample_order, sample_order])
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0] == predictions[1]


@requires_local_artifacts
def test_predict_batch_rejects_oversized_batch(client, sample_order, config):
    max_batch_size = config["api"]["max_batch_size"]
    response = client.post("/predict/batch", json=[sample_order] * (max_batch_size + 1))
    assert response.status_code == 422


@requires_local_artifacts
def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"python_gc_objects_collected_total" in response.content
