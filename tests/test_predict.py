"""Model tests: the frozen model loads, predicts the right shape, and
behaves sanely on known inputs. Requires the local notebook artifacts
(see tests/conftest.py::requires_local_artifacts for why they're gitignored).
"""

from __future__ import annotations

import pandas as pd
import pytest

from olist_mlops.artifacts import load_artifacts
from olist_mlops.config import find_repo_root
from olist_mlops.predict import PredictionService
from tests.conftest import requires_local_artifacts, requires_local_test_data


@pytest.fixture(scope="module")
def artifacts(config):
    return load_artifacts(config)


@pytest.fixture(scope="module")
def service(artifacts, config):
    return PredictionService(artifacts, config)


@requires_local_artifacts
def test_artifacts_load_with_expected_feature_count(artifacts, config):
    assert len(artifacts.feature_list) == config["model"]["expected_feature_count"]
    assert artifacts.source == "local_files"


@requires_local_artifacts
def test_artifacts_loader_reports_its_source(artifacts):
    # Whichever source it picked, it must say so — the API's /model/info
    # route surfaces this so a caller can tell registry-served predictions
    # from the local-file fallback.
    assert artifacts.source in ("local_files", "mlflow_registry")


@requires_local_artifacts
@requires_local_test_data
def test_predict_batch_returns_one_prediction_per_row(service):
    test_df = pd.read_parquet(find_repo_root() / "data/processed/test.parquet").head(10)
    predictions = service.predict_batch(test_df)
    assert len(predictions) == 10
    for p in predictions:
        assert 0.0 <= p.probability <= 1.0
        assert p.label in ("late", "on_time")
        assert p.model_version == service.model_version


@requires_local_artifacts
@requires_local_test_data
def test_predict_one_matches_threshold_logic(service, config):
    test_df = pd.read_parquet(find_repo_root() / "data/processed/test.parquet").head(1)
    order = test_df.iloc[0].to_dict()
    prediction = service.predict_one(order)

    threshold = config["model"]["decision_threshold"]
    if prediction.probability >= threshold:
        assert prediction.label == config["model"]["positive_label"]
    else:
        assert prediction.label == config["model"]["negative_label"]


def test_prediction_service_rejects_wrong_feature_count(config):
    from olist_mlops.artifacts import ModelArtifacts

    bad_artifacts = ModelArtifacts(
        preprocessor=None,
        model=None,
        feature_list=["a", "b"],
        model_version="x",
        source="local_files",
    )
    with pytest.raises(ValueError, match="expected_feature_count"):
        PredictionService(bad_artifacts, config)
