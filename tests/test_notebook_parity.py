"""Prove src/olist_mlops reproduces notebooks 05/06 exactly on the same input.

Two things must never silently drift:
1. config/config.yaml's feature_contract vs. notebook 05's own
   BASE_NUMERIC_FEATURES / BASE_CATEGORICAL_FEATURES / TIME_SOURCE_COLUMNS /
   FORBIDDEN_COLUMNS constants (referenced from config.yaml's own comment).
2. The service's prediction pipeline vs. notebook 06's frozen test-set
   metrics, computed once and saved to artifacts/notebook6_results.json
   when the model was frozen (before the test set was ever reopened).

Both require local artifacts/data that are intentionally gitignored
(Task 3: no fitted objects or raw data committed to Git) and so are
skipped — with the exact reason — when running in CI or a fresh clone
that hasn't run the notebooks yet.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from olist_mlops.config import find_repo_root
from olist_mlops.features import build_prediction_features
from tests.conftest import requires_local_artifacts, requires_local_test_data

# Transcribed verbatim from notebooks/05_feature_engineering.ipynb, cell 9.
# This is the notebook's OWN source of truth — config.yaml's feature_contract
# is checked against this, not the other way around.
NOTEBOOK_BASE_NUMERIC_FEATURES = [
    "item_count",
    "unique_product_count",
    "unique_seller_count",
    "item_price_total",
    "item_price_mean",
    "item_price_max",
    "freight_value_total",
    "freight_value_mean",
    "unique_product_category_count",
    "product_name_length_mean",
    "product_description_length_mean",
    "product_photos_qty_mean",
    "product_weight_g_total",
    "product_weight_g_mean",
    "product_length_cm_mean",
    "product_height_cm_mean",
    "product_width_cm_mean",
    "primary_category_item_count",
    "unique_seller_city_count",
    "unique_seller_state_count",
    "primary_seller_item_count",
    "payment_record_count",
    "unique_payment_type_count",
    "payment_value_total",
    "payment_value_mean",
    "payment_installments_max",
    "payment_installments_mean",
    "primary_payment_type_value",
    "primary_payment_type_record_count",
    "customer_lat_median",
    "customer_lng_median",
    "customer_geolocation_record_count",
    "primary_seller_lat_median",
    "primary_seller_lng_median",
    "primary_seller_geolocation_record_count",
]

NOTEBOOK_BASE_CATEGORICAL_FEATURES = [
    "customer_record_found",
    "customer_state",
    "primary_seller_state",
    "primary_product_category",
    "primary_payment_type",
]

NOTEBOOK_TIME_SOURCE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "shipping_limit_date_min",
    "shipping_limit_date_max",
]

NOTEBOOK_FORBIDDEN_COLUMNS = {
    "order_id",
    "customer_id",
    "customer_unique_id",
    "primary_seller_id",
    "order_status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "delivery_delay_days",
    "post_delivery_review_count",
    "post_delivery_review_score_mean",
    "post_delivery_review_score_min",
    "post_delivery_review_score_max",
    "post_delivery_review_title_count",
    "post_delivery_review_message_count",
    "post_delivery_review_message_length_mean",
    "post_delivery_review_creation_date_min",
    "post_delivery_review_answer_timestamp_max",
    "is_late",
}


def test_feature_contract_matches_notebooks(config):
    contract = config["feature_contract"]
    assert contract["numeric_features"] == NOTEBOOK_BASE_NUMERIC_FEATURES
    assert contract["categorical_features"] == NOTEBOOK_BASE_CATEGORICAL_FEATURES
    assert contract["timestamp_features"] == NOTEBOOK_TIME_SOURCE_COLUMNS
    assert set(contract["forbidden_columns"]) == NOTEBOOK_FORBIDDEN_COLUMNS


@requires_local_artifacts
@requires_local_test_data
def test_service_pipeline_matches_frozen_notebook_test_metrics(config):
    root = find_repo_root()
    preprocessor = joblib.load(root / "artifacts/notebook5_preprocessor.joblib")
    model = joblib.load(root / "artifacts/notebook6_trained_model.joblib")
    feature_list = json.loads((root / "artifacts/notebook5_feature_list.json").read_text())
    expected = json.loads((root / "artifacts/notebook6_results.json").read_text())["test"]

    test_df = pd.read_parquet(root / "data/processed/test.parquet")
    threshold = config["model"]["decision_threshold"]

    raw_features = build_prediction_features(test_df, config["feature_contract"])
    transformed = preprocessor.transform(raw_features)
    X_test = pd.DataFrame(transformed, columns=feature_list, index=test_df.index).astype("float32")
    y_test = test_df["is_late"].astype(int)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
    }

    for metric_name, value in metrics.items():
        assert value == pytest.approx(
            expected["metrics"][metric_name], abs=1e-9
        ), f"{metric_name}: service={value} notebook={expected['metrics'][metric_name]}"
