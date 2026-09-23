"""Unit tests for feature engineering — synthetic data, no artifacts needed."""

from __future__ import annotations

import math

import pandas as pd

from olist_mlops.features import build_prediction_features, haversine_km


def test_haversine_km_known_distance():
    # Sao Paulo -> Rio de Janeiro, ~360km great-circle distance.
    distance = haversine_km(
        pd.Series([-23.5505]), pd.Series([-46.6333]), pd.Series([-22.9068]), pd.Series([-43.1729])
    )
    assert 350 <= distance.iloc[0] <= 370


def test_haversine_km_zero_distance_for_identical_points():
    distance = haversine_km(
        pd.Series([10.0]), pd.Series([20.0]), pd.Series([10.0]), pd.Series([20.0])
    )
    assert math.isclose(distance.iloc[0], 0.0, abs_tol=1e-9)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_count": 2,
                "item_price_total": 100.0,
                "freight_value_total": 10.0,
                "customer_lat_median": -23.5505,
                "customer_lng_median": -46.6333,
                "primary_seller_lat_median": -22.9068,
                "primary_seller_lng_median": -43.1729,
                "customer_state": "SP",
                "primary_seller_state": "RJ",
                "order_purchase_timestamp": "2018-06-21T08:00:00",
                "order_approved_at": "2018-06-21T10:00:00",
                "order_estimated_delivery_date": "2018-07-01T00:00:00",
                "shipping_limit_date_min": "2018-06-25T00:00:00",
                "shipping_limit_date_max": "2018-06-26T00:00:00",
            }
        ]
    )


def _minimal_contract() -> dict:
    return {
        "numeric_features": [
            "item_count",
            "item_price_total",
            "freight_value_total",
            "customer_lat_median",
            "customer_lng_median",
            "primary_seller_lat_median",
            "primary_seller_lng_median",
        ],
        "categorical_features": ["customer_state", "primary_seller_state"],
    }


def test_build_prediction_features_engineered_columns():
    X = build_prediction_features(_sample_frame(), _minimal_contract())

    assert X.loc[0, "approval_delay_hours"] == 2.0
    assert X.loc[0, "purchase_month"] == 6.0
    assert X.loc[0, "purchase_day_of_week"] == 3.0  # 2018-06-21 was a Thursday
    assert X.loc[0, "purchase_hour"] == 8.0
    assert X.loc[0, "freight_to_price_ratio"] == 0.1
    assert X.loc[0, "customer_seller_same_state"] == 0.0
    assert 350 <= X.loc[0, "customer_seller_distance_km"] <= 370


def test_build_prediction_features_same_state_flag():
    frame = _sample_frame()
    frame["primary_seller_state"] = "SP"
    X = build_prediction_features(frame, _minimal_contract())
    assert X.loc[0, "customer_seller_same_state"] == 1.0


def test_build_prediction_features_missing_state_is_nan_not_false():
    frame = _sample_frame()
    frame["primary_seller_state"] = None
    X = build_prediction_features(frame, _minimal_contract())
    assert pd.isna(X.loc[0, "customer_seller_same_state"])
