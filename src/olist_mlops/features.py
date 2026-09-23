"""Prediction-time feature engineering.

This is a byte-for-byte port of ``build_prediction_features`` /
``haversine_km`` from notebooks/05_feature_engineering.ipynb. Column
lists are NOT duplicated here — they are read from the
``feature_contract`` block of config/config.yaml (the single source of
truth), so a change to the contract cannot silently drift between the
notebook, the config, and this module.

tests/test_features.py::test_matches_notebook_output reproduces the
notebook's own frozen test-set metrics numerically to prove this stays
byte-for-byte identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def build_prediction_features(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Reproduce notebooks 05/06's raw-predictor -> engineered-feature step.

    `contract` is `config["feature_contract"]` from olist_mlops.config.
    """
    numeric_features = contract["numeric_features"]
    categorical_features = contract["categorical_features"]

    X = frame[numeric_features + categorical_features].copy()

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in categorical_features:
        X[column] = X[column].astype("object")
        X.loc[X[column].isna(), column] = np.nan

    purchase = pd.to_datetime(frame["order_purchase_timestamp"], errors="coerce")
    approved = pd.to_datetime(frame["order_approved_at"], errors="coerce")
    estimated = pd.to_datetime(frame["order_estimated_delivery_date"], errors="coerce")
    shipping_min = pd.to_datetime(frame["shipping_limit_date_min"], errors="coerce")
    shipping_max = pd.to_datetime(frame["shipping_limit_date_max"], errors="coerce")

    X["approval_delay_hours"] = (approved - purchase).dt.total_seconds() / 3600
    X["estimated_delivery_window_days"] = (estimated - purchase).dt.total_seconds() / 86400
    X["shipping_limit_min_days"] = (shipping_min - purchase).dt.total_seconds() / 86400
    X["shipping_limit_max_days"] = (shipping_max - purchase).dt.total_seconds() / 86400
    X["shipping_limit_span_days"] = (shipping_max - shipping_min).dt.total_seconds() / 86400
    X["purchase_month"] = purchase.dt.month.astype("float64")
    X["purchase_day_of_week"] = purchase.dt.dayofweek.astype("float64")
    X["purchase_hour"] = purchase.dt.hour.astype("float64")

    X["customer_seller_distance_km"] = haversine_km(
        frame["customer_lat_median"],
        frame["customer_lng_median"],
        frame["primary_seller_lat_median"],
        frame["primary_seller_lng_median"],
    )

    X["freight_to_price_ratio"] = pd.to_numeric(
        frame["freight_value_total"], errors="coerce"
    ) / pd.to_numeric(frame["item_price_total"], errors="coerce").replace(0, np.nan)

    same_state = frame["customer_state"].eq(frame["primary_seller_state"])
    missing_state = frame[["customer_state", "primary_seller_state"]].isna().any(axis=1)
    X["customer_seller_same_state"] = same_state.mask(missing_state, np.nan).astype(float)

    return X
