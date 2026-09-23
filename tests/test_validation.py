"""Data tests: schema, ranges, allowed categories, missing-rate tolerance."""

from __future__ import annotations

import pandas as pd

from olist_mlops.validation import check_forbidden_columns, validate_orders


def _valid_row(**overrides) -> dict:
    row = {
        "item_count": 2.0,
        "item_price_total": 100.0,
        "freight_value_total": 15.0,
        "payment_installments_max": 3.0,
        "customer_state": "SP",
        "primary_seller_state": "RJ",
        "primary_payment_type": "credit_card",
        "customer_lat_median": -23.5,
        "order_purchase_timestamp": "2018-06-21T08:00:00",
        "order_estimated_delivery_date": "2018-07-01T00:00:00",
    }
    row.update(overrides)
    return row


def test_check_forbidden_columns_detects_leakage_field():
    result = check_forbidden_columns(
        ["item_count", "is_late"], {"forbidden_columns": ["is_late", "order_id"]}
    )
    assert result == ["is_late"]


def test_check_forbidden_columns_clean_request():
    result = check_forbidden_columns(["item_count"], {"forbidden_columns": ["is_late", "order_id"]})
    assert result == []


def test_validate_orders_accepts_well_formed_row(config):
    frame = pd.DataFrame([_valid_row()])
    result = validate_orders(frame, config["feature_contract"])
    assert result.success, result.errors


def test_validate_orders_rejects_null_required_field(config):
    frame = pd.DataFrame([_valid_row(item_count=None)])
    result = validate_orders(frame, config["feature_contract"])
    assert not result.success
    assert any("item_count" in e for e in result.errors)


def test_validate_orders_rejects_out_of_range_value(config):
    frame = pd.DataFrame([_valid_row(item_count=-5.0)])
    result = validate_orders(frame, config["feature_contract"])
    assert not result.success


def test_validate_orders_rejects_unknown_state_code(config):
    frame = pd.DataFrame([_valid_row(customer_state="ZZ")])
    result = validate_orders(frame, config["feature_contract"])
    assert not result.success


def test_validate_orders_rejects_forbidden_column(config):
    frame = pd.DataFrame([_valid_row(is_late=1)])
    result = validate_orders(frame, config["feature_contract"])
    assert not result.success
    assert "is_late" in result.errors[0]
