"""Validate incoming order data with Great Expectations before it reaches the model.

Two layers, cheapest first:
1. Forbidden-column check (pure Python) — reject any request containing
   a column from feature_contract.forbidden_columns (leakage sources).
2. A Great Expectations suite over the raw predictor columns: presence
   of required fields, numeric ranges, allowed categories, and a
   missing-rate tolerance on optional numeric fields.

Decision on failure: reject (HTTP 422 in the API layer) — a late-
delivery prediction on a value we know is out of contract is worse
than no prediction at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import great_expectations as gx
import pandas as pd

# Silence GE's tqdm "Calculating Metrics" progress bars — noisy on every
# single-row prediction request and irrelevant outside a notebook.
os.environ.setdefault("TQDM_DISABLE", "1")

BRAZIL_STATE_CODES = [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]

# Olist's known payment-type categories (source: 05_feature_engineering.ipynb
# primary_payment_type derivation); "not_defined" appears in the raw dataset
# even though it isn't present in this project's train/validation/test splits.
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]

# Structurally required: the feature-engineering step divides/subtracts on
# these, so a null here is a malformed request, not tolerable missingness.
REQUIRED_NOT_NULL_COLUMNS = [
    "item_count",
    "item_price_total",
    "freight_value_total",
    "customer_state",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
]


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    errors: list[str] = field(default_factory=list)


def check_forbidden_columns(payload_columns: list[str], contract: dict) -> list[str]:
    forbidden = set(contract["forbidden_columns"])
    return sorted(forbidden & set(payload_columns))


def _build_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="order_input_suite")

    for column in REQUIRED_NOT_NULL_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="item_count", min_value=1, max_value=100
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="item_price_total", min_value=0, max_value=None
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="freight_value_total", min_value=0, max_value=None
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="payment_installments_max", min_value=0, max_value=24, mostly=0.99
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="customer_state", value_set=BRAZIL_STATE_CODES
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="primary_seller_state", value_set=BRAZIL_STATE_CODES, mostly=0.99
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="primary_payment_type", value_set=PAYMENT_TYPES, mostly=0.99
        )
    )
    # Missing-rate tolerance: geolocation joins can legitimately fail to
    # match for a small share of orders (see notebook 4's EDA).
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_lat_median", mostly=0.8)
    )
    return suite


def validate_orders(frame: pd.DataFrame, contract: dict) -> ValidationResult:
    forbidden_present = check_forbidden_columns(list(frame.columns), contract)
    if forbidden_present:
        return ValidationResult(
            success=False,
            errors=[f"Forbidden column(s) in request: {forbidden_present}"],
        )

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("orders_source")
    asset = data_source.add_dataframe_asset("orders")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})

    suite = _build_suite()
    result = batch.validate(suite)

    if result.success:
        return ValidationResult(success=True)

    errors = [
        f"{r.expectation_config.kwargs.get('column')}: {r.expectation_config.type} failed"
        for r in result.results
        if not r.success and r.expectation_config is not None
    ]
    return ValidationResult(success=False, errors=errors)
