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


# A real row from data/processed/test.parquet (order_id anonymized away —
# only the predictor columns this API accepts are kept). FastAPI's /docs
# "Try it out" prefills this as the request body; without a real example
# it would prefill every field as null, which our own validation would
# reject with 422 — so "check the automatic API docs and make sure the
# examples work" would otherwise fail on the very first click.
EXAMPLE_ORDER = {
    "item_count": 1.0,
    "unique_product_count": 1.0,
    "unique_seller_count": 1.0,
    "item_price_total": 340.99,
    "item_price_mean": 340.99,
    "item_price_max": 340.99,
    "freight_value_total": 80.47,
    "freight_value_mean": 80.47,
    "unique_product_category_count": 1.0,
    "product_name_length_mean": 43.0,
    "product_description_length_mean": 1002.0,
    "product_photos_qty_mean": 3.0,
    "product_weight_g_total": 10150.0,
    "product_weight_g_mean": 10150.0,
    "product_length_cm_mean": 89.0,
    "product_height_cm_mean": 15.0,
    "product_width_cm_mean": 40.0,
    "primary_category_item_count": 1.0,
    "unique_seller_city_count": 1.0,
    "unique_seller_state_count": 1.0,
    "primary_seller_item_count": 1.0,
    "payment_record_count": 1.0,
    "unique_payment_type_count": 1.0,
    "payment_value_total": 421.46,
    "payment_value_mean": 421.46,
    "payment_installments_max": 4.0,
    "payment_installments_mean": 4.0,
    "primary_payment_type_value": 421.46,
    "primary_payment_type_record_count": 1.0,
    "customer_lat_median": -22.561171418258436,
    "customer_lng_median": -47.447471268798,
    "customer_geolocation_record_count": 83.0,
    "primary_seller_lat_median": -26.912615530002004,
    "primary_seller_lng_median": -48.67401528393623,
    "primary_seller_geolocation_record_count": 171.0,
    "customer_record_found": "True",
    "customer_state": "SP",
    "primary_seller_state": "SC",
    "primary_product_category": "housewares",
    "primary_payment_type": "credit_card",
    "order_purchase_timestamp": "2018-06-21T08:59:32",
    "order_approved_at": "2018-06-21T09:17:53",
    "order_estimated_delivery_date": "2018-07-18T00:00:00",
    "shipping_limit_date_min": "2018-06-25T09:17:53",
    "shipping_limit_date_max": "2018-06-25T09:17:53",
}


def build_order_request_model(contract: dict) -> type[BaseModel]:
    fields: dict[str, tuple] = {}

    for column in contract["numeric_features"]:
        fields[column] = (float | None, None)
    for column in contract["categorical_features"]:
        fields[column] = (str | None, None)
    for column in contract["timestamp_features"]:
        fields[column] = (datetime | None, None)

    model = create_model(  # type: ignore[call-overload,no-any-return]
        "OrderRequest",
        __base__=_ForbidExtraBase,
        **fields,
    )
    model.model_config["json_schema_extra"] = {"example": EXAMPLE_ORDER}
    return model


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
