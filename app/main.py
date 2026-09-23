"""Olist late-delivery prediction API.

Routes: health check, model info, single predict, batch predict, and a
Prometheus /metrics endpoint. Every predict call is validated with
Great Expectations before it reaches the model, logged (input, output,
latency, model version), and timed for the request-latency metric.
"""

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from olist_mlops.artifacts import load_artifacts
from olist_mlops.config import load_config
from olist_mlops.logging_setup import configure_logging
from olist_mlops.predict import PredictionService
from olist_mlops.prediction_log import log_prediction
from olist_mlops.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    build_order_request_model,
)
from olist_mlops.validation import ValidationResult, validate_orders

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["route", "status"])
REQUEST_LATENCY = Histogram(
    "predict_request_latency_seconds", "Prediction request latency", ["route"]
)


def create_app(config: dict | None = None) -> FastAPI:
    config = config or load_config()
    configure_logging(config)

    artifacts = load_artifacts(config)
    service = PredictionService(artifacts, config)
    OrderRequest = build_order_request_model(config["feature_contract"])

    app = FastAPI(
        title=config["api"]["title"],
        version=config["api"]["version"],
    )
    app.state.config = config
    app.state.service = service

    def _validate_or_raise(frame: pd.DataFrame) -> None:
        result: ValidationResult = validate_orders(frame, config["feature_contract"])
        if not result.success:
            raise HTTPException(status_code=422, detail={"errors": result.errors})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        model_cfg = config["model"]
        return ModelInfoResponse(
            model_name=model_cfg["name"],
            algorithm=model_cfg["algorithm"],
            model_version=service.model_version,
            artifact_source=artifacts.source,
            decision_threshold=model_cfg["decision_threshold"],
            expected_feature_count=model_cfg["expected_feature_count"],
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(order: OrderRequest) -> PredictionResponse:  # type: ignore[valid-type]
        start = time.perf_counter()
        route = "/predict"
        try:
            order_dict = order.model_dump(mode="json")
            frame = pd.DataFrame([order_dict])
            _validate_or_raise(frame)

            prediction = service.predict_one(order_dict)
            latency_ms = (time.perf_counter() - start) * 1000

            log_prediction(
                order_dict,
                prediction.label,
                prediction.probability,
                prediction.model_version,
                latency_ms,
                config,
            )
            REQUEST_COUNT.labels(route=route, status="success").inc()
            return PredictionResponse(
                label=prediction.label,
                probability=prediction.probability,
                model_version=prediction.model_version,
            )
        except HTTPException:
            REQUEST_COUNT.labels(route=route, status="rejected").inc()
            raise
        except Exception:
            REQUEST_COUNT.labels(route=route, status="error").inc()
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from None
        finally:
            REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - start)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(orders: list[OrderRequest]) -> BatchPredictionResponse:  # type: ignore[valid-type]
        start = time.perf_counter()
        route = "/predict/batch"
        max_batch_size = config["api"]["max_batch_size"]
        if len(orders) > max_batch_size:
            REQUEST_COUNT.labels(route=route, status="rejected").inc()
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(orders)} exceeds max_batch_size {max_batch_size}",
            )
        try:
            order_dicts = [o.model_dump(mode="json") for o in orders]
            frame = pd.DataFrame(order_dicts)
            _validate_or_raise(frame)

            predictions = service.predict_batch(frame)
            latency_ms = (time.perf_counter() - start) * 1000

            for order_dict, prediction in zip(order_dicts, predictions, strict=True):
                log_prediction(
                    order_dict,
                    prediction.label,
                    prediction.probability,
                    prediction.model_version,
                    latency_ms / len(order_dicts),
                    config,
                )

            REQUEST_COUNT.labels(route=route, status="success").inc()
            return BatchPredictionResponse(
                predictions=[
                    PredictionResponse(
                        label=p.label, probability=p.probability, model_version=p.model_version
                    )
                    for p in predictions
                ]
            )
        except HTTPException:
            REQUEST_COUNT.labels(route=route, status="rejected").inc()
            raise
        except Exception:
            REQUEST_COUNT.labels(route=route, status="error").inc()
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail="Batch prediction failed") from None
        finally:
            REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - start)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
