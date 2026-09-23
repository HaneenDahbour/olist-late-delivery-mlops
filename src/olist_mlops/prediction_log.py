"""Persist every prediction request: input, output, latency, model version.

Backend is chosen by config.monitoring.prediction_log_backend:
- "file": append a JSON line per prediction under config.paths.prediction_log_dir.
  Always available, zero external dependencies.
- "postgres": also insert a row into the `prediction_log` table. Requires
  the docker-compose postgres service; falls back to file-only (with a
  warning) if the database is unreachable, so the API never fails a
  request because logging couldn't reach the DB.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from olist_mlops.config import find_repo_root

logger = logging.getLogger(__name__)

_ENGINE = None  # lazily created SQLAlchemy engine, reused across calls


@dataclass(frozen=True)
class PredictionLogEntry:
    request_id: str
    timestamp: str
    input: dict
    label: str
    probability: float
    model_version: str
    latency_ms: float


def _build_entry(
    order: dict, label: str, probability: float, model_version: str, latency_ms: float
) -> PredictionLogEntry:
    return PredictionLogEntry(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
        input=order,
        label=label,
        probability=probability,
        model_version=model_version,
        latency_ms=latency_ms,
    )


def _append_to_file(entry: PredictionLogEntry, config: dict) -> None:
    log_dir = find_repo_root() / config["paths"]["prediction_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def _get_engine(config: dict):
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    from sqlalchemy import create_engine

    db = config["database"]
    password = os.environ.get(db["password_env"], "")
    url = f"postgresql+psycopg://{db['user']}:{password}@{db['host']}:{db['port']}/{db['name']}"
    # psycopg has no default connect timeout: an unreachable (as opposed to
    # actively refused) host would otherwise hang this call indefinitely.
    _ENGINE = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
    return _ENGINE


def ensure_prediction_log_table(config: dict) -> None:
    from sqlalchemy import text

    engine = _get_engine(config)
    with engine.begin() as conn:
        conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    request_id UUID PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    input JSONB NOT NULL,
                    label TEXT NOT NULL,
                    probability DOUBLE PRECISION NOT NULL,
                    model_version TEXT NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL
                )
                """))


def _insert_into_postgres(entry: PredictionLogEntry, config: dict) -> None:
    from sqlalchemy import text

    engine = _get_engine(config)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO prediction_log
                    (request_id, ts, input, label, probability, model_version, latency_ms)
                VALUES
                    (:request_id, :ts, :input, :label, :probability, :model_version, :latency_ms)
                """),
            {
                "request_id": entry.request_id,
                "ts": entry.timestamp,
                "input": json.dumps(entry.input),
                "label": entry.label,
                "probability": entry.probability,
                "model_version": entry.model_version,
                "latency_ms": entry.latency_ms,
            },
        )


def log_prediction(
    order: dict,
    label: str,
    probability: float,
    model_version: str,
    latency_ms: float,
    config: dict,
) -> PredictionLogEntry:
    entry = _build_entry(order, label, probability, model_version, latency_ms)
    _append_to_file(entry, config)

    if config["monitoring"]["prediction_log_backend"] == "postgres":
        try:
            _insert_into_postgres(entry, config)
        except Exception as exc:  # noqa: BLE001 - logging must never break a prediction
            logger.warning(
                "Postgres prediction log insert failed (%s); file log still written", exc
            )

    return entry
