"""Load the frozen model + preprocessing artifacts for inference.

Nothing here ever calls .fit()/.fit_transform() — only .transform() and
.predict_proba() on objects already fitted in notebooks 05/06.

Load order: MLflow model registry (config.mlflow.registry_stage) first,
falling back to the local artifact files named in config.paths. The
registry is the intended production source (Task 3 item 5); local
files keep the service usable when no MLflow server is reachable
(e.g. this repo's own test suite, or a laptop demo without the full
docker-compose stack running).
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib

from olist_mlops.config import find_repo_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArtifacts:
    preprocessor: Any
    model: Any
    feature_list: list[str]
    model_version: str
    source: str  # "mlflow_registry" | "local_files"


def _load_from_local_files(config: dict) -> ModelArtifacts:
    root = find_repo_root()
    paths = config["paths"]

    preprocessor = joblib.load(root / paths["local_preprocessor"])
    model = joblib.load(root / paths["local_model"])
    feature_list = json.loads(Path(root / paths["local_feature_list"]).read_text(encoding="utf-8"))

    manifest_path = root / paths["local_manifest"]
    version = "local-unversioned"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = manifest["artifacts"]["fitted_preprocessor"]["sha256"][:12]

    return ModelArtifacts(
        preprocessor=preprocessor,
        model=model,
        feature_list=feature_list,
        model_version=version,
        source="local_files",
    )


def _is_reachable(url: str, timeout_seconds: float = 1.5) -> bool:
    """Fast TCP precheck so an unreachable MLflow server fails in ~1s.

    mlflow's own HTTP client retries with exponential backoff for several
    minutes on a refused/unreachable connection — far too slow for a
    service startup path or a test suite. This check runs first so that
    case skips straight to the local-files fallback instead.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _load_from_mlflow_registry(config: dict) -> ModelArtifacts | None:
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        return None

    mlflow_cfg = config["mlflow"]
    if not _is_reachable(mlflow_cfg["tracking_uri"]):
        logger.warning(
            "MLflow tracking server at %s unreachable; falling back to local files",
            mlflow_cfg["tracking_uri"],
        )
        return None

    try:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        client = MlflowClient()
        model_name = mlflow_cfg["registry_model_name"]
        stage_or_alias = mlflow_cfg["registry_stage"]

        versions = client.search_model_versions(f"name='{model_name}'")
        matching = [v for v in versions if v.current_stage == stage_or_alias]
        if not matching:
            logger.warning(
                "No MLflow model version for %s in stage %s; falling back to local files",
                model_name,
                stage_or_alias,
            )
            return None

        latest = max(matching, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest.version}"
        # sklearn.load_model (not pyfunc.load_model) so .predict_proba() is
        # still available — pyfunc only exposes a generic .predict().
        sklearn_model = mlflow.sklearn.load_model(model_uri)

        run_id = latest.run_id
        preprocessor_path = client.download_artifacts(run_id, "preprocessor/preprocessor.joblib")
        feature_list_path = client.download_artifacts(run_id, "feature_list.json")

        preprocessor = joblib.load(preprocessor_path)
        feature_list = json.loads(Path(feature_list_path).read_text(encoding="utf-8"))

        return ModelArtifacts(
            preprocessor=preprocessor,
            model=sklearn_model,
            feature_list=feature_list,
            model_version=f"{model_name}/v{latest.version}",
            source="mlflow_registry",
        )
    except Exception as exc:  # noqa: BLE001 - any registry failure must fall back, not crash
        logger.warning("MLflow registry unreachable (%s); falling back to local files", exc)
        return None


def load_artifacts(config: dict) -> ModelArtifacts:
    registry_result = _load_from_mlflow_registry(config)
    if registry_result is not None:
        return registry_result
    return _load_from_local_files(config)
