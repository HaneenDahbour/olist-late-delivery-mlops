"""Register the frozen notebook 6 model + notebook 5 preprocessor in MLflow.

This does NOT train anything — it loads the already-fitted artifacts
that notebooks 05/06 produced and are frozen on disk (never refit,
never re-evaluated), logs them as one MLflow run, and registers the
model under config.mlflow.registry_model_name / registry_stage. That
run's artifacts become the source app/main.py's load_artifacts() will
find via the MLflow registry, instead of falling back to local files.

Usage (needs a reachable MLflow tracking server — see docker-compose.yml):
    python scripts/register_model.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

# mlflow prints a run-URL banner containing an emoji; Windows' default
# console codepage (cp1252) can't encode it and raises UnicodeEncodeError
# right as the run is closing, after the model was already registered.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from olist_mlops.config import find_repo_root, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    root = find_repo_root()
    paths = config["paths"]
    mlflow_cfg = config["mlflow"]

    preprocessor = joblib.load(root / paths["local_preprocessor"])
    model = joblib.load(root / paths["local_model"])
    feature_list = json.loads((root / paths["local_feature_list"]).read_text(encoding="utf-8"))
    results = json.loads((root / paths["local_results"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / paths["local_manifest"]).read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    selected = results["selected_model"]
    test_metrics = results["test"]["metrics"]

    with mlflow.start_run(run_name="freeze-notebook6-model") as run:
        mlflow.log_params(
            {
                "family": selected["family"],
                "class_weight": selected["class_weight"],
                "C": selected["C"],
                "decision_threshold": selected["threshold"],
                "feature_count": len(feature_list),
                "notebook5_manifest_sha256": manifest["artifacts"]["fitted_preprocessor"]["sha256"],
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        sample_input = None  # signature inference needs a real frame; skip for a frozen sklearn artifact
        signature = None
        try:
            import numpy as np

            sample_input = np.zeros((1, len(feature_list)), dtype="float32")
            signature = infer_signature(sample_input, model.predict_proba(sample_input))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not infer model signature (%s); logging without one", exc)

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name=mlflow_cfg["registry_model_name"],
        )

        # Filename must match what olist_mlops.artifacts._load_from_mlflow_registry
        # downloads: "preprocessor/preprocessor.joblib".
        preprocessor_path = root / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        preprocessor_path.unlink()

        # Filename must match what olist_mlops.artifacts._load_from_mlflow_registry
        # downloads: "feature_list.json" at the run's artifact root.
        feature_list_path = root / "feature_list.json"
        feature_list_path.write_text(json.dumps(feature_list), encoding="utf-8")
        mlflow.log_artifact(str(feature_list_path))
        feature_list_path.unlink()

        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{mlflow_cfg['registry_model_name']}'")
    this_version = next(v for v in versions if v.run_id == run_id)

    client.transition_model_version_stage(
        name=mlflow_cfg["registry_model_name"],
        version=this_version.version,
        stage=mlflow_cfg["registry_stage"],
        archive_existing_versions=True,
    )

    logger.info(
        "Registered %s version %s in stage %s (run_id=%s)",
        mlflow_cfg["registry_model_name"],
        this_version.version,
        mlflow_cfg["registry_stage"],
        run_id,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Model registration failed")
        sys.exit(1)
