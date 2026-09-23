from __future__ import annotations

from pathlib import Path

import pytest

from olist_mlops.config import find_repo_root, load_config


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


def _repo_file(relative: str) -> Path:
    return find_repo_root() / relative


LOCAL_ARTIFACT_PATHS = [
    "artifacts/notebook5_preprocessor.joblib",
    "artifacts/notebook5_feature_list.json",
    "artifacts/notebook5_manifest.json",
    "artifacts/notebook6_trained_model.joblib",
    "artifacts/notebook6_results.json",
]

LOCAL_DATA_PATHS = [
    "data/processed/test.parquet",
]


def artifacts_available() -> bool:
    return all(_repo_file(p).is_file() for p in LOCAL_ARTIFACT_PATHS)


def local_test_data_available() -> bool:
    return all(_repo_file(p).is_file() for p in LOCAL_DATA_PATHS)


requires_local_artifacts = pytest.mark.skipif(
    not artifacts_available(),
    reason=(
        "Local notebook artifacts (artifacts/notebook5_*.joblib, notebook6_*.joblib) "
        "are gitignored by design (Task 3: no fitted objects committed to Git). "
        "Run notebooks 01-06 locally to regenerate them, then re-run this test."
    ),
)

requires_local_test_data = pytest.mark.skipif(
    not local_test_data_available(),
    reason=(
        "data/processed/test.parquet is gitignored (raw Kaggle data + processed "
        "splits are never committed). Run notebooks 01-03 locally to regenerate it."
    ),
)
