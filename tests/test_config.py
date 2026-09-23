from olist_mlops.config import find_repo_root, load_config


def test_find_repo_root_locates_config_yaml():
    root = find_repo_root()
    assert (root / "config" / "config.yaml").is_file()


def test_load_config_matches_frozen_model_contract():
    config = load_config()
    model = config["model"]
    assert model["decision_threshold"] == 0.692968487739563
    assert model["expected_feature_count"] == 184
    assert model["algorithm"] == "LogisticRegression"
    assert model["target_column"] == "is_late"


def test_feature_contract_column_counts():
    contract = load_config()["feature_contract"]
    assert len(contract["numeric_features"]) == 35
    assert len(contract["categorical_features"]) == 5
    assert "is_late" in contract["forbidden_columns"]


def test_env_var_default_is_used_when_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    load_config.cache_clear()
    config = load_config()
    assert config["mlflow"]["tracking_uri"] == "http://localhost:5000"


def test_env_var_override_is_applied(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
    load_config.cache_clear()
    config = load_config()
    assert config["mlflow"]["tracking_uri"] == "http://mlflow.internal:5000"
    load_config.cache_clear()
