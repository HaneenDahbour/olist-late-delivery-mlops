# Olist Late Delivery Prediction - MLOps Project

[![CI](https://github.com/HaneenDahbour/olist-late-delivery-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/HaneenDahbour/olist-late-delivery-mlops/actions/workflows/ci.yml)

End-to-end MLOps project using the Brazilian E-Commerce Public Dataset by Olist.

## Objective

Predict whether an e-commerce order will be delivered late while preventing data leakage and evaluating performance honestly on future chronological data.

## Task 1 - Relational Data Foundation

Completed:

- PostgreSQL 16 with Docker Compose
- nine Olist relational tables
- SQL schema creation and bulk ingestion
- data-integrity validation
- Python/PostgreSQL integration
- order-level relational dataset construction

## Task 2 - Notebook ML Workflow

Completed with six notebooks:

1. `01_read_and_join.ipynb` - build the order-level ML table
2. `02_create_labels.ipynb` - create the late-delivery target
3. `03_train_validation_test_split.ipynb` - chronological split
4. `04_train_eda.ipynb` - training-only EDA
5. `05_feature_engineering.ipynb` - feature engineering and fitted preprocessing
6. `06_train_tune_evaluate.ipynb` - baseline, validation tuning, and final test

## Chronological Split

- Train: 67,529 rows
- Validation: 14,470 rows
- Test: 14,471 rows
- Cross-split overlap: 0

Detailed EDA is performed on training data only.

## EDA Controls

Notebook 4 includes:

- missing-value analysis
- target imbalance
- numerical distributions
- IQR outlier audit
- rare and high-cardinality categories
- categorical crosstabs
- geographic analysis
- temporal and seasonality analysis
- leakage auditing

No holiday feature is invented because the source dataset contains no authoritative Brazilian holiday calendar.

## Feature Engineering

Notebook 5 starts with 51 prediction-time predictors and produces 184 encoded model features.

The fitted preprocessing pipeline uses:

- median numerical imputation
- missing-value indicators
- standardization
- categorical imputation
- one-hot encoding
- infrequent-category handling

The preprocessor is fitted on training data only.

Validation and test use the same saved fitted object with `.transform()` and never refit preprocessing.

## Leakage Prevention

Excluded predictors include:

- delivery outcome timestamps
- `delivery_delay_days`
- post-delivery review information
- `is_late`
- order, customer, and seller identifiers

## Model Selection

Notebook 6 begins with a dummy baseline.

Selected model:

- Balanced Logistic Regression
- C = 0.1
- Threshold = 0.692968

Validation:

- Baseline PR-AUC: 0.043124
- Model PR-AUC: 0.134065
- ROC-AUC: 0.775557

## Final Test Results

The chronological test set is opened only after model and threshold selection are frozen.

- Test rows: 14,471
- Late orders: 620
- PR-AUC: 0.081893
- ROC-AUC: 0.657710
- Precision: 0.071931
- Recall: 0.280645
- F1: 0.114511
- PR-AUC lift: 1.91x

The weaker final test performance is retained rather than used for further tuning.

## Reproducibility (Task 2 notebooks)

Run notebooks in order:

`01 -> 02 -> 03 -> 04 -> 05 -> 06`

This produces `data/processed/*.parquet` and `artifacts/*` locally. Neither is committed as raw
files — see [DVC](#dvc-data--artifact-versioning) below for how they're actually versioned and
shared.

---

# Task 3 — Inference Service, MLOps Infrastructure

Task 2 ends with six notebooks and a frozen model. Task 3 turns that into something a second
person (or your trainer) can run without opening Jupyter: a FastAPI service that loads the exact
same fitted preprocessor and model, serves predictions over HTTP, validates its inputs, logs every
call, and runs the same way from a container as it does on a laptop. Nothing in this section
retrains or refits anything — every fitted object is frozen in Task 2 and only ever `.transform()`/
`.predict_proba()`'d from here on.

## Architecture at a glance

```
                     ┌─────────────────────────────────────────┐
                     │        notebooks/05, 06 (Task 2)         │
                     │   fit preprocessor + train + freeze      │
                     └───────────────────┬───────────────────────┘
                                         │ produces (frozen, never refit)
                                         ▼
              artifacts/notebook5_preprocessor.joblib
              artifacts/notebook5_feature_list.json
              artifacts/notebook6_trained_model.joblib
              artifacts/notebook6_results.json  (frozen test metrics)
                                         │
                          tracked by DVC │ pushed to
                                         ▼
                     ┌───────────────────────────────┐
                     │   MinIO (S3-compatible store)  │◄── dvc push/pull
                     └───────────────┬───────────────┘
                                     │ read by
                                     ▼
                     ┌───────────────────────────────┐
              once ─► │ scripts/register_model.py     │
                     │ logs + registers the frozen    │
                     │ model as one MLflow run/version│
                     └───────────────┬───────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │   MLflow (registry + tracking) │◄── backed by Postgres
                     └───────────────┬───────────────┘
                                     │ loaded at startup by
                                     ▼
                     ┌───────────────────────────────┐
   HTTP requests ──►  │        app/main.py (FastAPI)   │──► logs to
                     │  /predict /predict/batch        │    Postgres prediction_log
                     │  /model/info /health /metrics   │    (+ a JSONL file always)
                     └───────────────────────────────┘
```

If MLflow is unreachable, the service falls back to the local `artifacts/*.joblib` files instead
of refusing to start — this is deliberate (see [`src/olist_mlops/artifacts.py`](src/olist_mlops/artifacts.py))
and is reported honestly in `GET /model/info`'s `artifact_source` field
(`"mlflow_registry"` or `"local_files"`).

## Repository layout (everything new in Task 3)

| Path | What it is |
|---|---|
| `config/config.yaml` | **The single source of truth.** Every path, column name, threshold, and connection setting used anywhere in `src/` or `app/` is read from here — nothing is hardcoded. See [below](#configconfigyaml-field-by-field) for every field explained. |
| `src/olist_mlops/config.py` | Loads `config.yaml`, resolves `${VAR}` / `${VAR:-default}` placeholders against environment variables, and locates the repo root by walking up from itself until it finds `config/config.yaml` — so it works the same regardless of current working directory. |
| `src/olist_mlops/features.py` | `build_prediction_features()` / `haversine_km()` — a byte-for-byte port of notebook 5's feature engineering. Reads its column lists from `config["feature_contract"]`, never hardcodes them. |
| `src/olist_mlops/artifacts.py` | Loads the fitted preprocessor + model + feature list. Tries the MLflow registry first, falls back to local `artifacts/*.joblib` files if the registry is unreachable (with a fast socket-level precheck, not mlflow's own multi-minute retry). |
| `src/olist_mlops/predict.py` | `PredictionService`: raw order → engineered features → `.transform()` → `.predict_proba()` → threshold → label. Never fits anything. |
| `src/olist_mlops/validation.py` | A Great Expectations suite (required fields, numeric ranges, allowed categories, missing-rate tolerance) plus a forbidden-column check, run on every request before it reaches the model. |
| `src/olist_mlops/schemas.py` | Pydantic request/response models. The request model is *generated* from `config["feature_contract"]` at startup, so the API schema can never drift from the config. |
| `src/olist_mlops/logging_setup.py` | Console (plain text) + rotating JSON file logging. No module anywhere uses `print()`. |
| `src/olist_mlops/prediction_log.py` | Logs every prediction (input, output, latency, model version) to a JSONL file always, and to a Postgres `prediction_log` table when reachable (best-effort, 2-second connect timeout, never blocks a request). |
| `app/main.py` | The FastAPI app: `/health`, `/model/info`, `/predict`, `/predict/batch`, `/metrics` (Prometheus). |
| `scripts/register_model.py` | Loads the frozen local artifacts and registers them as one MLflow run + model version. Run once (or whenever the frozen model changes) — not part of the request path. |
| `Dockerfile` | Builds the API image: installs `requirements.txt` only (no notebook/dev tooling), copies `src/`, `app/`, `config/`, `scripts/`. |
| `docker-compose.yml` | The full local stack: `postgres`, `minio`, `create-bucket` (one-off), `dvc-pull` (one-off), `mlflow`, `register-model` (one-off), `api`. |
| `docker/mlflow.Dockerfile` | A minimal MLflow server image (mlflow + boto3 + psycopg, pinned to the same versions as `requirements.txt`). |
| `docker/dvc.Dockerfile` | Runs `dvc pull` inside the compose network (MinIO is `http://minio:9000` there, not `localhost:9000`) before `register-model` starts. Best-effort: exits 0 even if the remote is empty/unreachable, so it never blocks the pipeline when the real files are already on disk some other way (e.g. a submission zip). |
| `.dvc/config`, `*.dvc` files | DVC tracks `data/processed/*.parquet` and `artifacts/*` as content-addressed pointers, with MinIO as the remote. Actual bytes are never in Git; `dvc pull` fetches them. |
| `tests/` | `test_config.py`, `test_features.py` (pure unit tests, no external files needed), `test_notebook_parity.py`, `test_predict.py`, `test_api.py` (need the local artifacts/data — see below), `test_validation.py`, `conftest.py` (the skip logic that makes the previous point work). |
| `.github/workflows/ci.yml` | Runs ruff, black --check, mypy, and pytest on every push/PR, then builds the API image and pushes it to GHCR (only from `main`, using the automatic `GITHUB_TOKEN`). |
| `.pre-commit-config.yaml` | Same three checks as local git hooks. |
| `requirements.txt` / `requirements-dev.txt` | Runtime-only vs. runtime+notebook+test+lint tooling. The Docker image installs only `requirements.txt`. |
| `MONITORING.md` | What's actually live (Prometheus counters, prediction logs) vs. decided-but-not-wired (latency/error-rate/drift alert thresholds, with the reasoning for each) — the "decide what you would alert on, write it down" deliverable. |

### `config/config.yaml` field by field

```yaml
paths:              # All relative to the repo root (auto-detected). Where every artifact/log lives.
model:               # Frozen facts from notebook 6 — decision_threshold and expected_feature_count
                      # are asserted against at load time; if they ever mismatch, the service refuses
                      # to start rather than silently serving a different model than you think.
mlflow:               # tracking_uri (env-overridable), the registry model name, and which
                      # stage/alias counts as "the" production model.
artifact_store:       # The S3-compatible endpoint (MinIO locally) DVC and MLflow both use.
database:             # Postgres connection info for the prediction_log table.
api:                  # Host/port/title/version and max_batch_size (a batch predict request larger
                      # than this is rejected with 422, not silently truncated).
logging:              # Level, format, whether to log to console/file, file rotation size.
monitoring:           # prediction_log_backend (postgres|file), and the drift/latency/error-rate
                      # thresholds a future monitoring job would alert on (not yet wired to an
                      # alerting system — see Known Limitations).
feature_contract:     # numeric_features / categorical_features / timestamp_features (the exact
                      # raw predictor columns notebook 5 uses), engineered_features (computed, not
                      # accepted directly), and forbidden_columns (any request containing one of
                      # these — e.g. is_late — is rejected with 422; these are leakage sources).
```

## Running it yourself

### Option A — full stack in Docker (closest to how it'd really run)

```bash
git clone <this-repo>
cd mlops-olist
cp .env.example .env
docker compose up -d --build
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

That's genuinely the whole thing — one command after the one-time `.env` copy. A `dvc-pull`
service inside the compose stack itself fetches `data/processed/*.parquet` and `artifacts/*` from
MinIO before `register-model` runs, so no separate manual `dvc pull` is needed. Verified twice: on
a fresh `git clone` into a directory reusing this machine's already-populated MinIO volume, and
(harder test) on a copy with zero prior state at all — no `.git`, no DVC remote, MinIO started
empty — where `dvc-pull` fails fast and the pipeline still boots because the real files were
already on disk (see the zip-delivery note in [Known limitations](#known-limitations-honest-not-hidden)).

The one real prerequisite: **something** has to have run `dvc push` against this artifact store at
least once — same as any real S3 bucket doesn't spontaneously contain your model either. This
repo's `artifacts/` and `data/processed/` are already pushed to this project's MinIO volume; a
teammate cloning it on the same machine/Docker host reuses that volume automatically.

### Option B — API only, without Docker (needs `artifacts/` and `data/processed/` present locally)

```bash
python -m venv .venv-service
.venv-service\Scripts\pip install -r requirements.txt
.venv-service\Scripts\pip install --no-deps -e .
.venv-service\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Without a reachable MLflow registry, this loads the model from local `artifacts/*.joblib` files
(reported as `artifact_source: "local_files"` in `/model/info`) — no MLflow/Postgres/MinIO needed
for this option, at the cost of not proving the registry path.

## The five things you should be able to do (with the exact commands)

**1. Clean-clone startup with one command.**
```bash
docker compose up -d --build
```
Verified: the four long-running services (`postgres`, `minio`, `mlflow`, `api`) report healthy, and
the three one-off jobs (`create-bucket`, `dvc-pull`, `register-model`) complete successfully before
`api` starts.

**2. A prediction with probability and model version.**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
# → {"label":"on_time","probability":0.443...,"model_version":"olist-late-delivery/v3"}
```
`model_version` and `GET /model/info`'s `artifact_source` field tell you whether this came from the
MLflow registry or the local-file fallback.

**3. Notebook/service parity.**
```bash
pytest tests/test_notebook_parity.py -v
```
`test_service_pipeline_matches_frozen_notebook_test_metrics` reproduces notebook 6's own frozen
test-set metrics (precision, recall, F1, ROC-AUC, PR-AUC) to `1e-9` using this package's feature
engineering and the same frozen artifacts. Verified on a genuinely fresh `git clone` + `dvc pull`
(not just in the working copy that built it) — see the commit history for the exact reproduction.

**4. Bad data or a failing test being caught.**
```bash
curl -X POST http://localhost:8000/predict -d '{"item_count": null, ...}'          # → 422 (Great Expectations)
curl -X POST http://localhost:8000/predict -d '{"is_late": 1, ...}'                # → 422 (forbidden column)
pre-commit run --all-files                                                          # blocks lint/format/type violations
```

**5. Every folder, config value, and tool explained.**
See [Repository layout](#repository-layout-everything-new-in-task-3) and
[`config.yaml` field by field](#configconfigyaml-field-by-field) above.

## DVC: data & artifact versioning

`data/processed/*.parquet` and `artifacts/*` are tracked by DVC, not committed as raw files —
only small `.dvc` pointer files (a content hash + size) live in Git. The actual bytes live in the
MinIO bucket configured as DVC's remote (`.dvc/config`).

```bash
dvc pull    # fetch the real files a .dvc pointer refers to
dvc push    # after regenerating data/artifacts (re-running notebooks), publish the new versions
dvc status --cloud   # confirm local cache and remote agree
```

**A bug we found and fixed here**: a `.gitignore` pattern of the form `dir/* ` + `!dir/*.dvc`
(blanket-ignore then negate) works correctly for `git`, but DVC's own ignore-matcher prunes the
whole directory once the blanket glob matches and never rescans it for the negated exception. This
silently made `dvc push`/`pull` (no arguments) see only half the tracked files, while `dvc status
--cloud` reported "in sync" the entire time — because it was only checking the half it could see.
Confirmed by cloning fresh and running `dvc pull`: `artifacts/` came back empty. Fixed by ignoring
by extension (`artifacts/*.joblib`, `artifacts/*.json`, `artifacts/*.parquet`) instead, which needs
no negation. Re-verified on a second fresh clone: `dvc pull` now checks out all 11 tracked files.

## Known limitations (honest, not hidden)

- **Great Expectations coverage is not exhaustive.** The suite in `validation.py` covers the
  structurally-critical fields (required-not-null, numeric ranges on a handful of columns, allowed
  categories for state/payment-type codes, one missing-rate check) — not all 35 numeric columns
  individually.
- **`predict_requests_total{status="rejected"}` under-counts.** FastAPI's own request-body schema
  validation (e.g. an extra/forbidden field) returns 422 before the handler function runs, so that
  rejection path isn't counted — only Great-Expectations-layer rejections inside the handler are.
- **Monitoring thresholds in `config.yaml` (`drift_check_window_days`, `latency_alert_ms`,
  `error_rate_alert_pct`) are not wired to an alerting system.** They're read by nothing yet; the
  `/metrics` endpoint exposes the raw Prometheus counters/histograms a future alert rule would use.
- **A zipped/offline copy of this repo has no `.git` and no access to this machine's MinIO.**
  `dvc-pull` fails fast in that case (by design — see its Dockerfile) and the pipeline still boots
  because a zip built for submission includes the real `data/processed/*.parquet` and `artifacts/*`
  files directly, not just their `.dvc` pointers. Verified: extracted such a zip fresh, ran
  `pytest` standalone (32/32 passed, no git/DVC involved), then `docker compose up -d --build`
  under a brand-new project name with an empty MinIO volume — all four services came up healthy and
  served a correct prediction.
- **Model-dependent tests need local artifacts.** `test_notebook_parity.py`, `test_predict.py`,
  and `test_api.py` skip (with an explicit reason, not silently) when `artifacts/*` and
  `data/processed/*.parquet` aren't present — which is always true in CI today, since nothing
  provisions them there. Run `dvc pull` (against a running MinIO) or the notebooks locally, then
  re-run `pytest`, to execute them for real.
- **Running `pytest` locally while `docker compose`'s MLflow container has port 5000 bound to the
  host** makes the test suite reach the *live* registry instead of the local-file fallback, and
  downloading the real model artifact hung once in this environment (most likely Windows Defender
  scanning the freshly-downloaded file). Workaround: `docker compose stop mlflow` before running
  `pytest` locally, `docker compose start mlflow` after.
