# Monitoring & Alerting Plan

What's already implemented, what's only decided-and-written-down (per Task 3 item 10's own
wording: "decide what you would alert on, and write it down" — not every alert needs a live
alerting system for this task), and why.

## What's live today

| Signal | Where | How to see it |
|---|---|---|
| Request count, by route and status | `predict_requests_total` (Prometheus counter) | `GET /metrics` |
| Request latency | `predict_request_latency_seconds` (Prometheus histogram) | `GET /metrics` |
| Every prediction (input, output, latency, model version) | `logs/predictions/*.jsonl` always; `prediction_log` Postgres table when reachable | `SELECT * FROM prediction_log`, or read the JSONL files |
| Service logs (startup, validation failures, registry fallbacks) | `logs/service/service.log` (JSON) + console | `tail -f logs/service/service.log` |

## What's decided but not wired to a live alerting system

`config.yaml`'s `monitoring:` block holds the thresholds below. Nothing pages anyone yet — there's
no Alertmanager/PagerDuty/Slack webhook in this repo — but the numbers and the reasoning for each
are fixed here so wiring one up later is a config-reading exercise, not a design one.

### 1. Latency alert — `latency_alert_ms: 500`
**Alert when**: p95 of `predict_request_latency_seconds` over a 5-minute window exceeds 500ms.
**Why 500ms**: this is a synchronous single-order prediction over a small LogisticRegression model
— anything above half a second means something is wrong upstream (MLflow registry reachability
falling back mid-request, Postgres connection pool exhaustion), not model compute cost.
**What to do**: page on-call only if sustained for 3+ consecutive windows (avoid paging on a single
slow request from cold-start).

### 2. Error rate alert — `error_rate_alert_pct: 5`
**Alert when**: `status="error"` (not `"rejected"` — a 422 for bad input is a client problem, not a
service problem) exceeds 5% of `predict_requests_total` over a 10-minute window.
**Why separate "error" from "rejected"**: a spike in 422s means upstream client data changed
(a new order field, a new category value) — worth investigating but not paging anyone at 2am. A
spike in 500s means the service itself is broken.
**What to do**: page immediately — a 500 means an unhandled exception reached the top of the
request handler, which `app/main.py` treats as a bug by design.

### 3. Drift check — `drift_check_window_days: 7`
**Alert when**: the weekly share of `label="late"` predictions drifts more than 5 percentage
points from the training-set base rate (7.99% late, from notebook 4's EDA — see the main README).
**Why this check, not a full feature-distribution drift test**: Olist's order volume swings
seasonally (Black Friday, Christmas) in ways that shift genuine feature distributions without the
*model* being wrong — a full per-feature KS-test would false-positive constantly. Watching the
*prediction* distribution instead catches the case that actually matters operationally: the model
starting to call everything late (or nothing late), which usually means an upstream feature-join
broke, not that delivery got genuinely worse.
**What to do**: don't page — flag for the next model-review cycle. This dataset's model is weak
(test PR-AUC 0.082, see README) by design of Task 2's evaluation; a week-to-week wobble is
expected and shouldn't trigger urgent action.

### 4. Ground-truth evaluation — the actual point of storing prediction logs
Every prediction is logged with a `request_id`. When an order's real `order_delivered_customer_date`
becomes known (days after the prediction was made), the plan is: join `prediction_log` against the
orders table on order identity, recompute precision/recall/F1 on that cohort, and compare against
the frozen notebook 6 test metrics. This is the actual reason item 10 asks to store prediction
logs — "evaluate later when the real delivery date arrives" — and is not yet automated (no join
script exists yet); the data needed for it is already being collected correctly.

## Known gap

None of the above triggers a real notification today. The honest state is: the signals exist, the
thresholds and reasoning are decided, and `/metrics` exposes what a Prometheus/Grafana/Alertmanager
stack would need — but that stack itself isn't deployed here. Wiring it up is the natural next
step, not attempted in this submission to avoid standing up infrastructure (a metrics scraper, an
alert-routing target) with no real destination to alert to.
