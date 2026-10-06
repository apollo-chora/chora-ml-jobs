# chora-ml-jobs

## About

`chora-ml-jobs` is a Python package for Chora ML batch jobs. It provides jobs for embedding drift detection, model evaluation against golden datasets, and ML token-usage billing aggregation, with one container image selecting the job through the `JOB` environment variable. The jobs use PostgreSQL for Familiar data, MLflow for experiment tracking, and NATS JetStream for ML alert events.

## Quick start

Prerequisites:

- Python 3.13 or Docker
- PostgreSQL for the Familiar database
- NATS JetStream for alert publishing when running `drift_detector` or `eval_runner`
- MLflow for experiment tracking

Install the development environment with uv:

```bash
uv sync --all-groups
```

Set the required database connection and, when needed, the NATS URL:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chora_familiar
export CHORA_NATS_URL=nats://localhost:4222
```

Run a job:

```bash
JOB=drift_detector python -m chora_ml_jobs.entrypoint
```

Or build and run the container:

```bash
docker build -t chora-ml-jobs .

docker run --rm \
  -e JOB=drift_detector \
  -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/chora_familiar \
  -e CHORA_NATS_URL=nats://host.docker.internal:4222 \
  chora-ml-jobs
```

## Usage

The container entrypoint is:

```text
python -m chora_ml_jobs.entrypoint
```

Select the job with `JOB`. Supported values are:

| Job | What it does |
| --- | --- |
| `drift_detector` | Compares recent and baseline Familiar persona embeddings, logs drift metrics to MLflow, stores results in `drift_detection_results`, and publishes a `chora.ml.drift_detected` alert when the configured cosine-drift threshold is exceeded. |
| `eval_runner` | Loads JSON golden datasets from `FIXTURES_DIR`, evaluates production models from `model_registry_entries`, logs evaluation metrics to MLflow, and publishes `chora.ml.model_eval_regression` alerts for regressions greater than 5 points. |
| `billing_aggregator` | Entry point for the billing aggregation job. The current implementation is a no-op stub. |

`drift_detector` uses a 30-day baseline window and a 7-day evaluation window by default. Its default cosine-shift warning and critical thresholds are 0.15 and 0.25; database configuration in `retraining_trigger_configs` can override the warning threshold and evaluation window.

Alerts from the drift and evaluation jobs are published to the NATS JetStream subject `chora.ml.alerts`. The event envelope is sent in message headers, with a JSON envelope in the message body.

Configuration is provided through environment variables:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `JOB` | Yes | | Selects the batch job. |
| `DATABASE_URL` | Yes | | PostgreSQL connection string for the Familiar database. |
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | MLflow tracking server. |
| `MLFLOW_EXPERIMENT_NAME` | No | `chora-ml-jobs` | Base MLflow experiment name. |
| `CHORA_NATS_URL` | No | `nats://localhost:4222` | NATS broker used for alert events. |
| `FIXTURES_DIR` | No | `fixtures` in a source checkout, `/app/fixtures` in the container | Golden dataset directory used by `eval_runner`. |
| `LOG_LEVEL` | No | `INFO` | Logging level. |

The jobs are batch processes and do not listen on HTTP ports.

## Development

The project uses a `src` layout:

```text
src/chora_ml_jobs/
  alerting.py
  billing_aggregator.py
  config.py
  drift_detector.py
  entrypoint.py
  eval_runner.py
fixtures/
  *.json
```

Install development dependencies:

```bash
uv sync --all-groups
```

Run the available checks:

```bash
ruff check src
ruff format --check src
mypy src
```

Run the test suite with:

```bash
pytest
```

The Dockerfile builds a Python 3.13 image, includes the source tree and fixture files, and runs as a non-root user. The package uses Hatchling for builds and pins resolved dependencies in `uv.lock`.
