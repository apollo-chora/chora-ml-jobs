# chora-ml-jobs

Batch ML jobs for the Chora platform: drift detection, model evaluation, and
billing aggregation. One Docker image serves all jobs — the job is selected at
runtime via the `JOB` environment variable.

The unit is cloud-neutral: PostgreSQL (Familiar DB) for persistence, NATS
JetStream for alert events, MLflow for experiment tracking, and env-backed
configuration. No cloud account or managed services (Cloud Pub/Sub, Cloud
Storage, Cloud Run) are required.

## Jobs

| Job | Description | Schedule (prod) |
|-----|-------------|-----------------|
| `drift_detector` | Detects embedding distribution drift in Familiar persona vectors | Daily |
| `eval_runner` | Evaluates ML model quality against golden test datasets | Weekly / post-training |
| `billing_aggregator` | Aggregates ML token usage per tenant for billing periods | End of billing period |

## Architecture

- **Compute**: any host running the container image; one job per invocation.
- **Database**: PostgreSQL (`chora_familiar`, pgvector extension). The jobs read
  `familiar_memories`, `retraining_trigger_configs`, and
  `model_registry_entries`, and write `drift_detection_results`. The schema is
  owned by the Familiar service — no migrations ship with this repo.
- **Event bus**: NATS JetStream. Drift and regression alerts are published to
  the `chora.ml.alerts` subject (envelope as message headers, JSON body) for
  chora-communication to consume.
- **Experiment tracking**: MLflow at `MLFLOW_TRACKING_URI`.
- **Ports**: none — batch jobs listen on no port.

## Configuration

All configuration is via environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JOB` | Yes | — | Job to run: `drift_detector`, `eval_runner`, `billing_aggregator` |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string (Familiar DB) |
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | MLflow tracking server URL |
| `MLFLOW_EXPERIMENT_NAME` | No | `chora-ml-jobs` | MLflow experiment name |
| `CHORA_NATS_URL` | No | `nats://localhost:4222` | NATS broker URL for alert events |
| `FIXTURES_DIR` | No | `/app/fixtures` (container) | Golden dataset directory for `eval_runner` |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Local Development

### Run with Docker

```bash
# Build the image
docker build -t chora-ml-jobs .

# Run a specific job
docker run --rm \
  -e JOB=drift_detector \
  -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/chora_familiar \
  -e CHORA_NATS_URL=nats://host.docker.internal:4222 \
  chora-ml-jobs
```

### Run Without Docker

```bash
# Install dev dependencies
uv sync --all-groups
```

# Set environment variables
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chora_familiar
export CHORA_NATS_URL=nats://localhost:4222

# Run a job
JOB=drift_detector python -m chora_ml_jobs.entrypoint
```

### In the Chora compose stack

The ML jobs container is included in the main docker-compose stack
(`chora-stack`), which provides Postgres, NATS, and MLflow:

```bash
docker compose up -d mlflow chora-ml-jobs

# Exec into the container to run a specific job
docker compose exec chora-ml-jobs bash
JOB=drift_detector python -m chora_ml_jobs.entrypoint
```

## Development

```bash
uv sync --all-groups   # or: pip install -e ".[dev]"
ruff check src
ruff format --check src
mypy src
```

## License

UNLICENSED — Chora Platform.
