"""Drift Detector — detects embedding distribution drift in Familiar persona vectors.

This job compares the current distribution of persona embeddings against the
reference distribution stored at the last model training run. When drift exceeds
a configurable threshold, it publishes a `chora.ml.drift_detected` event to
trigger model retraining.

Metrics tracked in MLflow:
    - Mean cosine similarity between recent and baseline distributions
    - PSI (Population Stability Index) per embedding dimension
    - Jensen-Shannon divergence of embedding clusters
    - Drift severity classification (none / low / medium / high)
"""

from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import mlflow
import numpy as np
import psycopg2
import psycopg2.extras
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

from chora_ml_jobs.alerting import publish_drift_alert
from chora_ml_jobs.config import Config, setup_logging

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

EVALUATION_WINDOW_DAYS = 7
BASELINE_WINDOW_DAYS = 30
PSI_BUCKET_COUNT = 10
EMBEDDING_DIM = 1536

# Fallback thresholds if no RetrainingTriggerConfig found in DB.
DEFAULT_COSINE_WARNING_THRESHOLD = 0.15
DEFAULT_COSINE_CRITICAL_THRESHOLD = 0.25


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class DriftThresholds:
    """Configurable drift detection thresholds from retraining_trigger_configs."""

    cosine_warning: float = DEFAULT_COSINE_WARNING_THRESHOLD
    cosine_critical: float = DEFAULT_COSINE_CRITICAL_THRESHOLD
    evaluation_window_days: int = EVALUATION_WINDOW_DAYS
    cooldown_hours: int = 24
    is_enabled: bool = True
    last_triggered_at: datetime | None = None


@dataclass
class DriftMetrics:
    """Results of drift metric computation."""

    mean_cosine_similarity: float
    psi_score: float
    js_divergence: float
    sample_count_baseline: int
    sample_count_evaluation: int
    drift_detected: bool
    severity: str  # none, low, medium, high


# ── Database Access ──────────────────────────────────────────────────────────


def _get_connection(database_url: str) -> psycopg2.extensions.connection:
    """Create a direct psycopg2 connection (batch job, not via service API)."""
    return psycopg2.connect(database_url)


def _fetch_embeddings(
    conn: psycopg2.extensions.connection,
    start: datetime,
    end: datetime,
) -> np.ndarray:
    """Fetch embedding vectors from familiar_memories within a time window.

    Args:
        conn: PostgreSQL connection.
        start: Window start (inclusive).
        end: Window end (exclusive).

    Returns:
        numpy array of shape (n_samples, EMBEDDING_DIM). Empty array if none found.
    """
    query = """
        SELECT embedding::text
        FROM familiar_memories
        WHERE embedding IS NOT NULL
          AND created_at >= %s
          AND created_at < %s
        ORDER BY created_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(query, (start, end))
        rows = cur.fetchall()

    if not rows:
        return np.empty((0, EMBEDDING_DIM))

    embeddings = []
    for (embedding_text,) in rows:
        # pgvector returns text like '[0.1,0.2,...]'
        vec = np.fromstring(embedding_text.strip("[]"), sep=",", dtype=np.float64)
        if vec.shape[0] == EMBEDDING_DIM:
            embeddings.append(vec)

    if not embeddings:
        return np.empty((0, EMBEDDING_DIM))

    return np.array(embeddings)


def _load_thresholds(conn: psycopg2.extensions.connection) -> DriftThresholds:
    """Load drift thresholds from retraining_trigger_configs table.

    Falls back to defaults if the table has no 'cosine_drift' config.
    """
    query = """
        SELECT warning_threshold, critical_threshold, evaluation_window_days,
               cooldown_hours, is_enabled, last_triggered_at
        FROM retraining_trigger_configs
        WHERE metric_name = 'cosine_drift'
          AND is_enabled = TRUE
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()

    if row is None:
        logger.info("no cosine_drift config found, using defaults")
        return DriftThresholds()

    return DriftThresholds(
        cosine_warning=float(row[0]),
        cosine_critical=float(row[1]),
        evaluation_window_days=int(row[2]),
        cooldown_hours=int(row[3]),
        is_enabled=bool(row[4]),
        last_triggered_at=row[5],
    )


def _save_drift_result(
    conn: psycopg2.extensions.connection,
    run_id: str,
    metrics: DriftMetrics,
    baseline_start: datetime,
    baseline_end: datetime,
    eval_start: datetime,
    eval_end: datetime,
    threshold_used: float,
) -> str:
    """Persist DriftDetectionResult to the database.

    Returns:
        The UUID of the inserted row.
    """
    result_id = str(uuid.uuid4())
    query = """
        INSERT INTO drift_detection_results (
            id, detection_run_id, metric_name,
            baseline_window_start, baseline_window_end,
            evaluation_window_start, evaluation_window_end,
            mean_cosine_similarity, psi_score, js_divergence,
            sample_count_baseline, sample_count_evaluation,
            drift_detected, threshold_used
        ) VALUES (
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s
        )
    """
    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                result_id,
                run_id,
                "embedding_cosine_drift",
                baseline_start,
                baseline_end,
                eval_start,
                eval_end,
                round(metrics.mean_cosine_similarity, 4),
                round(metrics.psi_score, 4),
                round(metrics.js_divergence, 4),
                metrics.sample_count_baseline,
                metrics.sample_count_evaluation,
                metrics.drift_detected,
                round(threshold_used, 4),
            ),
        )
    conn.commit()
    return result_id


def _update_last_triggered(
    conn: psycopg2.extensions.connection,
    now: datetime,
) -> None:
    """Update last_triggered_at on the cosine_drift retraining config."""
    query = """
        UPDATE retraining_trigger_configs
        SET last_triggered_at = %s
        WHERE metric_name = 'cosine_drift'
    """
    with conn.cursor() as cur:
        cur.execute(query, (now,))
    conn.commit()


# ── Drift Metric Computation ────────────────────────────────────────────────


def compute_mean_cosine_similarity(
    baseline: np.ndarray,
    evaluation: np.ndarray,
) -> float:
    """Compute mean cosine similarity between baseline and evaluation centroids.

    Uses centroid-to-centroid comparison for a single aggregate drift signal.
    """
    baseline_centroid = baseline.mean(axis=0).reshape(1, -1)
    eval_centroid = evaluation.mean(axis=0).reshape(1, -1)
    sim = cosine_similarity(baseline_centroid, eval_centroid)[0][0]
    return float(sim)


def compute_psi(
    baseline: np.ndarray,
    evaluation: np.ndarray,
    buckets: int = PSI_BUCKET_COUNT,
) -> float:
    """Compute Population Stability Index (PSI) across embedding dimensions.

    PSI measures how much the distribution of embedding values has shifted.
    Returns the mean PSI across all dimensions.

    Interpretation:
        PSI < 0.1  => no significant drift
        PSI 0.1-0.25 => moderate drift
        PSI > 0.25 => significant drift
    """
    eps = 1e-8
    psi_per_dim = []

    for dim in range(baseline.shape[1]):
        base_vals = baseline[:, dim]
        eval_vals = evaluation[:, dim]

        # Create buckets from combined range
        combined = np.concatenate([base_vals, eval_vals])
        bin_edges = np.linspace(combined.min(), combined.max(), buckets + 1)

        base_hist, _ = np.histogram(base_vals, bins=bin_edges)
        eval_hist, _ = np.histogram(eval_vals, bins=bin_edges)

        # Normalize to proportions
        base_pct = base_hist / max(base_hist.sum(), 1) + eps
        eval_pct = eval_hist / max(eval_hist.sum(), 1) + eps

        # PSI formula: sum( (eval% - base%) * ln(eval% / base%) )
        psi_dim = float(np.sum((eval_pct - base_pct) * np.log(eval_pct / base_pct)))
        psi_per_dim.append(psi_dim)

    return float(np.mean(psi_per_dim))


def compute_js_divergence(
    baseline: np.ndarray,
    evaluation: np.ndarray,
    buckets: int = PSI_BUCKET_COUNT,
) -> float:
    """Compute Jensen-Shannon divergence between embedding distributions.

    Uses per-dimension histograms and averages the JS divergence across all dims.

    Returns:
        Mean JS divergence (0 = identical, 1 = maximally different).
    """
    js_per_dim = []

    for dim in range(baseline.shape[1]):
        base_vals = baseline[:, dim]
        eval_vals = evaluation[:, dim]

        combined = np.concatenate([base_vals, eval_vals])
        bin_edges = np.linspace(combined.min(), combined.max(), buckets + 1)

        base_hist, _ = np.histogram(base_vals, bins=bin_edges, density=True)
        eval_hist, _ = np.histogram(eval_vals, bins=bin_edges, density=True)

        # Normalize to probability distributions
        base_prob = base_hist / max(base_hist.sum(), 1e-10)
        eval_prob = eval_hist / max(eval_hist.sum(), 1e-10)

        js_val = float(jensenshannon(base_prob, eval_prob))
        # jensenshannon returns the square root of JS divergence
        js_per_dim.append(js_val**2)

    return float(np.mean(js_per_dim))


def classify_severity(cosine_shift: float, thresholds: DriftThresholds) -> str:
    """Classify drift severity based on cosine shift magnitude.

    Cosine shift = 1 - mean_cosine_similarity (higher = more drift).
    """
    if cosine_shift >= thresholds.cosine_critical:
        return "high"
    if cosine_shift >= thresholds.cosine_warning:
        return "medium"
    if cosine_shift >= thresholds.cosine_warning * 0.5:
        return "low"
    return "none"


def compute_drift_metrics(
    baseline: np.ndarray,
    evaluation: np.ndarray,
    thresholds: DriftThresholds,
) -> DriftMetrics:
    """Compute all drift metrics between baseline and evaluation sets."""
    mean_cosine = compute_mean_cosine_similarity(baseline, evaluation)
    psi = compute_psi(baseline, evaluation)
    js_div = compute_js_divergence(baseline, evaluation)

    cosine_shift = 1.0 - mean_cosine
    drift_detected = cosine_shift > thresholds.cosine_warning
    severity = classify_severity(cosine_shift, thresholds)

    return DriftMetrics(
        mean_cosine_similarity=mean_cosine,
        psi_score=psi,
        js_divergence=js_div,
        sample_count_baseline=baseline.shape[0],
        sample_count_evaluation=evaluation.shape[0],
        drift_detected=drift_detected,
        severity=severity,
    )


# ── Main Entrypoint ─────────────────────────────────────────────────────────


def main() -> int:
    """Run the drift detection job.

    Returns:
        int: Exit code (0 = success, 1 = error).
    """
    config = Config.from_env()
    setup_logging(config)

    logger.info(
        "drift_detector starting",
        extra={
            "mlflow_uri": config.mlflow_tracking_uri,
            "database": config.database_url[:20] + "...",
        },
    )

    conn: psycopg2.extensions.connection | None = None
    try:
        # 1. Connect to PostgreSQL (chora-familiar DB)
        conn = _get_connection(config.database_url)
        logger.info("connected to familiar database")

        # 2. Load configurable thresholds from RetrainingTriggerConfig
        thresholds = _load_thresholds(conn)
        if not thresholds.is_enabled:
            logger.info("drift detection disabled via retraining_trigger_configs")
            return 0

        # Check cooldown
        now = datetime.now(UTC)
        if thresholds.last_triggered_at is not None:
            cooldown_end = thresholds.last_triggered_at + timedelta(hours=thresholds.cooldown_hours)
            if now < cooldown_end:
                logger.info(
                    "drift detection in cooldown until %s, skipping",
                    cooldown_end.isoformat(),
                )
                return 0

        # 3. Define time windows
        eval_window_days = thresholds.evaluation_window_days
        eval_end = now
        eval_start = now - timedelta(days=eval_window_days)
        baseline_end = eval_start
        baseline_start = baseline_end - timedelta(days=BASELINE_WINDOW_DAYS)

        logger.info(
            "time windows: baseline=%s..%s, evaluation=%s..%s",
            baseline_start.isoformat(),
            baseline_end.isoformat(),
            eval_start.isoformat(),
            eval_end.isoformat(),
        )

        # 4. Fetch embeddings
        baseline_embeddings = _fetch_embeddings(conn, baseline_start, baseline_end)
        eval_embeddings = _fetch_embeddings(conn, eval_start, eval_end)

        logger.info(
            "fetched embeddings: baseline=%d, evaluation=%d",
            baseline_embeddings.shape[0],
            eval_embeddings.shape[0],
        )

        # Minimum sample sizes for meaningful comparison
        min_samples = 10
        if baseline_embeddings.shape[0] < min_samples:
            logger.warning(
                "insufficient baseline samples (%d < %d), skipping drift detection",
                baseline_embeddings.shape[0],
                min_samples,
            )
            return 0

        if eval_embeddings.shape[0] < min_samples:
            logger.warning(
                "insufficient evaluation samples (%d < %d), skipping drift detection",
                eval_embeddings.shape[0],
                min_samples,
            )
            return 0

        # 5. Set up MLflow experiment
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(f"{config.mlflow_experiment_name}/drift-detection")

        with mlflow.start_run(run_name=f"drift-detection-{now.strftime('%Y%m%d-%H%M%S')}") as run:
            run_id = run.info.run_id

            # 6. Compute drift metrics
            metrics = compute_drift_metrics(baseline_embeddings, eval_embeddings, thresholds)
            cosine_shift = 1.0 - metrics.mean_cosine_similarity

            logger.info(
                "drift metrics computed",
                extra={
                    "mean_cosine_similarity": metrics.mean_cosine_similarity,
                    "cosine_shift": cosine_shift,
                    "psi_score": metrics.psi_score,
                    "js_divergence": metrics.js_divergence,
                    "drift_detected": metrics.drift_detected,
                    "severity": metrics.severity,
                },
            )

            # 7. Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "mean_cosine_similarity": metrics.mean_cosine_similarity,
                    "cosine_shift": cosine_shift,
                    "psi_score": metrics.psi_score,
                    "js_divergence": metrics.js_divergence,
                    "sample_count_baseline": metrics.sample_count_baseline,
                    "sample_count_evaluation": metrics.sample_count_evaluation,
                    "drift_detected": 1.0 if metrics.drift_detected else 0.0,
                }
            )
            mlflow.log_params(
                {
                    "warning_threshold": thresholds.cosine_warning,
                    "critical_threshold": thresholds.cosine_critical,
                    "evaluation_window_days": thresholds.evaluation_window_days,
                    "baseline_window_days": BASELINE_WINDOW_DAYS,
                    "severity": metrics.severity,
                }
            )

            # 8. Save DriftDetectionResult to database
            result_id = _save_drift_result(
                conn,
                run_id,
                metrics,
                baseline_start,
                baseline_end,
                eval_start,
                eval_end,
                thresholds.cosine_warning,
            )
            logger.info("saved drift result: %s", result_id)

            # 9. If drift detected, publish alert and update cooldown
            if metrics.drift_detected:
                logger.warning(
                    "DRIFT DETECTED: cosine_shift=%.4f, severity=%s",
                    cosine_shift,
                    metrics.severity,
                )

                alert_data = {
                    "event_type": "chora.ml.drift_detected",
                    "run_id": run_id,
                    "result_id": result_id,
                    "cosine_shift": round(cosine_shift, 4),
                    "psi_score": round(metrics.psi_score, 4),
                    "js_divergence": round(metrics.js_divergence, 4),
                    "severity": metrics.severity,
                    "sample_count_baseline": metrics.sample_count_baseline,
                    "sample_count_evaluation": metrics.sample_count_evaluation,
                    "baseline_window": f"{baseline_start.isoformat()}..{baseline_end.isoformat()}",
                    "evaluation_window": f"{eval_start.isoformat()}..{eval_end.isoformat()}",
                    "threshold_used": thresholds.cosine_warning,
                    "detected_at": now.isoformat(),
                }

                publish_drift_alert(config, alert_data)
                _update_last_triggered(conn, now)

                mlflow.set_tag("drift.alert_published", "true")
                mlflow.set_tag("drift.severity", metrics.severity)
            else:
                logger.info("no significant drift detected (cosine_shift=%.4f)", cosine_shift)
                mlflow.set_tag("drift.alert_published", "false")

        logger.info("drift_detector completed successfully")
        return 0

    except Exception:
        logger.exception("drift_detector failed")
        return 1

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
