"""Alerting utilities — publish ML pipeline alerts to NATS JetStream.

Drift detection and model evaluation jobs publish alert events to the
``chora.ml.alerts`` NATS subject for chora-communication to pick up and
deliver via configured notification channels (email, Slack, etc).

Subject: chora.ml.alerts

The event envelope is published as message HEADERS (event_type, source,
timestamp) so subscribers can filter without parsing the payload, mirroring
the Pub/Sub attributes contract. The JSON envelope is the message body.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from chora_ml_jobs.config import Config

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ALERT_SUBJECT = "chora.ml.alerts"


# ── NATS Publisher ──────────────────────────────────────────────────────────


def _publish_event(
    config: Config,
    event_type: str,
    data: dict[str, Any],
) -> str | None:
    """Publish an alert event to the ML alerts NATS subject.

    Args:
        config: Application configuration.
        event_type: Event type identifier (e.g., 'chora.ml.drift_detected').
        data: Event payload.

    Returns:
        Published message ID, or None if publishing failed.
    """
    try:
        import nats

        envelope = {
            "event_type": event_type,
            "source": "chora-ml-jobs",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": data,
        }

        message_bytes = json.dumps(envelope, default=str).encode("utf-8")
        headers = {
            "event_type": event_type,
            "source": "chora-ml-jobs",
            "timestamp": envelope["timestamp"],
        }

        async def _send() -> str:
            # Bounded reconnect budget: a batch job must not hang for minutes
            # when the broker is unreachable — fail fast, log, and continue.
            nc = await nats.connect(
                config.nats_url,
                connect_timeout=2.0,
                reconnect_time_wait=1.0,
                max_reconnect_attempts=5,
            )
            try:
                js = nc.jetstream()
                ack = await js.publish(ALERT_SUBJECT, message_bytes, headers=headers)
                return f"{ack.stream}-{ack.seq}"
            finally:
                await nc.close()

        message_id = asyncio.run(_send())

        logger.info(
            "published alert event: %s (message_id=%s)",
            event_type,
            message_id,
        )
        return message_id

    except Exception:
        logger.exception(
            "failed to publish alert event: %s (subject=%s)",
            event_type,
            ALERT_SUBJECT,
        )
        return None


# ── Public API ──────────────────────────────────────────────────────────────


def publish_drift_alert(
    config: Config,
    alert_data: dict[str, Any],
) -> str | None:
    """Publish a drift detection alert.

    Called by drift_detector when cosine shift exceeds the warning threshold.

    Args:
        config: Application configuration.
        alert_data: Drift detection metrics and metadata.

    Returns:
        Published message ID, or None if publishing failed.
    """
    severity = alert_data.get("severity", "unknown")
    cosine_shift = alert_data.get("cosine_shift", 0.0)

    logger.warning(
        "publishing drift alert: severity=%s, cosine_shift=%.4f",
        severity,
        cosine_shift,
    )

    return _publish_event(
        config,
        event_type="chora.ml.drift_detected",
        data={
            "alert_type": "drift_detection",
            "severity": severity,
            "summary": f"Embedding drift detected (cosine shift: {cosine_shift:.4f}, severity: {severity})",
            **alert_data,
        },
    )


def publish_eval_regression_alert(
    config: Config,
    alert_data: dict[str, Any],
) -> str | None:
    """Publish a model evaluation regression alert.

    Called by eval_runner when a model's score drops more than 5 points
    below its baseline.

    Args:
        config: Application configuration.
        alert_data: Evaluation results and regression details.

    Returns:
        Published message ID, or None if publishing failed.
    """
    model_id = alert_data.get("model_id", "unknown")
    regression_amount = alert_data.get("regression_amount", 0.0)

    logger.warning(
        "publishing eval regression alert: model=%s, regression=%.1f points",
        model_id,
        regression_amount,
    )

    return _publish_event(
        config,
        event_type="chora.ml.model_eval_regression",
        data={
            "alert_type": "model_eval_regression",
            "severity": "high" if regression_amount > 10.0 else "medium",
            "summary": f"Model {model_id} score dropped {regression_amount:.1f} points below baseline",
            **alert_data,
        },
    )
