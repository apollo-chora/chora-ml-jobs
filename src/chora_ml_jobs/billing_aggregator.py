"""Billing Aggregator — aggregates ML token usage per tenant for billing periods.

This job queries the token usage records from the Familiar DB, aggregates them
per tenant and billing period, and publishes the results to the billing service
via NATS JetStream events.

Implementation will be completed in Stream C (ML Pipeline).

Metrics tracked in MLflow:
    - Total tokens processed per billing period
    - Token usage distribution across tenants
    - Cost allocation accuracy (reconciliation with provider invoices)
    - Aggregation job duration and row counts
"""

from __future__ import annotations

import logging
import sys

from chora_ml_jobs.config import Config, setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the billing aggregation job.

    Returns:
        int: Exit code (0 = success, 1 = error).
    """
    config = Config.from_env()
    setup_logging(config)

    logger.info(
        "billing_aggregator starting",
        extra={
            "mlflow_uri": config.mlflow_tracking_uri,
            "nats_url": config.nats_url,
        },
    )

    # TODO(stream-c): Implement billing aggregation pipeline
    # 1. Query token_usage_records from Familiar DB for current billing period
    # 2. Aggregate by tenant_id, model_provider, model_name
    # 3. Compute cost estimates based on provider pricing tables
    # 4. Log aggregation metrics to MLflow
    # 5. Publish chora.billing.token_usage_aggregated events per tenant via NATS
    # 6. Store aggregation report as MLflow artifact

    logger.info("billing_aggregator completed (stub — no-op)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
