"""Configuration for chora-ml-jobs — loaded from environment variables.

All ML jobs share the same configuration. Environment variables are the sole
configuration source, matching the 12-factor app pattern used across Chora services.

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URL (default: http://localhost:5000)
    MLFLOW_EXPERIMENT_NAME: MLflow experiment name (default: chora-ml-jobs)
    DATABASE_URL: PostgreSQL connection string for the Familiar DB
    CHORA_NATS_URL: NATS broker URL for alert events (default: nats://localhost:4222)
    LOG_LEVEL: Logging level (default: INFO)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable configuration loaded from environment variables."""

    # MLflow
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    # Database (Familiar DB — contains embeddings and persona data)
    database_url: str

    # NATS JetStream (alert event publishing)
    nats_url: str

    # Logging
    log_level: str

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.

        Returns:
            Config: Validated, immutable configuration object.

        Raises:
            ValueError: If required environment variables are missing.
        """
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            msg = "DATABASE_URL environment variable is required"
            raise ValueError(msg)

        return cls(
            mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            mlflow_experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "chora-ml-jobs"),
            database_url=database_url,
            nats_url=os.environ.get("CHORA_NATS_URL", "nats://localhost:4222"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


def setup_logging(config: Config) -> logging.Logger:
    """Configure structured logging for ML jobs.

    Args:
        config: Application configuration.

    Returns:
        Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    return logging.getLogger("chora_ml_jobs")
