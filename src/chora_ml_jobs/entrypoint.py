"""Entrypoint dispatcher for chora-ml-jobs Docker container.

Reads the JOB environment variable and dispatches to the appropriate job module.
This allows a single Docker image to serve all ML batch jobs, with the job type
selected at runtime via environment variable or command override.

Usage:
    JOB=drift_detector python -m chora_ml_jobs.entrypoint
    JOB=eval_runner python -m chora_ml_jobs.entrypoint
    JOB=billing_aggregator python -m chora_ml_jobs.entrypoint
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Dispatch to the appropriate job based on the JOB environment variable.

    Returns:
        int: Exit code from the dispatched job.

    Raises:
        SystemExit: If JOB is not set or is an unknown value.
    """
    job = os.environ.get("JOB", "").strip()

    if not job:
        print("ERROR: JOB environment variable is required", file=sys.stderr)  # noqa: T201
        print("Valid values: drift_detector, eval_runner, billing_aggregator", file=sys.stderr)  # noqa: T201
        return 1

    if job == "drift_detector":
        from chora_ml_jobs.drift_detector import main as job_main
    elif job == "eval_runner":
        from chora_ml_jobs.eval_runner import main as job_main
    elif job == "billing_aggregator":
        from chora_ml_jobs.billing_aggregator import main as job_main
    else:
        print(f"ERROR: Unknown job '{job}'", file=sys.stderr)  # noqa: T201
        print("Valid values: drift_detector, eval_runner, billing_aggregator", file=sys.stderr)  # noqa: T201
        return 1

    return job_main()


if __name__ == "__main__":
    sys.exit(main())
