"""
AADS Command Line Interface (CLI) — entry point for running autonomous data science projects.

Usage:
    python -m aads.cli --data path/to/dataset.csv --objective "Predict customer churn" --target churn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aads.agents.orchestrator import AADSOrchestrator
from aads.core.config import AADSConfig
from aads.core.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Autonomous AI Data Scientist (AADS) — End-to-End ML Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to the input dataset (CSV, XLSX, or Parquet).",
    )
    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default="Analyze the dataset and build a high-performing predictive model.",
        help="Natural language goal or task objective.",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default=None,
        help="Target column name for supervised learning tasks (optional; auto-detected if omitted).",
    )
    parser.add_argument(
        "--storage",
        "-s",
        type=str,
        default="storage/runs",
        help="Base directory for project run outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    data_file = Path(args.data)

    if not data_file.exists():
        print(f"Error: Data file not found at '{data_file}'")
        return 1

    print("\n" + "=" * 70)
    print("AADS: Autonomous AI Data Scientist — Launching Pipeline")
    print("=" * 70)
    print(f"Dataset:   {data_file.resolve()}")
    print(f"Objective: {args.objective}")
    if args.target:
        print(f"Target:    {args.target}")
    print(f"Storage:   {Path(args.storage).resolve()}")
    print(f"Seed:      {args.seed}")
    print("=" * 70 + "\n")

    config = AADSConfig(
        storage_root=Path(args.storage),
        random_seed=args.seed,
    )

    orchestrator = AADSOrchestrator(config=config, storage_root=args.storage)

    try:
        result = orchestrator.run_pipeline(
            data_path=data_file,
            user_objective=args.objective,
            target_column=args.target,
        )

        print("\n" + "=" * 70)
        print("Pipeline Completed Successfully!")
        print("=" * 70)
        print(f"Run ID:        {result['run_id']}")
        print(f"Best Model:    {result['best_model_name']}")
        print(f"Best Metrics:  {result['best_metrics']}")
        print(f"Artifacts:     {result['total_artifacts']} files generated")
        print(f"Output Folder: {result['run_dir']}")
        print("=" * 70)
        print("\nExecutive Summary Preview:")
        print("-" * 50)
        print(result["executive_summary"][:600] + "...\n")
        return 0

    except Exception as e:
        logger.exception("pipeline_execution_error", error=str(e))
        print(f"\nPipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
