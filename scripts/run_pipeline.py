"""
Master orchestrator script for the AI News Intelligence Platform.
Runs the end-to-end pipeline: Collection -> Preprocessing -> Analysis -> Reporting -> Delivery.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


async def run_full_pipeline(
    config_path: str | None = None,
    steps: list[str] | None = None,
    limit: int | None = None,
    use_tiny: bool = False,
    as_file: bool = False,
) -> None:
    """
    Orchestrates the entire news intelligence workflow.
    """
    logger.info("================================================================")
    logger.info("AI News Intelligence Platform — Pipeline Execution started")
    logger.info("================================================================")

    # Define all possible steps
    all_steps = ["collect", "preprocess", "analyze", "report", "deliver"]
    active_steps = steps if steps else all_steps

    try:
        # 1. Collection
        if "collect" in active_steps:
            from scripts.collect_news import run_collection

            logger.info("--- Step 1: Collection ---")
            run_collection(config_path)

        # 2. Preprocessing
        if "preprocess" in active_steps:
            from scripts.preprocess_data import run_preprocessing

            logger.info("--- Step 2: Preprocessing ---")
            run_preprocessing(config_path)

        # 3. Analysis
        if "analyze" in active_steps:
            from scripts.analyze_news import run_analysis

            logger.info("--- Step 3: Intelligence Analysis ---")
            run_analysis(config_path, limit=limit, use_tiny=use_tiny)

        # 4. Reporting
        if "report" in active_steps:
            from scripts.generate_report import run_reporting

            logger.info("--- Step 4: Report Generation ---")
            if not run_reporting(config_path=config_path):
                raise RuntimeError("Report generation did not produce a report.")

        # 5. Delivery
        if "deliver" in active_steps:
            from scripts.send_report import run_delivery

            logger.info("--- Step 5: Telegram Delivery ---")
            if not await run_delivery(config_path=config_path, as_file=as_file):
                raise RuntimeError("Telegram delivery failed.")

        logger.info("================================================================")
        logger.info("Pipeline Execution Completed Successfully!")
        logger.info("================================================================")

    except Exception:
        logger.exception("Pipeline execution failed at one of the steps.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI News Platform - Master Pipeline Orchestrator")
    parser.add_argument("--config", type=str, help="Path to pipeline_config.yaml")

    # Step selection
    parser.add_argument("--collect", action="append_const", dest="steps", const="collect", help="Run collection step")
    parser.add_argument(
        "--preprocess", action="append_const", dest="steps", const="preprocess", help="Run preprocessing step"
    )
    parser.add_argument("--analyze", action="append_const", dest="steps", const="analyze", help="Run analysis step")
    parser.add_argument("--report", action="append_const", dest="steps", const="report", help="Run reporting step")
    parser.add_argument("--deliver", action="append_const", dest="steps", const="deliver", help="Run delivery step")

    # Analysis options
    parser.add_argument("--limit", type=int, help="Limit number of articles to analyze")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model for testing analysis")

    # Delivery options
    parser.add_argument("--file", action="store_true", help="Send report as file instead of message")

    args = parser.parse_args()

    # If no steps specified, run all
    steps_to_run = args.steps if args.steps else None

    asyncio.run(
        run_full_pipeline(
            config_path=args.config,
            steps=steps_to_run,
            limit=args.limit,
            use_tiny=args.tiny,
            as_file=args.file,
        )
    )


if __name__ == "__main__":
    main()
