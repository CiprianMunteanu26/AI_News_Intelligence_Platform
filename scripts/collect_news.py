"""
CLI entry point for news collection.

Usage:
    python scripts/collect_news.py [--config CONFIG_PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.collector import NewsCollector
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_collection(config_path: str | None = None, output_dir_path: str | None = None) -> None:
    """
    Main orchestration function for news collection.
    """
    # Load config
    config = PlatformConfig.load(config_path)
    output_dir = Path(output_dir_path) if output_dir_path else None

    # Collect
    logger.info("Starting news collection...")
    collector = NewsCollector(config)
    articles = collector.collect(output_dir=output_dir)
    logger.info("Collection complete — %d articles saved.", len(articles))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect news articles from all configured sources.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for raw articles (default: data/raw)",
    )
    args = parser.parse_args()

    run_collection(args.config, args.output_dir)


if __name__ == "__main__":
    main()
