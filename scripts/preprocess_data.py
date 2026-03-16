"""
Script to run the news article preprocessing pipeline.

Loads raw articles from data/raw, cleans them using NewsPreprocessor,
and saves the result as a NewsDataset in data/processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import NewsDataset
from src.data.preprocess import NewsPreprocessor
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing(config_path: str | None = None) -> None:
    """
    Main orchestration function for preprocessing.
    """
    # 1. Load config
    config = PlatformConfig.load(config_path)

    raw_dir = config.paths.raw_data_dir
    processed_dir = config.paths.processed_data_dir

    logger.info("Starting preprocessing pipeline")
    logger.info("Raw data dir: %s", raw_dir)
    logger.info("Processed data dir: %s", processed_dir)

    # 2. Initialize preprocessor
    preprocessor = NewsPreprocessor()

    # 3. Load and preprocess articles into a NewsDataset
    dataset = NewsDataset.from_json_files(raw_dir, preprocessor=preprocessor)

    if len(dataset) == 0:
        logger.warning("No articles were found/processed. Skipping save.")
        return

    logger.info("Processed %d articles", len(dataset))

    # 4. Save to disk
    # We save as a daily version or a 'latest' version
    # For now, let's save to processed_data_dir / "latest_news_dataset"
    output_path = processed_dir / "latest_news_dataset"
    dataset.save(output_path)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News Platform - Preprocessing Pipeline")
    parser.add_argument("--config", type=str, help="Path to pipeline_config.yaml")

    args = parser.parse_args()

    try:
        run_preprocessing(args.config)
    except Exception:
        logger.exception("Preprocessing pipeline failed")
        exit(1)
