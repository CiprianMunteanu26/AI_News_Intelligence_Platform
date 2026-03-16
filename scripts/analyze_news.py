"""
Script to run the intelligence analysis pipeline.

Loads the processed NewsDataset, runs LLM analysis on each article,
and saves the structured reports to data/analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import NewsDataset
from src.inference.analyzer import NewsAnalyzer
from src.inference.engine import InferenceEngine
from src.models.factory import ModelFactory
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_analysis(config_path: str | None = None, limit: int | None = None, use_tiny: bool = False) -> None:
    """
    Main orchestration function for intelligence analysis.
    """
    # 1. Load config
    config = PlatformConfig.load(config_path)

    processed_dataset_path = config.paths.processed_data_dir / "latest_news_dataset"
    analysis_dir = config.paths.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting intelligence analysis pipeline")
    logger.info("Dataset path: %s", processed_dataset_path)
    logger.info("Output dir: %s", analysis_dir)

    # 2. Load model
    if use_tiny:
        model, tokenizer = ModelFactory.load_tiny_fallback()
    else:
        model, tokenizer = ModelFactory.load_model(
            config.training.model_name,
            use_half_precision=config.training.fp16,
            load_in_4bit=False,  # Can be toggled if needed
        )

    # 3. Initialize engine and analyzer
    engine = InferenceEngine(
        model, tokenizer, max_new_tokens=config.inference.max_new_tokens, temperature=config.inference.temperature
    )
    analyzer = NewsAnalyzer(engine, tasks=config.inference.tasks)

    # 4. Load dataset
    if not processed_dataset_path.exists():
        logger.error("Processed dataset not found at %s. Run preprocessing first.", processed_dataset_path)
        return

    dataset = NewsDataset.load(processed_dataset_path)

    # 5. Run analysis
    num_articles = len(dataset)
    if limit:
        num_articles = min(num_articles, limit)

    logger.info("Analyzing %d articles...", num_articles)

    reports = []
    for i in range(num_articles):
        article = dataset[i]
        report = analyzer.analyze_article(article)
        reports.append(report)

        # Log progress periodically
        if (i + 1) % 5 == 0:
            logger.info("Progress: %d/%d", i + 1, num_articles)

    # 6. Save results
    from datetime import timezone

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_file = analysis_dir / f"intelligence_report_{date_str}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    logger.info("Analysis complete. Saved %d reports to %s", len(reports), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News Platform - Intelligence Analysis")
    parser.add_argument("--config", type=str, help="Path to pipeline_config.yaml")
    parser.add_argument("--limit", type=int, help="Limit the number of articles to analyze")
    parser.add_argument("--tiny", action="store_true", help="Use tiny fallback model for testing")

    args = parser.parse_args()

    try:
        run_analysis(args.config, args.limit, args.tiny)
    except Exception:
        logger.exception("Analysis pipeline failed")
        exit(1)
