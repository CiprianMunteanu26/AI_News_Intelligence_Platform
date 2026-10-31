"""
Script to run the intelligence analysis pipeline.

Loads the processed NewsDataset, runs LLM analysis on each article,
and saves the structured reports to data/analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import NewsDataset
from src.inference.analyzer import NewsAnalyzer
from src.inference.bonsai_runtime import BonsaiRuntime
from src.inference.engine import InferenceEngine
from src.inference.protocols import TextGenerator
from src.models.factory import ModelFactory
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_analysis(
    config_path: str | None = None,
    limit: int | None = None,
    use_tiny: bool = False,
    check_runtime: bool = False,
) -> None:
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

    if use_tiny and check_runtime:
        raise ValueError("--check-runtime cannot be combined with --tiny.")

    engine: TextGenerator
    if not use_tiny:
        engine = BonsaiRuntime(
            base_url=config.inference.runtime_url,
            max_new_tokens=config.inference.max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
            presence_penalty=config.inference.presence_penalty,
            timeout_seconds=config.inference.timeout_seconds,
            enable_thinking=config.inference.enable_thinking,
        )
        if check_runtime:
            engine.check_health()
            return

    if not processed_dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_dataset_path}. Run preprocessing first.")

    # 2. Load model
    if use_tiny:
        model, tokenizer = ModelFactory.load_tiny_fallback()
        engine = InferenceEngine(
            model,
            tokenizer,
            max_new_tokens=config.inference.max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
        )

    # 3. Initialize analyzer
    analyzer = NewsAnalyzer(engine, tasks=config.inference.tasks)

    # 4. Load dataset
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
    date_str = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    output_file = analysis_dir / f"intelligence_report_{date_str}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    logger.info("Analysis complete. Saved %d reports to %s", len(reports), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News Platform - Intelligence Analysis")
    parser.add_argument("--config", type=str, help="Path to pipeline_config.yaml")
    parser.add_argument("--limit", type=int, help="Limit the number of articles to analyze")
    parser.add_argument("--tiny", action="store_true", help="Use the tiny Transformers fallback instead of Bonsai")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Check that the configured local Bonsai llama.cpp server is ready, then exit",
    )

    args = parser.parse_args()

    try:
        run_analysis(args.config, args.limit, args.tiny, args.check_runtime)
    except Exception:
        logger.exception("Analysis pipeline failed")
        sys.exit(1)
