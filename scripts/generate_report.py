"""
Script to generate an intelligence report from the analyzed news articles.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting.aggregator import NewsAggregator
from src.reporting.report_generator import ReportGenerator
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_reporting(
    input_file: str | None = None, config_path: str | None = None, output_file: str | None = None
) -> bool:
    """
    Main orchestration function for report generation.
    """
    # 1. Load config
    config = PlatformConfig.load(config_path)
    analysis_dir = config.paths.analysis_dir
    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Identify input file
    if input_file:
        input_file_path = Path(input_file)
    else:
        # Pick the latest .json file in analysis_dir
        json_files = sorted(analysis_dir.glob("intelligence_report_*.json"))
        if not json_files:
            logger.error("No analysis results found in %s. Run scripts/analyze_news.py first.", analysis_dir)
            return False
        input_file_path = json_files[-1]

    if not input_file_path.exists():
        logger.error("Input file not found: %s", input_file_path)
        return False

    logger.info("Generating report from: %s", input_file_path)

    # 3. Load reports
    with open(input_file_path, "r", encoding="utf-8") as f:
        reports = json.load(f)

    # 4. Aggregation
    aggregator = NewsAggregator(sections=config.reporting.sections)
    agg_results = aggregator.aggregate(reports)

    # 5. Generation
    generator = ReportGenerator()
    markdown_content = generator.generate_markdown(agg_results)

    # 6. Save report
    if not output_file:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file_path = reports_dir / f"intelligence_report_{date_str}.md"
    else:
        output_file_path = Path(output_file)

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info("Intelligence report successfully generated at: %s", output_file_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News Platform - Report Generation")
    parser.add_argument("--input", type=str, help="Path to input analysis JSON file")
    parser.add_argument("--config", type=str, help="Path to pipeline_config.yaml")
    parser.add_argument("--output", type=str, help="Path to output markdown file")

    args = parser.parse_args()

    try:
        if not run_reporting(args.input, args.config, args.output):
            sys.exit(1)
    except Exception:
        logger.exception("Report generation failed")
        sys.exit(1)
