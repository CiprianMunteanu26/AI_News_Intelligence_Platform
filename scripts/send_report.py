"""
Script to send a generated intelligence report via Telegram.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.delivery.telegram_bot import TelegramDelivery
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def run_delivery(report_path: str | None = None, config_path: str | None = None, as_file: bool = False) -> bool:
    """
    Main orchestration function for Telegram delivery.
    """
    # 1. Load config
    config = PlatformConfig.load(config_path)
    reports_dir = config.paths.reports_dir

    # 2. Identify report file
    if report_path:
        report_file_path = Path(report_path)
    else:
        # Pick the latest .md file in reports_dir
        md_files = sorted(reports_dir.glob("intelligence_report_*.md"))
        if not md_files:
            logger.error("No reports found in %s. Run scripts/generate_report.py first.", reports_dir)
            return False
        report_file_path = md_files[-1]

    if not report_file_path.exists():
        logger.error("Report file not found: %s", report_file_path)
        return False

    logger.info("Delivering report from: %s", report_file_path)

    # 3. Initialize delivery
    try:
        delivery = TelegramDelivery(token=config.telegram_bot_token, chat_id=config.telegram_chat_id)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return False

    # 4. Validate connection before sending
    if not await delivery.validate_connection():
        logger.error("Telegram validation failed. Ensure the bot is added to the chat and the ID is correct.")
        return False

    # 5. Send report
    if as_file:
        success = await delivery.send_report_file(report_file_path)
    else:
        content = await asyncio.to_thread(report_file_path.read_text, encoding="utf-8")
        success = await delivery.send_report_message(content)

    if success:
        logger.info("Report successfully delivered to Telegram!")
    else:
        logger.error("Failed to deliver report.")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI News Platform - Telegram Delivery")
    parser.add_argument("--report", type=str, help="Path to the markdown report file")
    parser.add_argument("--config", type=str, help="Path to pipeline_config.yaml")
    parser.add_argument("--file", action="store_true", help="Send the report as a file instead of a message")

    args = parser.parse_args()

    try:
        if not asyncio.run(run_delivery(args.report, args.config, args.file)):
            sys.exit(1)
    except Exception:
        logger.exception("Telegram delivery failed")
        sys.exit(1)
