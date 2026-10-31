"""
One-off test script to verify REAL Telegram connectivity using your .env credentials.
Usage:
    poetry run python scripts/test_telegram_connection.py
"""

import asyncio
import sys
from pathlib import Path

from telegram.error import TelegramError

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.delivery.telegram_bot import TelegramDelivery
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def test_real_connection():
    logger.info("Loading configuration from .env...")
    config = PlatformConfig.load()

    if not config.telegram_bot_token or not config.telegram_chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file!")
        return

    logger.info("Initializing Telegram bot...")
    try:
        delivery = TelegramDelivery(
            token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id
        )

        # Validate connection
        if not await delivery.validate_connection():
            logger.error("Validation failed! Check bot token and chat_id in .env.")
            return

        # Send test message
        test_msg = "🚀 *AI News Platform: Connection Test*\nYour Telegram integration is working perfectly!"
        logger.info("Sending test message...")
        success = await delivery.send_report_message(test_msg)

        if success:
            logger.info("Connection test PASSED! Check your Telegram.")
        else:
            logger.error("Connection test FAILED to send message.")

    except TelegramError as e:
        logger.error(f"Telegram connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_connection())
