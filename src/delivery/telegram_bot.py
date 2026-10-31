"""
Telegram delivery — interface for sending intelligence reports via Telegram.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramDelivery:
    """
    Handles delivery of reports to a Telegram chat.
    """

    def __init__(self, token: str, chat_id: str) -> None:
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be provided.")
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def validate_connection(self) -> bool:
        """
        Verify that the token is valid and the bot can access the chat.
        """
        try:
            me = await self.bot.get_me()
            logger.info("Connected as bot: @%s", me.username)
            # Try to get chat info to verify chat_id
            await self.bot.get_chat(chat_id=self.chat_id)
            return True
        except TelegramError as e:
            logger.error("Telegram validation failed: %s", e)
            return False

    async def send_report_message(self, text: str) -> bool:
        """
        Send a text report to the configured chat.
        Automatically splits messages that exceed the 4096 character limit.
        """
        if not text:
            logger.warning("Empty report text. Nothing to send.")
            return False

        # Telegram limit is 4096. We use 4000 to be safe.
        MAX_LENGTH = 4000
        chunks = [text[i : i + MAX_LENGTH] for i in range(0, len(text), MAX_LENGTH)]

        try:
            for i, chunk in enumerate(chunks):
                logger.info("Sending chunk %d/%d to Telegram...", i + 1, len(chunks))
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True,
                )
            return True
        except TelegramError as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    async def send_report_file(self, file_path: Path, caption: str | None = None) -> bool:
        """
        Send the report as a file document.
        """
        if not file_path.exists():
            logger.error("Report file not found: %s", file_path)
            return False

        try:
            logger.info("Sending report file %s to Telegram...", file_path.name)
            document = await asyncio.to_thread(file_path.read_bytes)
            await self.bot.send_document(
                chat_id=self.chat_id,
                document=document,
                filename=file_path.name,
                caption=caption or f"Intelligence Report: {file_path.stem}",
            )
            return True
        except (OSError, TelegramError) as e:
            logger.error("Failed to send Telegram document: %s", e)
            return False
