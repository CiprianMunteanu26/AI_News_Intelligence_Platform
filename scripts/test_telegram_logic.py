"""Offline tests for Telegram delivery behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

# Make direct execution from the repository root work as well as module execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.delivery.telegram_bot import TelegramDelivery


class TelegramDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_message_is_split_into_safe_chunks(self) -> None:
        delivery = TelegramDelivery("fake_token", "fake_chat_id")
        delivery.bot = AsyncMock()

        success = await delivery.send_report_message("A" * 5000)

        self.assertTrue(success)
        self.assertEqual(delivery.bot.send_message.call_count, 2)
        self.assertEqual(
            [len(call.kwargs["text"]) for call in delivery.bot.send_message.call_args_list],
            [4000, 1000],
        )


if __name__ == "__main__":
    unittest.main()
