from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts.run_pipeline import run_full_pipeline


class PipelineFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_failure_stops_before_delivery(self) -> None:
        reporting = types.ModuleType("scripts.generate_report")
        reporting.run_reporting = Mock(return_value=False)
        delivery = types.ModuleType("scripts.send_report")
        delivery.run_delivery = AsyncMock(return_value=True)

        with (
            patch.dict(sys.modules, {"scripts.generate_report": reporting, "scripts.send_report": delivery}),
            self.assertRaises(SystemExit) as raised,
        ):
            await run_full_pipeline(steps=["report", "deliver"])

        self.assertEqual(raised.exception.code, 1)
        delivery.run_delivery.assert_not_awaited()

    async def test_delivery_failure_is_reported_as_pipeline_failure(self) -> None:
        reporting = types.ModuleType("scripts.generate_report")
        reporting.run_reporting = Mock(return_value=True)
        delivery = types.ModuleType("scripts.send_report")
        delivery.run_delivery = AsyncMock(return_value=False)

        with (
            patch.dict(sys.modules, {"scripts.generate_report": reporting, "scripts.send_report": delivery}),
            self.assertRaises(SystemExit) as raised,
        ):
            await run_full_pipeline(steps=["report", "deliver"])

        self.assertEqual(raised.exception.code, 1)
        delivery.run_delivery.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
