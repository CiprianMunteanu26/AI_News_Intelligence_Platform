from __future__ import annotations

import unittest

from src.reporting.aggregator import NewsAggregator
from src.reporting.report_generator import ReportGenerator


class NewsAggregatorTests(unittest.TestCase):
    def test_custom_sections_receive_unmatched_reports(self) -> None:
        report = {"title": "Routine update", "analysis": {"sentiment": "Neutral"}}

        result = NewsAggregator(sections=["My Briefing"]).aggregate([report])

        self.assertEqual(result["sections"]["My Briefing"], [report])
        self.assertEqual(result["total_articles"], 1)

    def test_report_contains_source_link_and_summary(self) -> None:
        report = {
            "title": "Chip investment announced",
            "url": "https://example.com/story",
            "analysis": {
                "summarize": "The company announced a new chip investment.",
                "sentiment": "Positive",
                "entities": "Example Corp, AI",
            },
        }
        aggregate = NewsAggregator().aggregate([report])

        markdown = ReportGenerator().generate_markdown(aggregate)

        self.assertIn("https://example.com/story", markdown)
        self.assertIn("The company announced a new chip investment.", markdown)
        self.assertIn("Example Corp", markdown)


if __name__ == "__main__":
    unittest.main()
