from __future__ import annotations

import unittest

from src.inference.analyzer import NewsAnalyzer


class FailingGenerator:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("local model runtime unavailable")


class NewsAnalyzerTests(unittest.TestCase):
    def test_inference_failure_stops_analysis(self) -> None:
        analyzer = NewsAnalyzer(FailingGenerator(), tasks=["summarize"])

        with self.assertRaisesRegex(RuntimeError, "runtime unavailable"):
            analyzer.analyze_article({"title": "Example", "body": "Article body", "url": "https://example.com"})


if __name__ == "__main__":
    unittest.main()
