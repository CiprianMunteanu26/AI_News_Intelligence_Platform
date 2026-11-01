from __future__ import annotations

import unittest

from src.inference.analyzer import NewsAnalyzer


class FailingGenerator:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("local model runtime unavailable")


class RecordingGenerator:
    def __init__(self, response: str = "Summary: A verified development occurred. It has a stated effect.") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class NewsAnalyzerTests(unittest.TestCase):
    def test_inference_failure_stops_analysis(self) -> None:
        analyzer = NewsAnalyzer(FailingGenerator(), tasks=["summarize"], minimum_body_characters=1)

        with self.assertRaisesRegex(RuntimeError, "runtime unavailable"):
            analyzer.analyze_article({"title": "Example", "body": "Article body", "url": "https://example.com"})

    def test_short_article_abstains_without_calling_model(self) -> None:
        engine = RecordingGenerator()
        analyzer = NewsAnalyzer(engine, minimum_body_characters=20)

        result = analyzer.analyze_article(
            {"title": "Example", "body": "Only a teaser.", "url": "https://example.com/article"}
        )

        self.assertEqual(engine.prompts, [])
        self.assertEqual(
            result["analysis"],
            {
                "summarize": "Analysis abstained: Insufficient source text (14 characters; 20 required).",
                "sentiment": "Sentiment: Not assessed | Evidence: Insufficient source text (14 characters; 20 required).",
                "impact": "Impact Score: Not assessed | Description: Insufficient source text (14 characters; 20 required).",
                "entities": "None identified. Analysis abstained: Insufficient source text (14 characters; 20 required).",
            },
        )

    def test_analysis_removes_thinking_and_normalizes_output(self) -> None:
        engine = RecordingGenerator(
            "<think>private reasoning</think> "
            "Sentiment: Positive | Evidence: The article reports a signed agreement."
        )
        analyzer = NewsAnalyzer(engine, tasks=["sentiment"], minimum_body_characters=10)

        result = analyzer.analyze_article(
            {"title": "Example", "body": "A sufficiently long article body.", "url": "https://example.com/article"}
        )

        self.assertEqual(
            result["analysis"]["sentiment"],
            "Sentiment: Positive | Evidence: The article reports a signed agreement.",
        )

    def test_malformed_structured_output_fails_clearly(self) -> None:
        analyzer = NewsAnalyzer(
            RecordingGenerator("probably positive"),
            tasks=["sentiment"],
            minimum_body_characters=1,
        )

        with self.assertRaisesRegex(RuntimeError, "malformed sentiment"):
            analyzer.analyze_article(
                {"title": "Example", "body": "Sufficient evidence here.", "url": "https://example.com/article"}
            )

    def test_impact_output_is_constrained_to_one_line(self) -> None:
        analyzer = NewsAnalyzer(
            RecordingGenerator(
                "<think>extended private reasoning</think>"
                "Impact Score: 7 | Description: The report confirms a major factory closure.\n"
                "Additional analysis should be discarded."
            ),
            tasks=["impact"],
            minimum_body_characters=1,
        )

        result = analyzer.analyze_article(
            {"title": "Example", "body": "Sufficient evidence here.", "url": "https://example.com/article"}
        )

        self.assertEqual(
            result["analysis"]["impact"],
            "Impact Score: 7 | Description: The report confirms a major factory closure.",
        )


if __name__ == "__main__":
    unittest.main()
