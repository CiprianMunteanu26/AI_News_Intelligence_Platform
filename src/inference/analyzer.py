"""
News analyzer — high-level logic for processing articles into intelligence reports.
"""

from __future__ import annotations

from typing import Any

from src.inference.engine import InferenceEngine
from src.inference.prompts import build_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsAnalyzer:
    """
    Orchestrates multiple analysis tasks for a news article.
    """

    def __init__(self, engine: InferenceEngine, tasks: list[str] | None = None) -> None:
        self.engine = engine
        self.tasks = tasks or ["summarize", "sentiment", "impact", "entities"]

    def analyze_article(self, article: dict[str, Any]) -> dict[str, Any]:
        """
        Run all configured analysis tasks on a single article.
        """
        title = article.get("title", "No Title")
        body = article.get("body", "")
        url = article.get("url", "No URL")

        logger.info("Analyzing article: %s", title)

        results: dict[str, Any] = {"title": title, "url": url, "analysis": {}}

        for task in self.tasks:
            try:
                logger.debug("Running task: %s", task)
                prompt = build_prompt(task, title, body)
                response = self.engine.generate(prompt)
                results["analysis"][task] = response
            except Exception as e:
                logger.error("Analysis task '%s' failed for '%s': %s", task, title, e)
                results["analysis"][task] = f"Error: {e}"

        return results

    def analyze_batch(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze a list of articles.
        """
        return [self.analyze_article(a) for a in articles]
