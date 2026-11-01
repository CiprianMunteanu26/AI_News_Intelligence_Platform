"""
News analyzer — high-level logic for processing articles into intelligence reports.
"""

from __future__ import annotations

import re
from typing import Any

from src.inference.prompts import build_prompt
from src.inference.protocols import TextGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsAnalyzer:
    """
    Orchestrates multiple analysis tasks for a news article.
    """

    def __init__(
        self,
        engine: TextGenerator,
        tasks: list[str] | None = None,
        minimum_body_characters: int = 500,
    ) -> None:
        if minimum_body_characters < 1:
            raise ValueError("minimum_body_characters must be a positive integer.")
        self.engine = engine
        self.tasks = tasks or ["summarize", "sentiment", "impact", "entities"]
        self.minimum_body_characters = minimum_body_characters

    def analyze_article(self, article: dict[str, Any]) -> dict[str, Any]:
        """
        Run all configured analysis tasks on a single article.
        """
        title = article.get("title", "No Title")
        body = article.get("body") or ""
        url = article.get("url", "No URL")
        if not isinstance(body, str):
            raise TypeError("Article body must be a string.")

        logger.info("Analyzing article: %s", title)

        results: dict[str, Any] = {"title": title, "url": url, "analysis": {}}

        if len(body.strip()) < self.minimum_body_characters:
            logger.info(
                "Abstaining from model analysis for %s: article body has %d characters; %d required",
                title,
                len(body.strip()),
                self.minimum_body_characters,
            )
            results["analysis"] = {
                task: _abstention(task, len(body.strip()), self.minimum_body_characters) for task in self.tasks
            }
            return results

        for task in self.tasks:
            logger.debug("Running task: %s", task)
            prompt = build_prompt(task, title, body)
            results["analysis"][task] = _clean_model_output(task, self.engine.generate(prompt))

        return results

    def analyze_batch(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Analyze a list of articles.
        """
        return [self.analyze_article(a) for a in articles]


def _abstention(task: str, body_length: int, minimum: int) -> str:
    """Return a deterministic, task-shaped result when source evidence is too short."""
    reason = f"Insufficient source text ({body_length} characters; {minimum} required)."
    if task == "sentiment":
        return f"Sentiment: Not assessed | Evidence: {reason}"
    if task == "impact":
        return f"Impact Score: Not assessed | Description: {reason}"
    if task == "entities":
        return f"None identified. Analysis abstained: {reason}"
    return f"Analysis abstained: {reason}"


def _clean_model_output(task: str, output: str) -> str:
    """Remove explicit reasoning blocks and constrain responses to the requested shape."""
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.IGNORECASE | re.DOTALL)
    output = re.sub(r"<think>.*$", "", output, flags=re.IGNORECASE | re.DOTALL)
    output = re.sub(r"</?think>", "", output, flags=re.IGNORECASE).strip()
    if not output:
        raise RuntimeError(f"The model returned no usable {task} output after reasoning cleanup.")

    if task == "sentiment":
        match = re.search(
            r"Sentiment:\s*(Positive|Negative|Neutral)\s*(?:\|\s*(?:Evidence|Reasoning):\s*(.*))?",
            output,
            flags=re.IGNORECASE,
        )
        if not match:
            raise RuntimeError("The model returned a malformed sentiment response.")
        label = match.group(1).capitalize()
        evidence = _first_line(match.group(2) or "No supporting evidence provided.")
        return f"Sentiment: {label} | Evidence: {evidence}"

    if task == "impact":
        match = re.search(
            r"Impact Score:\s*(\d{1,2})\s*\|\s*Description:\s*(.*)",
            output,
            flags=re.IGNORECASE,
        )
        if not match or not 1 <= int(match.group(1)) <= 10:
            raise RuntimeError("The model returned a malformed impact response.")
        description = _first_line(match.group(2))
        return f"Impact Score: {int(match.group(1))} | Description: {description}"

    if task == "entities":
        line = _first_line(output)
        line = re.sub(r"^Entities:\s*", "", line, flags=re.IGNORECASE).strip(" -*")
        if not line or re.match(r"^(none|no entities)\b", line, flags=re.IGNORECASE):
            return "None identified."
        entities = [entity.strip(" -*.") for entity in line.split(",") if entity.strip(" -*.")]
        if not entities:
            return "None identified."
        return ", ".join(entities[:5])

    output = re.sub(r"^Summary:\s*", "", output, count=1, flags=re.IGNORECASE).strip()
    sentences = re.split(r"(?<=[.!?])\s+", output)
    return " ".join(sentences[:3]).strip()


def _first_line(value: str) -> str:
    """Return a single whitespace-normalized line capped to a concise length."""
    lines = value.splitlines()
    line = re.sub(r"\s+", " ", lines[0] if lines else "").strip(" |")
    if not line:
        raise RuntimeError("The model returned an empty evidence or description field.")
    return line[:500]
