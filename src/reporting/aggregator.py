"""
News aggregator — logic for condensing multiple article analyses into a single overview.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsAggregator:
    """
    Aggregates individual article analyses into themes and metrics.
    """

    def __init__(self, sections: list[str] | None = None) -> None:
        self.sections = sections or [
            "Top Developments",
            "Key Signals",
            "Technology Highlights",
            "Market Signals",
            "Geopolitical Events",
        ]

    def aggregate(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Produce an aggregated overview of the provided article reports.
        """
        logger.info("Aggregating %d article reports", len(reports))

        total_sentiment = Counter()
        all_entities = Counter()
        section_mapping: dict[str, list[dict[str, Any]]] = {s: [] for s in self.sections}

        for report in reports:
            analysis = report.get("analysis", {})

            # 1. Sentiment aggregation
            sentiment = analysis.get("sentiment", "Neutral")
            if "Positive" in sentiment:
                total_sentiment["Positive"] += 1
            elif "Negative" in sentiment:
                total_sentiment["Negative"] += 1
            else:
                total_sentiment["Neutral"] += 1

            # 2. Entity aggregation
            entities_str = analysis.get("entities", "")
            # Simple parsing: assume comma separated or bulleted
            for entity in self._parse_entities(entities_str):
                all_entities[entity] += 1

            # 3. Section assignment (Simple keyword matching)
            assigned_section = self._assign_section(report)
            if assigned_section not in section_mapping:
                assigned_section = self.sections[0]
            section_mapping[assigned_section].append(report)

        return {
            "total_articles": len(reports),
            "sentiment_distribution": dict(total_sentiment),
            "top_entities": all_entities.most_common(10),
            "sections": section_mapping,
        }

    def _parse_entities(self, entities_text: str) -> list[str]:
        """Extract a list of entities from the raw LLM output text."""
        if not entities_text or "Error" in entities_text:
            return []

        entities = []
        # Handle bullet points or commas
        lines = entities_text.split("\n")
        for line in lines:
            line = line.strip("- ").strip("* ").strip()
            if not line:
                continue
            # Further split by commas if they exist on a single line
            for part in line.split(","):
                part = part.strip()
                if part and len(part) > 2:
                    entities.append(part)
        return entities

    def _assign_section(self, report: dict[str, Any]) -> str:
        """Assign an article to a thematic section based on content."""
        title = report.get("title", "").lower()
        summary = report.get("analysis", {}).get("summarize", "").lower()
        content = f"{title} {summary}"

        keywords = {
            "Technology Highlights": ["tech", "ai", "software", "chip", "robot", "crypto", "internet"],
            "Market Signals": ["market", "stock", "economy", "fed", "inflation", "trade", "price", "bank"],
            "Geopolitical Events": ["war", "election", "conflict", "diplomacy", "nato", "eu", "border", "protest"],
            "Key Signals": ["critical", "major", "warning", "breakthrough", "urgent"],
        }

        for section, keys in keywords.items():
            if any(k in content for k in keys):
                return section

        return "Top Developments"  # Default section
