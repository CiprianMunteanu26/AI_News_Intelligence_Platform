"""
Report generator — formatting aggregated intelligence into Markdown reports.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Transforms aggregated article data into a human-readable Markdown report.
    """

    def __init__(self, title: str = "Daily Intelligence Briefing") -> None:
        self.title = title

    def generate_markdown(self, agg_results: dict[str, Any]) -> str:
        """
        Generate a Markdown string from the aggregated output of NewsAggregator.
        """
        date_str = datetime.now(UTC).strftime("%A, %B %d, %Y")

        md = f"# {self.title}\n"
        md += f"*Generated on {date_str}*\n\n"

        # Executive Summary Section
        md += "## 📊 Executive Summary\n"
        md += f"- **Total Articles Analyzed**: {agg_results['total_articles']}\n"

        sentiment = agg_results['sentiment_distribution']
        sentiment_summary = ", ".join([f"{k}: {v}" for k, v in sentiment.items()])
        md += f"- **Sentiment Distribution**: {sentiment_summary}\n"

        entities = ", ".join([f"{e[0]} ({e[1]})" for e in agg_results['top_entities'][:5]])
        md += f"- **Key Entities**: {entities}\n\n"

        md += "---\n\n"

        # Thematic Sections
        sections = agg_results['sections']
        for section_name, articles in sections.items():
            if not articles:
                continue

            md += f"## {self._get_section_emoji(section_name)} {section_name}\n\n"
            for article in articles:
                title = article.get("title", "Untitled")
                url = article.get("url", "#")
                summarize = article.get("analysis", {}).get("summarize", "No summary available.")
                sentiment = article.get("analysis", {}).get("sentiment", "Neutral")

                md += f"### [{title}]({url})\n"
                md += f"**Sentiment**: {sentiment}\n\n"
                md += f"{summarize}\n\n"

            md += "---\n\n"

        # Footer
        md += "*AI News Intelligence Platform - Automated Report*\n"

        return md

    def _get_section_emoji(self, name: str) -> str:
        """Map section names to emojis for better visual appeal."""
        mapping = {
            "Top Developments": "🔥",
            "Key Signals": "📡",
            "Technology Highlights": "💻",
            "Market Signals": "📈",
            "Geopolitical Events": "🌍",
        }
        return mapping.get(name, "📌")
