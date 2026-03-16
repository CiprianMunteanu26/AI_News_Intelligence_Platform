"""
NewsAPI ingestion source.

Fetches articles from https://newsapi.org using their REST API.
Requires a valid API key set via the ``NEWSAPI_KEY`` environment variable.
"""

from __future__ import annotations

from typing import Any

import requests

from src.ingestion.base_source import Article, NewsSource
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
_BASE_URL = "https://newsapi.org/v2"
_TOP_HEADLINES = f"{_BASE_URL}/top-headlines"
_EVERYTHING = f"{_BASE_URL}/everything"


class NewsAPISource(NewsSource):
    """
    Collects articles from the NewsAPI service.

    Parameters
    ----------
    api_key : str
        NewsAPI API key.
    categories : list[str]
        News categories to fetch (e.g. ``["technology", "business"]``).
    country : str
        Two-letter country code for top-headlines (default ``"us"``).
    page_size : int
        Number of articles per request (max 100).
    """

    def __init__(
        self,
        api_key: str,
        categories: list[str] | None = None,
        country: str = "us",
        page_size: int = 50,
    ) -> None:
        self._api_key = api_key
        self._categories = categories or ["technology", "business"]
        self._country = country
        self._page_size = min(page_size, 100)

    # ------------------------------------------------------------------
    # NewsSource interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "NewsAPI"

    def fetch_articles(self) -> list[Article]:
        """Fetch top headlines across all configured categories."""
        articles: list[Article] = []

        for category in self._categories:
            try:
                raw = self._request_top_headlines(category)
                for item in raw:
                    article = self._to_article(item, category)
                    if article:
                        articles.append(article)
            except Exception:
                logger.exception("Failed to fetch category '%s' from NewsAPI", category)

        logger.info("NewsAPI: fetched %d articles across %d categories", len(articles), len(self._categories))
        return articles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request_top_headlines(self, category: str) -> list[dict[str, Any]]:
        """Call the /top-headlines endpoint for a single category."""
        params = {
            "apiKey": self._api_key,
            "category": category,
            "country": self._country,
            "pageSize": self._page_size,
        }
        resp = requests.get(_TOP_HEADLINES, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.warning("NewsAPI returned non-ok status: %s", data.get("message", "unknown"))
            return []

        return data.get("articles", [])

    @staticmethod
    def _to_article(raw: dict[str, Any], category: str) -> Article | None:
        """Convert a raw NewsAPI article dict into an Article."""
        url = raw.get("url", "")
        title = raw.get("title", "")
        body = raw.get("content") or raw.get("description") or ""

        if not url or not title:
            return None

        source_info = raw.get("source", {})
        return Article(
            title=title,
            body=body,
            url=url,
            source=source_info.get("name", "unknown"),
            published_at=raw.get("publishedAt", ""),
            author=raw.get("author", ""),
            metadata={"category": category, "provider": "newsapi"},
        )
