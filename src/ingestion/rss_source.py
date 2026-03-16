"""
RSS feed ingestion source.

Parses RSS/Atom feeds from major news outlets using ``feedparser``.
Supports any number of feed URLs configured via pipeline_config.yaml.
"""

from __future__ import annotations

import feedparser

from src.ingestion.base_source import Article, NewsSource
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RSSSource(NewsSource):
    """
    Collects articles from one or more RSS / Atom feeds.

    Parameters
    ----------
    feed_urls : list[str]
        List of RSS/Atom feed URLs to parse.
    """

    def __init__(self, feed_urls: list[str] | None = None) -> None:
        self._feed_urls = feed_urls or []

    # ------------------------------------------------------------------
    # NewsSource interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "RSS"

    def fetch_articles(self) -> list[Article]:
        """Parse all configured feeds and return standardized articles."""
        articles: list[Article] = []

        for url in self._feed_urls:
            try:
                feed_articles = self._parse_feed(url)
                articles.extend(feed_articles)
            except Exception:
                logger.exception("Failed to parse RSS feed: %s", url)

        logger.info("RSS: fetched %d articles from %d feeds", len(articles), len(self._feed_urls))
        return articles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_feed(self, feed_url: str) -> list[Article]:
        """Parse a single RSS/Atom feed URL."""
        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            logger.warning("Feed parse error for %s: %s", feed_url, feed.bozo_exception)
            return []

        feed_title = feed.feed.get("title", feed_url)
        articles: list[Article] = []

        for entry in feed.entries:
            article = self._entry_to_article(entry, feed_title, feed_url)
            if article:
                articles.append(article)

        return articles

    @staticmethod
    def _entry_to_article(entry: dict, feed_title: str, feed_url: str) -> Article | None:
        """Convert a feedparser entry into an Article."""
        link = entry.get("link", "")
        title = entry.get("title", "")

        if not link or not title:
            return None

        # Prefer summary/description; some feeds include full content
        body = ""
        if hasattr(entry, "content") and entry.content:
            body = entry.content[0].get("value", "")
        if not body:
            body = entry.get("summary", "") or entry.get("description", "")

        # Published date
        published = entry.get("published", "") or entry.get("updated", "")

        # Author
        author = entry.get("author", "")

        return Article(
            title=title,
            body=body,
            url=link,
            source=feed_title,
            published_at=published,
            author=author,
            metadata={"feed_url": feed_url, "provider": "rss"},
        )
