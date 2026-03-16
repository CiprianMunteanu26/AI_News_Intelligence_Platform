"""
News collector — orchestrates all ingestion sources.

Instantiates the configured sources, fetches articles from each,
deduplicates by URL, and persists the results as a daily JSON file.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.ingestion.base_source import Article, NewsSource
from src.ingestion.newsapi_source import NewsAPISource
from src.ingestion.rss_source import RSSSource
from src.utils.config import PlatformConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsCollector:
    """
    Orchestrates article collection across all configured sources.

    Parameters
    ----------
    config : PlatformConfig
        Platform configuration (sources, paths, API keys).
    """

    def __init__(self, config: PlatformConfig) -> None:
        self._config = config
        self._sources: list[NewsSource] = self._build_sources()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(self, output_dir: Path | None = None) -> list[Article]:
        """
        Run all sources, deduplicate, and save results.

        Parameters
        ----------
        output_dir : optional override for the output directory.

        Returns
        -------
        list[Article]
            Deduplicated articles collected in this run.
        """
        output_dir = output_dir or self._config.paths.raw_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        all_articles: list[Article] = []

        for source in self._sources:
            logger.info("Collecting from source: %s", source.name)
            try:
                articles = source.fetch_articles()
                all_articles.extend(articles)
            except Exception:
                logger.exception("Source '%s' failed", source.name)

        # Deduplicate by URL
        unique = self._deduplicate(all_articles)
        logger.info("Collected %d articles (%d unique)", len(all_articles), len(unique))

        # Persist
        if unique:
            self._save(unique, output_dir)

        return unique

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_sources(self) -> list[NewsSource]:
        """Instantiate sources from config."""
        sources: list[NewsSource] = []

        for src_cfg in self._config.ingestion.sources:
            if not src_cfg.enabled:
                continue

            if src_cfg.type == "newsapi":
                if not self._config.newsapi_key:
                    logger.warning("NewsAPI source enabled but NEWSAPI_KEY not set — skipping")
                    continue
                sources.append(
                    NewsAPISource(
                        api_key=self._config.newsapi_key,
                        categories=src_cfg.categories,
                        country=src_cfg.country,
                        page_size=src_cfg.page_size,
                    )
                )

            elif src_cfg.type == "rss":
                sources.append(RSSSource(feed_urls=src_cfg.feeds))

            else:
                logger.warning("Unknown source type '%s' — skipping", src_cfg.type)

        logger.info("Initialized %d ingestion source(s)", len(sources))
        return sources

    @staticmethod
    def _deduplicate(articles: list[Article]) -> list[Article]:
        """Remove duplicate articles based on URL."""
        seen: set[str] = set()
        unique: list[Article] = []
        for article in articles:
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)
        return unique

    @staticmethod
    def _save(articles: list[Article], output_dir: Path) -> Path:
        """Save articles to a dated JSON file."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        filepath = output_dir / f"{date_str}_articles.json"

        data = [a.to_dict() for a in articles]

        # If file already exists for today, merge
        if filepath.exists():
            try:
                existing = json.loads(filepath.read_text(encoding="utf-8"))
                existing_urls = {a["url"] for a in existing}
                for item in data:
                    if item["url"] not in existing_urls:
                        existing.append(item)
                data = existing
            except (json.JSONDecodeError, KeyError):
                pass

        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d articles to %s", len(data), filepath)
        return filepath
