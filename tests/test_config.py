from __future__ import annotations

import unittest

from src.utils.config import _build_ingestion


class IngestionConfigTests(unittest.TestCase):
    def test_build_ingestion_keeps_enrichment_settings_and_sources(self) -> None:
        config = _build_ingestion(
            {
                "enrich_article_bodies": False,
                "article_body_min_characters": 650,
                "article_fetch_timeout_seconds": 4,
                "max_article_page_fetches": 7,
                "article_domains": ["example.com"],
                "sources": [{"type": "rss", "feeds": ["https://example.com/feed.xml"]}],
            }
        )

        self.assertFalse(config.enrich_article_bodies)
        self.assertEqual(config.article_body_min_characters, 650)
        self.assertEqual(config.article_fetch_timeout_seconds, 4)
        self.assertEqual(config.max_article_page_fetches, 7)
        self.assertEqual(config.article_domains, ["example.com"])
        self.assertEqual(config.sources[0].feeds, ["https://example.com/feed.xml"])


if __name__ == "__main__":
    unittest.main()
