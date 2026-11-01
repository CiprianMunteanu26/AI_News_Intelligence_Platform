from __future__ import annotations

import unittest
from unittest.mock import patch

from src.ingestion.article_body import ArticleBodyFetcher
from src.ingestion.base_source import Article


class ArticleBodyFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = ArticleBodyFetcher(
            allowed_domains=["example.com"],
            minimum_characters=50,
            timeout_seconds=3,
        )
        self.article = Article(
            title="A headline",
            body="A short feed excerpt.",
            url="https://news.example.com/story",
            source="Example News",
        )

    def test_extracts_article_paragraphs_and_preserves_excerpt_provenance(self) -> None:
        paragraph = "This is the full article text with enough detail to exceed the configured minimum length."
        html = f"<html><article><p>{paragraph}</p><p>Short</p></article></html>".encode()

        class FakeResponse:
            status_code = 200

            @property
            def headers(self) -> dict[str, str]:
                return {"Content-Type": "text/html; charset=utf-8"}

            def iter_content(self, chunk_size: int):
                return iter([html])

            def close(self) -> None:
                pass

        with patch("src.ingestion.article_body.requests.get", return_value=FakeResponse()) as get:
            enriched = self.fetcher.enrich(self.article)

        self.assertEqual(enriched.body, paragraph)
        self.assertEqual(enriched.metadata["body_source"], "publisher_page")
        self.assertEqual(enriched.metadata["original_feed_excerpt"], self.article.body)
        self.assertEqual(enriched.url, self.article.url)
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["allow_redirects"], False)
        self.assertEqual(get.call_args.kwargs["timeout"], 3)

    def test_does_not_fetch_untrusted_hosts_or_non_https_urls(self) -> None:
        untrusted = Article(
            title=self.article.title,
            body=self.article.body,
            url="https://example.com.attacker.test/story",
            source=self.article.source,
        )
        insecure = Article(
            title=self.article.title,
            body=self.article.body,
            url="http://news.example.com/story",
            source=self.article.source,
        )
        unusual_port = Article(
            title=self.article.title,
            body=self.article.body,
            url="https://news.example.com:8443/story",
            source=self.article.source,
        )

        with patch("src.ingestion.article_body.requests.get") as get:
            self.assertIs(self.fetcher.enrich(untrusted), untrusted)
            self.assertIs(self.fetcher.enrich(insecure), insecure)
            self.assertIs(self.fetcher.enrich(unusual_port), unusual_port)

        get.assert_not_called()

    def test_keeps_feed_excerpt_when_publisher_page_is_unavailable(self) -> None:
        class FakeResponse:
            status_code = 403

            @property
            def headers(self) -> dict[str, str]:
                return {}

            def close(self) -> None:
                pass

        with patch("src.ingestion.article_body.requests.get", return_value=FakeResponse()):
            enriched = self.fetcher.enrich(self.article)

        self.assertIs(enriched, self.article)
        self.assertEqual(enriched.body, "A short feed excerpt.")

    def test_follows_redirect_within_allowlisted_https_domains(self) -> None:
        paragraph = "This is the full article text with enough detail to exceed the configured minimum length."
        html = f"<html><article><p>{paragraph}</p></article></html>".encode()

        class RedirectResponse:
            status_code = 301

            @property
            def headers(self) -> dict[str, str]:
                return {"Location": "https://www.example.com/story"}

            def close(self) -> None:
                pass

        class HtmlResponse:
            status_code = 200

            @property
            def headers(self) -> dict[str, str]:
                return {"Content-Type": "text/html"}

            def iter_content(self, chunk_size: int):
                return iter([html])

            def close(self) -> None:
                pass

        with patch(
            "src.ingestion.article_body.requests.get",
            side_effect=[RedirectResponse(), HtmlResponse()],
        ) as get:
            enriched = self.fetcher.enrich(self.article)

        self.assertEqual(enriched.body, paragraph)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[1].args[0], "https://www.example.com/story")
        self.assertFalse(get.call_args_list[0].kwargs["allow_redirects"])
        self.assertFalse(get.call_args_list[1].kwargs["allow_redirects"])

    def test_rejects_redirect_outside_allowlisted_domains(self) -> None:
        class RedirectResponse:
            status_code = 302

            @property
            def headers(self) -> dict[str, str]:
                return {"Location": "https://attacker.test/story"}

            def close(self) -> None:
                pass

        with patch("src.ingestion.article_body.requests.get", return_value=RedirectResponse()) as get:
            enriched = self.fetcher.enrich(self.article)

        self.assertIs(enriched, self.article)
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
