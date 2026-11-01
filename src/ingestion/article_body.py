"""Bounded extraction of article text from allowlisted publisher pages."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from src.ingestion.base_source import Article
from src.utils.logger import get_logger

logger = get_logger(__name__)
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_BODY_CHARACTERS = 10000
_MIN_PARAGRAPH_CHARACTERS = 40
_MAX_REDIRECTS = 3
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class ArticleBodyFetcher:
    """Fetch full text only from configured publisher domains."""

    def __init__(
        self,
        allowed_domains: list[str],
        minimum_characters: int = 500,
        timeout_seconds: int = 10,
    ) -> None:
        if minimum_characters < 1:
            raise ValueError("minimum_characters must be a positive integer.")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer.")
        self._allowed_domains = tuple(
            domain.strip().lower().lstrip(".").rstrip(".") for domain in allowed_domains if domain.strip()
        )
        self._minimum_characters = minimum_characters
        self._timeout_seconds = timeout_seconds

    def is_candidate(self, article: Article) -> bool:
        """Return whether an article is short and hosted on an allowed HTTPS domain."""
        if len(article.body.strip()) >= self._minimum_characters:
            return False

        return self._is_allowed_url(article.url)

    def enrich(self, article: Article) -> Article:
        """Replace a short feed excerpt with publisher article text when available."""
        if not self.is_candidate(article):
            return article

        content = self._fetch_html(article.url)
        if content is None:
            return article

        extracted = self._extract_paragraphs(content)
        if len(extracted) < self._minimum_characters or len(extracted) <= len(article.body.strip()):
            logger.info("Publisher page did not provide enough additional article text: %s", article.url)
            return article

        metadata = {
            **article.metadata,
            "body_source": "publisher_page",
            "original_feed_excerpt": article.body,
        }
        logger.info("Enriched article body from publisher page: %s", article.url)
        return Article(
            title=article.title,
            body=extracted,
            url=article.url,
            source=article.source,
            published_at=article.published_at,
            author=article.author,
            metadata=metadata,
            fetched_at=article.fetched_at,
        )

    @staticmethod
    def _extract_paragraphs(html: bytes) -> str:
        """Extract substantial article paragraphs from semantic page content."""
        soup = BeautifulSoup(html, "html.parser")
        container = soup.find("article") or soup.find("main")
        if container is None:
            container = soup

        paragraphs = [
            re.sub(r"\s+", " ", paragraph.get_text(" ", strip=True))
            for paragraph in container.find_all("p")
        ]
        body = " ".join(text for text in paragraphs if len(text) >= _MIN_PARAGRAPH_CHARACTERS)
        return body[:_MAX_BODY_CHARACTERS].strip()

    def _fetch_html(self, url: str) -> bytes | None:
        """Fetch bounded HTML while validating every HTTPS redirect against the allowlist."""
        current_url = url
        for redirect_count in range(_MAX_REDIRECTS + 1):
            try:
                response = requests.get(
                    current_url,
                    headers={"User-Agent": "AI-News-Intelligence-Platform/1.0 (article text extraction)"},
                    timeout=self._timeout_seconds,
                    stream=True,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                logger.warning("Could not fetch publisher article text from %s: %s", current_url, exc)
                return None

            if response.status_code in _REDIRECT_STATUS_CODES:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    logger.info("Publisher redirect had no Location header: %s", current_url)
                    return None
                redirect_url = urljoin(current_url, location)
                if not self._is_allowed_url(redirect_url):
                    logger.warning("Skipping publisher redirect outside the HTTPS domain allowlist: %s", redirect_url)
                    return None
                if redirect_count == _MAX_REDIRECTS:
                    logger.warning("Publisher page exceeded the %d-redirect limit: %s", _MAX_REDIRECTS, url)
                    return None
                current_url = redirect_url
                continue

            try:
                if response.status_code != 200:
                    logger.info(
                        "Skipping article text extraction for %s (HTTP %d)",
                        current_url,
                        response.status_code,
                    )
                    return None

                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    logger.info("Skipping non-HTML publisher page: %s", current_url)
                    return None

                content = bytearray()
                for chunk in response.iter_content(chunk_size=65536):
                    content.extend(chunk)
                    if len(content) >= _MAX_RESPONSE_BYTES:
                        del content[_MAX_RESPONSE_BYTES:]
                        break
                return bytes(content)
            except requests.RequestException as exc:
                logger.warning("Could not read publisher article text from %s: %s", current_url, exc)
                return None
            finally:
                response.close()

        return None

    def _is_allowed_url(self, url: str) -> bool:
        """Check that a URL uses HTTPS and its hostname matches a configured domain."""
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return False

        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password or port not in (None, 443):
            return False

        hostname = hostname.lower().rstrip(".")
        return any(hostname == domain or hostname.endswith(f".{domain}") for domain in self._allowed_domains)
