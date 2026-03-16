"""
News preprocessor — cleaning and normalization logic.

Provides tools to clean HTML tags, normalize whitespace, and standardize
article content before dataset creation.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsPreprocessor:
    """
    Handles cleaning and normalization of news articles.
    """

    def __init__(self) -> None:
        pass

    def preprocess_article(self, article: dict) -> dict:
        """
        Clean and normalize a single article dictionary.

        Parameters
        ----------
        article : dict
            Article data containing 'title', 'body', etc.

        Returns
        -------
        dict
            Modified article with cleaned content.
        """
        article["title"] = self.clean_text(article.get("title", ""))
        article["body"] = self.clean_text(article.get("body", ""))

        # Additional metadata cleaning if needed
        if "author" in article and article["author"]:
            article["author"] = self.clean_text(article["author"])

        return article

    def clean_text(self, text: str) -> str:
        """
        Apply full cleaning pipeline to text.
        """
        if not text:
            return ""

        # 1. Remove HTML tags
        text = self._remove_html(text)

        # 2. Normalize unicode (NFKD decomposes combined characters)
        text = unicodedata.normalize("NFKD", text)

        # 3. Basic normalization
        text = self.normalize_text(text)

        return text

    def normalize_text(self, text: str) -> str:
        """
        Normalize whitespace and basic characters.
        """
        # Replace various whitespace with single space
        text = re.sub(r"\s+", " ", text)

        # Trim
        text = text.strip()

        return text

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags using BeautifulSoup."""
        try:
            # Use 'lxml' if available, otherwise fallback to 'html.parser'
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ")
        except Exception as e:
            logger.warning("HTML cleaning failed, falling back to regex: %s", e)
            # Simple regex fallback if BS fails
            return re.sub(r"<[^>]+>", "", text)
