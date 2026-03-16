"""
Abstract base class for all news data sources.

Every ingestion source (NewsAPI, RSS, scraper, etc.) must implement
the ``fetch_articles`` interface so the collector can treat them uniformly.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Article:
    """Standardized article schema used across the entire platform."""

    title: str
    body: str
    url: str
    source: str
    published_at: str = ""
    author: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def uid(self) -> str:
        """Deterministic unique ID derived from the URL."""
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Article:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NewsSource(ABC):
    """
    Base class every ingestion source must extend.

    Subclasses implement ``fetch_articles`` which returns a list of
    :class:`Article` objects in the standard schema.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this source (e.g. ``'NewsAPI'``)."""

    @abstractmethod
    def fetch_articles(self) -> list[Article]:
        """
        Fetch articles from the external source.

        Returns
        -------
        list[Article]
            Newly fetched articles in the standardized schema.
        """
