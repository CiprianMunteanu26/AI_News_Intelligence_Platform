from __future__ import annotations

from typing import Protocol


class TextGenerator(Protocol):
    """Interface shared by local inference runtimes."""

    def generate(self, prompt: str) -> str:
        """Generate a response for a prompt."""
