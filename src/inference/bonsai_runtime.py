"""Client for a locally hosted Bonsai model served by llama.cpp."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit

import requests  # type: ignore[import-untyped]

from src.inference.prompts import SYSTEM_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)
_NO_PROXY = {"http": "", "https": ""}


class BonsaiRuntime:
    """Generate text through llama.cpp's local OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        max_new_tokens: int = 192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        presence_penalty: float = 1.5,
        timeout_seconds: int = 300,
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = self._validate_local_url(base_url)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer.")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.timeout_seconds = timeout_seconds
        self.enable_thinking = enable_thinking
        self._session = requests.Session()
        self._session.trust_env = False
        self._chat_completions_url = f"{self.base_url}/chat/completions"
        parts = urlsplit(self.base_url)
        self._health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))

    @staticmethod
    def _validate_local_url(base_url: str) -> str:
        """Reject non-loopback endpoints so article text stays on this machine."""
        try:
            parts = urlsplit(base_url)
            hostname = parts.hostname
            port = parts.port
        except ValueError as exc:
            raise ValueError("BONSAI_RUNTIME_URL must be a valid local HTTP URL.") from exc

        if (
            parts.scheme != "http"
            or not hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("BONSAI_RUNTIME_URL must be a local HTTP URL without credentials or query parameters.")

        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = hostname.lower() == "localhost"

        if not is_loopback:
            raise ValueError("Bonsai inference must use a loopback URL; remote model endpoints are not allowed.")

        normalized_path = parts.path.rstrip("/")
        if normalized_path != "/v1":
            raise ValueError("BONSAI_RUNTIME_URL must point to the llama.cpp /v1 API path.")

        netloc = f"[{hostname}]" if ":" in hostname else hostname
        if port is not None:
            netloc = f"{netloc}:{port}"
        return urlunsplit((parts.scheme, netloc, normalized_path, "", ""))

    def check_health(self) -> None:
        """Verify the local llama.cpp server has loaded its model."""
        try:
            response = self._session.get(
                self._health_url,
                timeout=min(self.timeout_seconds, 10),
                proxies=_NO_PROXY,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach the local Bonsai runtime. Start llama-server and check BONSAI_RUNTIME_URL."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"The local Bonsai runtime is not ready (HTTP {response.status_code}). "
                "Wait for llama-server to finish loading the model."
            )

        logger.info("Local Bonsai runtime is ready at %s", self.base_url)

    def generate(self, prompt: str) -> str:
        """Send a prompt to the local chat-completions API."""
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        try:
            response = self._session.post(
                self._chat_completions_url,
                json=payload,
                timeout=self.timeout_seconds,
                proxies=_NO_PROXY,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach the local Bonsai runtime. Start llama-server and check BONSAI_RUNTIME_URL."
            ) from exc

        if not response.ok:
            raise RuntimeError(f"The local Bonsai runtime returned HTTP {response.status_code}.")

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The local Bonsai runtime returned an invalid chat-completions response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The local Bonsai runtime returned an empty response.")

        return content.strip()
