from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.inference.analyzer import NewsAnalyzer
from src.inference.bonsai_runtime import BonsaiRuntime


class BonsaiRuntimeTests(unittest.TestCase):
    def test_posts_chat_request_to_loopback_runtime(self) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"choices": [{"message": {"content": "  A grounded summary.  "}}]}

        runtime = BonsaiRuntime()
        with patch.object(runtime._session, "post", return_value=response) as post:
            result = runtime.generate("Summarize this article.")

        self.assertEqual(result, "A grounded summary.")
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(kwargs["json"]["messages"][1]["content"], "Summarize this article.")
        self.assertEqual(kwargs["json"]["messages"][0]["role"], "system")
        self.assertFalse(kwargs["json"]["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["proxies"], {"http": "", "https": ""})

    def test_analyzer_uses_bonsai_runtime_for_article(self) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"choices": [{"message": {"content": "A concise summary."}}]}
        runtime = BonsaiRuntime()
        analyzer = NewsAnalyzer(runtime, tasks=["summarize"], minimum_body_characters=1)
        article = {"title": "Local model test", "body": "A fixture article.", "url": "https://example.com"}

        with patch.object(runtime._session, "post", return_value=response) as post:
            report = analyzer.analyze_article(article)

        self.assertEqual(report["analysis"]["summarize"], "A concise summary.")
        prompt = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("Local model test", prompt)
        self.assertIn("A fixture article.", prompt)

    def test_rejects_non_loopback_runtime_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            BonsaiRuntime(base_url="http://api.example.com/v1")

    def test_rejects_urls_without_llama_cpp_api_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "/v1"):
            BonsaiRuntime(base_url="http://127.0.0.1:8080")

    def test_reports_server_not_ready(self) -> None:
        response = Mock(status_code=503)
        runtime = BonsaiRuntime()

        with (
            patch.object(runtime._session, "get", return_value=response),
            self.assertRaisesRegex(RuntimeError, "not ready"),
        ):
            runtime.check_health()

    def test_reports_invalid_completion_shape(self) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"choices": []}
        runtime = BonsaiRuntime()

        with (
            patch.object(runtime._session, "post", return_value=response),
            self.assertRaisesRegex(RuntimeError, "invalid chat-completions response"),
        ):
            runtime.generate("A prompt")


if __name__ == "__main__":
    unittest.main()
