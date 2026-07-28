import os
import unittest
from unittest.mock import Mock, patch

import httpx

from providers import ollama_provider


class ProviderRoutingTests(unittest.TestCase):
    def test_vercel_oidc_token_is_not_used_as_gateway_key(self):
        with patch.dict(os.environ, {"VERCEL_OIDC_TOKEN": "oidc-token"}, clear=True):
            self.assertIsNone(ollama_provider._gateway_token())

    def test_vercel_without_cloud_key_uses_fallback(self):
        messages = [{"role": "user", "content": "Hello"}]

        with patch.dict(
            os.environ,
            {"VERCEL": "1", "VERCEL_OIDC_TOKEN": "oidc-token"},
            clear=True,
        ), patch("providers.ollama_provider.httpx.post") as post:
            reply = ollama_provider.ask_model(messages=messages)

        post.assert_not_called()
        self.assertIn("InfinityAI is online", reply)
        self.assertNotIn("403", reply)

    def test_explicit_gateway_key_still_uses_gateway(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "gateway response"}}]
        }

        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "real-key"}, clear=True), patch(
            "providers.ollama_provider.httpx.post", return_value=response
        ) as post:
            reply = ollama_provider.ask_model(messages=[{"role": "user", "content": "Hi"}])

        self.assertEqual(reply, "gateway response")
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer real-key",
        )

    def test_gateway_403_returns_fallback_not_raw_service_error(self):
        request = httpx.Request("POST", ollama_provider.AI_GATEWAY_URL)
        response = httpx.Response(403, request=request)

        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "bad-key"}, clear=True), patch(
            "providers.ollama_provider.httpx.post"
        ) as post:
            post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Forbidden",
                request=request,
                response=response,
            )
            with self.assertLogs("providers.ollama_provider", level="WARNING"):
                reply = ollama_provider.ask_model(messages=[{"role": "user", "content": "Hi"}])

        self.assertIn("InfinityAI is online", reply)
        self.assertNotIn("service returned an error (403)", reply)


if __name__ == "__main__":
    unittest.main()
