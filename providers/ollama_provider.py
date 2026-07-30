import logging
import os

import httpx


logger = logging.getLogger(__name__)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
AI_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_MODEL = "qwen2.5"
DEFAULT_CLOUD_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def _gateway_token():
    return os.getenv("AI_GATEWAY_API_KEY")


def _explicit_ollama_url():
    return "OLLAMA_URL" in os.environ


def _running_on_vercel():
    return bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))


def _last_user_message(messages):
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message.get("content", "").strip()
    return ""


def _fallback_response(messages):
    user_message = _last_user_message(messages)
    if not user_message:
        return (
            "InfinityAI is online, but the cloud model is not configured for this "
            "deployment yet."
        )

    lower = user_message.lower()
    if any(marker in lower for marker in ["remember", "my name is", "don't forget", "note that"]):
        return "Got it. I saved that in this workspace and will use it when it helps."

    if any(term in lower for term in ["latest", "current", "today", "news", "look up", "search"]):
        return (
            "I cannot fetch live web results from this fallback mode yet. "
            "Add a Gemini or AI Gateway key to enable full current-information answers."
        )

    if any(term in lower for term in ["code", "program", "script", "function", "build", "implement"]):
        return (
            "I can help with that. InfinityAI is using the built-in fallback while "
            "the cloud model is unavailable, so here is a practical starting point:\n\n"
            "1. Define the exact input and expected output.\n"
            "2. Build the smallest working version first.\n"
            "3. Test the success path and the main failure cases.\n\n"
            f"Request: {user_message[:500]}"
        )

    return (
        "InfinityAI is online. The model backend is unavailable right now, "
        "so I am using the built-in fallback instead of showing a service error.\n\n"
        f"You said: {user_message[:500]}"
    )


def _ask_openai_compatible(
    url,
    token,
    model,
    messages,
    token_limit_field="max_tokens",
):
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": model,
            "messages": messages or [],
            token_limit_field: 1024,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _ask_gateway(messages):
    token = _gateway_token()
    model = os.getenv("INFINITYAI_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)
    return _ask_openai_compatible(
        AI_GATEWAY_URL,
        token,
        model,
        messages,
    )


def _ask_openai(messages):
    return _ask_openai_compatible(
        OPENAI_URL,
        os.environ["OPENAI_API_KEY"],
        os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        messages,
        token_limit_field="max_completion_tokens",
    )


def _ask_gemini(messages):
    return _ask_openai_compatible(
        GEMINI_URL,
        os.environ["GEMINI_API_KEY"],
        os.getenv("INFINITYAI_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        messages,
    )


def _ask_ollama(model, messages):
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages or [],
            "stream": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def ask_model(model=DEFAULT_MODEL, messages=None):
    backend = "ollama"
    try:
        if os.getenv("OPENAI_API_KEY"):
            backend = "openai"
            return _ask_openai(messages)
        if os.getenv("GEMINI_API_KEY"):
            backend = "gemini"
            return _ask_gemini(messages)
        if _gateway_token():
            backend = "vercel-ai-gateway"
            return _ask_gateway(messages)
        if _running_on_vercel() and not _explicit_ollama_url():
            backend = "fallback"
            return _fallback_response(messages)
        return _ask_ollama(model, messages)
    except httpx.ConnectError:
        logger.exception("Could not connect to the %s model backend.", backend)
        return _fallback_response(messages)
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "The %s model backend returned HTTP %s.",
            backend,
            exc.response.status_code,
        )
        return _fallback_response(messages)
    except Exception:
        logger.exception("The %s model backend returned an unexpected error.", backend)
        return "An error occurred while generating a response. Please try again."
