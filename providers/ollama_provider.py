import logging
import os

import httpx


logger = logging.getLogger(__name__)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
AI_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_MODEL = "qwen2.5"
DEFAULT_CLOUD_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def _gateway_token():
    return os.getenv("AI_GATEWAY_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN")


def _ask_openai_compatible(url, token, model, messages):
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": model,
            "messages": messages or [],
            "max_tokens": 1024,
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
        if os.getenv("GEMINI_API_KEY"):
            backend = "gemini"
            return _ask_gemini(messages)
        if _gateway_token():
            backend = "vercel-ai-gateway"
            return _ask_gateway(messages)
        return _ask_ollama(model, messages)
    except httpx.ConnectError:
        logger.exception("Could not connect to the %s model backend.", backend)
        return "Sorry, I'm having trouble connecting to the model right now."
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "The %s model backend returned HTTP %s.",
            backend,
            exc.response.status_code,
        )
        return f"The model service returned an error ({exc.response.status_code}). Please try again."
    except Exception:
        logger.exception("The %s model backend returned an unexpected error.", backend)
        return "An error occurred while generating a response. Please try again."
