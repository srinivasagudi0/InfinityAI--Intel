import httpx # sends HTTP requests- same idea as broser fetching a page

OLLAMA_URL = "http://localhost:11434/api/chat" # Ollama runs locally on this port
DEFAULT_MODEL = "qwen2.5" # the name of your Ollama model

def ask_model(model=DEFAULT_MODEL, messages:list=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except httpx.ConnectionError as exc:
        return "Sorry, I'm having trouble connecting to the model right now."
    except Exception as exc:
        return f"An error occurred: {str(exc)}"
    