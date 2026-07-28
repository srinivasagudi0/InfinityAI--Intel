import os

import requests


BASE_URL = os.getenv("INFINITYAI_BASE_URL", "http://127.0.0.1:8000")
WORKSPACE_ID = os.getenv("INFINITYAI_WORKSPACE_ID", "manual-api-check")


def main():
    headers = {
        "X-Workspace-ID": WORKSPACE_ID,
        "Content-Type": "application/json",
    }

    models = requests.get(f"{BASE_URL}/v1/models", headers=headers, timeout=20)
    models.raise_for_status()
    print("Models:", models.json())

    payload = {
        "model": "infinity-1",
        "session_id": "test-session",
        "messages": [{"role": "user", "content": "Explain what an API is"}],
    }
    chat = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=60,
    )
    chat.raise_for_status()
    data = chat.json()
    print("Mode:", data["mode"])
    print("Reply:", data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
