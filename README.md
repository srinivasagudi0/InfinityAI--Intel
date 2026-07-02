# InfinityAI--Intel

A local-first InfinityAI assistant built on FastAPI, SQLite, and a local Ollama model.

People can create an account, chat with Infinity, upload files, keep persistent memory, generate code, use browser voice, create AI agents, build workflows, and use their own default Jarvis assistant.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the web app:

```text
http://127.0.0.1:8000
```

The API stores local data in `data/infinity.db` and uploads in `uploads/<user_id>/`.

## Environment

Create a `.env` file:

```bash
INFINITYAI_API_KEY=my-secret-key-123
INFINITYAI_JWT_SECRET=change-this-for-login-tokens
```

Ollama should be running locally at `http://127.0.0.1:11434` with the configured model in `providers/ollama_provider.py`.

## Auth

User endpoints use a bearer token from login/register:

```http
Authorization: Bearer <token>
```

The legacy model/chat API can still use:

```http
X-API-Key: my-secret-key-123
```

## Main endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Static web app |
| GET | `/health` | Health check |
| POST | `/auth/register` | Create account and default Jarvis agent |
| POST | `/auth/login` | Get bearer token |
| GET | `/me` | Current user |
| GET | `/v1/models` | List models |
| POST | `/v1/chat/completions` | Chat with Infinity, memory, files, agents, or workflow |
| GET/POST/DELETE | `/v1/memories` | Persistent memory |
| GET/POST/DELETE | `/v1/files` | File upload and listing |
| CRUD | `/v1/agents` | Custom AI agents |
| CRUD | `/v1/workflows` | Prompt workflows |
| POST | `/v1/workflows/{id}/run` | Run workflow steps |

## Chat example

```python
import requests

token = requests.post(
    "http://127.0.0.1:8000/auth/login",
    json={"email": "you@example.com", "password": "secret123"},
).json()["access_token"]

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "model": "infinity-1",
        "session_id": "main",
        "messages": [{"role": "user", "content": "Remember my favorite stack is FastAPI"}],
        "use_memory": True,
    },
)

print(response.json()["choices"][0]["message"]["content"])
```

## Test

```bash
python -m unittest test_mvp.py
```
