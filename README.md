# InfinityAI--Intel

An OpenAI like compatible API with routing, memory and a brain layer.

## Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Authentication

Every request needs this header:

`X-API-Key: my-secret-key-123`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check — no key needed |
| GET | /v1/models | List available models |
| POST | /v1/chat/completions | Send a message |

## Example request

```python
import requests

requests.post(
    "http://YOUR-IP:8000/v1/chat/completions",
    headers={"X-API-Key": "my-secret-key-123"}, # this is the key from the .env file
    json={
        "model": "infinity-1",
        "session_id": "abc",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)
```

## Modes

The API auto-detects which mode to use based on the message:
- `chat` — general conversation
- `code` — fix bugs, write functions
- `tutor` — explain concepts
- `planner` — build plans and roadmaps
- `search` — factual / current events questions
