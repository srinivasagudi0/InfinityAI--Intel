# InfinityAI--Intel

This is my cool local AI helper! It runs on your computer and talks to you. It saves stuff, remembers things, and can make code and play with files.

## How to start (easy!)

Open a terminal and type:
```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Then open: http://127.0.0.1:8000

Files go in `uploads/<user_id>/` and the database is `data/infinity.db`.

## Setup (like instructions)

Make a file called `.env` and put these inside:
```
INFINITYAI_API_KEY=your-secret-key
INFINITYAI_JWT_SECRET=your-jwt-secret
```
Also make sure Ollama is running here: `http://127.0.0.1:11434`

## Login (so you can use it)

When you log in you get a token. Send it like this:
```
Authorization: Bearer <token>
```
Or use the old way:
```
X-API-Key: your-key
```

## What it can do (short)

- Register and login
- Chat with memory
- Upload files
- Make agents and workflows

## Important URLs

- POST /auth/register — sign up
- POST /auth/login — log in
- POST /v1/chat/completions — chat
- /v1/memories — manage memory
- /v1/files — upload files
- /v1/agents and /v1/workflows — make things

## Example (quick)

```python
import requests

token = requests.post("http://127.0.0.1:8000/auth/login",
    json={"email": "user@example.com", "password": "pass"}).json()["access_token"]

response = requests.post("http://127.0.0.1:8000/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    json={"model": "infinity-1", "messages": [{"role": "user", "content": "Hi"}], "use_memory": True})

print(response.json()["choices"][0]["message"]["content"])
```

## Test

Run this to check:
```
python -m unittest test_mvp.py
```

Have fun! 🎉
