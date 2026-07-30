# InfinityAI--Intel

This is an open AI workspace for chat, memory, files, agents, and workflows. It opens directly into the app without accounts, passwords, or a login screen.


# AI Declaration

I used AI to help draft parts of the original CSS and the README's “How to
start” introduction. I also used AI suggestions while reviewing the later CSS
cleanup. 

## How to start (easy!)

Open a terminal and type:
```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Then open: http://127.0.0.1:8000

Files go in `uploads/<user_id>/` and the database is `data/infinity.db`.

## Setup (like instructions)

Make a file called `.env` and add the model configuration you want to use:
```
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-5.6-terra
```
`OPENAI_MODEL` is optional. Without a cloud key, local development falls back to
Ollama at `http://127.0.0.1:11434`.

For Vercel, add `OPENAI_API_KEY` in the project environment variables, then
redeploy. You can also set `OPENAI_MODEL` if you want a different OpenAI model.
The app still supports `GEMINI_API_KEY` or `AI_GATEWAY_API_KEY`, but OpenAI is
used first when `OPENAI_API_KEY` is present. If no cloud key is configured, the
hosted app uses a built-in fallback response instead of showing a model service
error.

The browser creates a random local workspace ID automatically. API clients can optionally send their own value in the `X-Workspace-ID` header; requests without one use the public workspace.

## What it can do (short)

- Opens directly without login
- Chat with memory
- Upload files
- Make agents and workflows

## Important URLs

- POST /v1/chat/completions — chat
- /v1/memories — manage memory
- /v1/files — upload files
- /v1/agents and /v1/workflows — make things

## Example (quick)

```python
import requests

response = requests.post("http://127.0.0.1:8000/v1/chat/completions",
    headers={"X-Workspace-ID": "my-private-workspace"},
    json={"model": "infinity-1", "messages": [{"role": "user", "content": "Hi"}], "use_memory": True})

print(response.json()["choices"][0]["message"]["content"])
```

## Test

Run this to check:
```
python -m unittest test_mvp.py
```

Have fun! 🎉

I have hosted it on vercel if you ant tot try it ou without installing it. You can find it here: [https://infinityai-intel.vercel.app](https://infinityai-intel.vercel.app)

This project took me about 2 weeks to make. I didn't know we had to devlog, so sorry about that.
