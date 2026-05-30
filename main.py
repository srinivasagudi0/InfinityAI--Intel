from fastapi import FastAPI # The framwork that handles your API
#import uvicorn # the server that runs your API
from pydantic import BaseModel # defines the structure of the incoming data
from typing import List, Optional
from providers.ollama_provider import ask_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from brain.prompt_builder import build_prompt as bp
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
import os

memory_store: dict = {}

load_dotenv() # Load environment variables from .env file

app = FastAPI(title="InfinityAI API", description="An API to interact with InfinityAI models", version="0.6.0")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# API key protection
API_KEY = os.getenv("INFINITYAI_API_KEY", "default-key")
api_key = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return key

# the below describes what a message looks like.
class Message(BaseModel):
    role: str
    content: str

# this describes the full 'request' body

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "default"

# MEMORY functions

def get_history(session_id):
    history = memory_store.get(session_id, [])
    return history[-5:]

def save_to_memory(session_id, role, content):
    if session_id not in memory_store:
        memory_store[session_id] = []
    memory_store[session_id].append({"role": role, "content": content})

def summarize_history(history):
    if not history:
        return "No conversation history."
    lines = []
    for msg in history:
        lines.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(lines)

# ROUTER function

def detect_mode(text):
    
    text = text.lower()

    search_terms = ["search", "latest", "news", "current", "today", "find", 'look up']
    if any(term in text for term in search_terms):
        return "search"
    
    memory_keywords = ["remember", "my name is", "don't forget", "recall", "note that", "keep in mind"]
    if any(keyword in text for keyword in memory_keywords):
        return "memory"
    
    code_keywords = ["code", "program", "script", "function", "algorithm", "plan", "implement", "build", "develop"]
    if any(keyword in text for keyword in code_keywords):
        return "code"
    
    return "chat"

# Build the message list for mode

@app.get("/v1/models", dependencies = [Depends(verify_key)])
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "infinity-1",
                "object": "model",
                "owned_by": "infinityai",
                "description": "InfinityAI prototyp - powered by QWEN 2.5(for now)"
            }
        ]
    }
# FAKE handler deleted!
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.6.0"}


# this is my one endpoint
@app.post("/v1/chat/completions")
def chat(request: ChatRequest, dependencies=[Depends(verify_key)]):
    user_message = request.messages[-1].content
    session_id = request.session_id
    user_id = request.user_id


    history = get_history(session_id)
    save_to_memory(session_id, "user", user_message)

    mode = detect_mode(user_message)

    model_messages = bp(user_message, mode, history)
    reply = ask_model(messages=model_messages)

    save_to_memory(session_id, "assistant", reply)

    return {
        "model": request.model,
        "user_id": user_id,
        "session_id": session_id,
        "route": mode,
        "memory_context": summarize_history(history),
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": reply
                }
            }
        ]
    }


# __Notes__
# to host it publicly (same internet)
# use `uvicorn main:app --host 0.0.0.0 --port 8000`
# then find your public IP using `ifconfig | grep inet | grep -v 127.0.0.1`
# then simply do this http://YOUR_PUBLIC_IP:8000/v1/chat/completions to access the API from anywhere on the same internet (with the right API key of course)