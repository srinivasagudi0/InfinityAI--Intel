from fastapi import FastAPI # The framwork that handles your API
#import uvicorn # the server that runs your API
from pydantic import BaseModel # defines the structure of the incoming data
from typing import List, Optional
from providers.ollama_provider import ask_model

memory_store: dict = {}

app = FastAPI(title="InfinityAI API", description="An API to interact with InfinityAI models", version="0.1.0")

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

def detect_intent(text):
    
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

def build_message_messages(history, user_message, intent):
    systemm_prompt = (
        "You are InfinityAI, a helpful, most intelligent, and precise assistant. Always respond in a concise manner. You are also very very powerful."
        f"The users request seems to be tied to the following intent: {intent}. "
        "Respond helpfully and concisely, Good luck!"
           )
    messages = [{"role": "system", "content": systemm_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages

# FAKE handler deleted!

# this is my one endpoint
@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    user_message = request.messages[-1].content
    session_id = request.session_id
    user_id = request.user_id

    history = get_history(session_id)
    save_to_memory(session_id, "user", user_message)

    intent = detect_intent(user_message)

    model_messages = build_message_messages(history, user_message, intent)
    reply = ask_model(messages=model_messages)

    save_to_memory(session_id, "assistant", reply)

    return {
        "model": request.model,
        "user_id": user_id,
        "session_id": session_id,
        "route": intent,
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

