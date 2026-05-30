from fastapi import FastAPI # The framwork that handles your API
#import uvicorn # the server that runs your API
from pydantic import BaseModel # defines the structure of the incoming data
from typing import List

app = FastAPI(title="InfinityAI API", description="An API to interact with InfinityAI models", version="0.1.0")

# the below describes what a message looks like.
class Message(BaseModel):
    role: str
    content: str

# this describes the full 'request' body

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

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

# FAKE handler
def handle_search(message):
    return f"[SEARCH] I would search for: {message}. Real search functionality is coming soon!"
def handle_memory(message):
    return f"[MEMORY] I would remember: {message}. Real memory functionality is coming soon!"
def handle_code(message):
    return  f"[CODE] I would write code for: {message}. Real code generation functionality is coming soon!"
def handle_chat(message):
    return f"[CHAT] You said: {message}. Real chat functionality is coming soon!"



# this is my one endpoint
@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    last_message = request.messages[-1].content

    intent = detect_intent(last_message)

    if intent == "search":
        reply = handle_search(last_message)
    elif intent == "memory":
        reply = handle_memory(last_message)
    elif intent == "code":
        reply = handle_code(last_message)
    else:
        reply = handle_chat(last_message)

    return {
        "model": request.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": reply
                }
            }
        ]
    }