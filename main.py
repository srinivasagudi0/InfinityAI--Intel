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

# this is my one endpoint
@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    last_message = request.messages[-1].content

    return {
        "model": request.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"Hello from InfinityAI! You said: {last_message}"
                }
            }
        ]
    }