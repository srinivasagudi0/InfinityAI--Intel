import hmac
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db
from auth import create_token, decode_token, hash_password, verify_password
from brain.prompt_builder import build_prompt as bp
from providers.ollama_provider import ask_model


load_dotenv()

APP_VERSION = "1.0.0"
API_KEY = os.getenv("INFINITYAI_API_KEY", "default-key")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_FILE_CONTEXT_CHARS = 12_000
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json",
    ".csv",
    ".log",
    ".yml",
    ".yaml",
}

app = FastAPI(
    title="InfinityAI API",
    description="A local-first InfinityAI assistant with accounts, memory, files, agents, workflows, and Jarvis.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "infinity-1"
    messages: List[Message]
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "default"
    agent_id: Optional[int] = None
    workflow_id: Optional[int] = None
    file_ids: List[int] = Field(default_factory=list)
    use_memory: bool = True


class MemoryRequest(BaseModel):
    content: str


class AgentRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    instructions: str


class WorkflowRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    steps: List[str]


class WorkflowRunRequest(BaseModel):
    input: str
    agent_id: Optional[int] = None
    file_ids: List[int] = Field(default_factory=list)
    use_memory: bool = True


@app.on_event("startup")
def startup():
    db.init_db()


def normalize_email(email):
    return email.strip().lower()


def require_password(password):
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")


def token_response(user):
    return {
        "access_token": create_token(user["id"]),
        "token_type": "bearer",
        "user": db.public_user(user),
    }


def current_user_from_credentials(credentials):
    if not credentials:
        return None
    try:
        user = db.get_user(decode_token(credentials.credentials))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid bearer token.")
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


def auth_context(
    key: Optional[str] = Depends(api_key),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
):
    user = current_user_from_credentials(credentials)
    if user:
        return {"type": "bearer", "user": user}
    if key and hmac.compare_digest(key, API_KEY):
        return {"type": "api_key", "user": None}
    raise HTTPException(status_code=401, detail="Missing or invalid API key/token.")


def require_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    user = current_user_from_credentials(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Bearer login required.")
    return user


def detect_mode(text):
    text = text.lower()

    search_terms = ["search", "latest", "news", "current", "today", "find", "look up"]
    if any(term in text for term in search_terms):
        return "search"

    memory_keywords = ["remember", "my name is", "don't forget", "recall", "note that", "keep in mind"]
    if any(keyword in text for keyword in memory_keywords):
        return "memory"

    code_keywords = [
        "code",
        "program",
        "script",
        "function",
        "algorithm",
        "plan",
        "implement",
        "build",
        "develop",
    ]
    if any(keyword in text for keyword in code_keywords):
        return "code"

    return "chat"


def extract_memory(text):
    lower = text.lower()
    markers = ["remember", "my name is", "don't forget", "note that", "keep in mind"]
    return text.strip() if any(marker in lower for marker in markers) else None


def summarize_history(history):
    if not history:
        return "No conversation history."
    return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history)


def user_memories(user_id, enabled=True):
    if not user_id or not enabled:
        return []
    memories = db.list_memories(user_id)[:20]
    return [memory["content"] for memory in reversed(memories)]


def agent_instructions(user_id, agent_id):
    if not agent_id:
        return None
    if not user_id:
        raise HTTPException(status_code=401, detail="Bearer login required for agents.")
    agent = db.get_owned("agents", agent_id, user_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return agent["instructions"]


def file_context(user_id, file_ids):
    if not file_ids:
        return None
    if not user_id:
        raise HTTPException(status_code=401, detail="Bearer login required for file context.")
    files = db.get_files(user_id, file_ids)
    parts = []
    for item in files:
        if item["text_content"]:
            parts.append(f"FILE: {item['filename']}\n{item['text_content']}")
    return "\n\n".join(parts)[:MAX_FILE_CONTEXT_CHARS] if parts else None


def render_workflow_step(step, original_input, previous_output):
    return step.replace("{{input}}", original_input).replace("{{previous}}", previous_output)


def run_workflow(user_id, workflow_id, run_request):
    workflow = db.normalize_workflow(db.get_owned("workflows", workflow_id, user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")

    steps = [step.strip() for step in workflow["steps"] if step.strip()]
    if not steps:
        raise HTTPException(status_code=400, detail="Workflow needs at least one step.")

    memories = user_memories(user_id, run_request.use_memory)
    files = file_context(user_id, run_request.file_ids)
    instructions = agent_instructions(user_id, run_request.agent_id)
    previous = ""
    outputs = []

    for index, step in enumerate(steps, start=1):
        prompt = render_workflow_step(step, run_request.input, previous)
        messages = bp(
            prompt,
            detect_mode(prompt),
            [],
            memories=memories,
            file_context=files,
            agent_instructions=instructions,
        )
        previous = ask_model(messages=messages)
        outputs.append({"step": index, "prompt": prompt, "output": previous})

    return {"workflow": workflow, "steps": outputs, "final_output": previous}


def safe_filename(filename):
    clean = Path(filename or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", clean)[:120] or "upload"


def is_text_upload(filename, content_type):
    suffix = Path(filename or "").suffix.lower()
    return (content_type or "").startswith("text/") or suffix in TEXT_EXTENSIONS


def public_file(item):
    return {
        "id": item["id"],
        "filename": item["filename"],
        "content_type": item["content_type"],
        "text_available": bool(item["text_content"]),
        "created_at": item["created_at"],
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/auth/register")
def register(request: RegisterRequest):
    db.init_db()
    email = normalize_email(request.email)
    require_password(request.password)
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    try:
        user_id = db.create_user(email, hash_password(request.password), request.display_name)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account already exists for this email.")
    return token_response(db.get_user(user_id))


@app.post("/auth/login")
def login(request: LoginRequest):
    user = db.get_user_by_email(normalize_email(request.email))
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return token_response(user)


@app.get("/me")
def me(user=Depends(require_user)):
    return db.public_user(user)


@app.get("/v1/models")
def list_models(_context=Depends(auth_context)):
    return {
        "object": "list",
        "data": [
            {
                "id": "infinity-1",
                "object": "model",
                "owned_by": "infinityai",
                "description": "InfinityAI prototype powered by the configured local Ollama model.",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(request: ChatRequest, context=Depends(auth_context)):
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    user = context["user"]
    user_db_id = user["id"] if user else None
    external_user_id = str(user_db_id) if user else (request.user_id or "anonymous")
    session_id = request.session_id or "default"
    user_message = request.messages[-1].content

    history = db.get_history(user_db_id, external_user_id, session_id)
    db.save_chat_message(user_db_id, external_user_id, session_id, "user", user_message)

    mode = detect_mode(user_message)
    if user_db_id and request.use_memory:
        memory = extract_memory(user_message)
        if memory:
            db.add_memory(user_db_id, memory)

    if request.workflow_id:
        if not user_db_id:
            raise HTTPException(status_code=401, detail="Bearer login required for workflows.")
        workflow_result = run_workflow(
            user_db_id,
            request.workflow_id,
            WorkflowRunRequest(
                input=user_message,
                agent_id=request.agent_id,
                file_ids=request.file_ids,
                use_memory=request.use_memory,
            ),
        )
        reply = workflow_result["final_output"]
    else:
        workflow_result = None
        model_messages = bp(
            user_message,
            mode,
            history,
            memories=user_memories(user_db_id, request.use_memory),
            file_context=file_context(user_db_id, request.file_ids),
            agent_instructions=agent_instructions(user_db_id, request.agent_id),
        )
        reply = ask_model(messages=model_messages)

    db.save_chat_message(user_db_id, external_user_id, session_id, "assistant", reply)

    return {
        "model": request.model,
        "user_id": external_user_id,
        "session_id": session_id,
        "mode": mode,
        "route": mode,
        "memory_context": summarize_history(history),
        "workflow": workflow_result,
        "choices": [{"message": {"role": "assistant", "content": reply}}],
    }


@app.get("/v1/memories")
def list_user_memories(user=Depends(require_user)):
    return {"data": db.list_memories(user["id"])}


@app.post("/v1/memories")
def create_memory(request: MemoryRequest, user=Depends(require_user)):
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content is required.")
    return db.add_memory(user["id"], content)


@app.delete("/v1/memories/{memory_id}")
def delete_memory(memory_id: int, user=Depends(require_user)):
    if not db.delete_owned("memories", memory_id, user["id"]):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"deleted": True}


@app.get("/v1/files")
def list_files(user=Depends(require_user)):
    return {"data": [public_file(item) for item in db.list_owned("files", user["id"])]}


@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...), user=Depends(require_user)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Limit is 5 MB.")

    user_dir = db.UPLOAD_DIR / str(user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(file.filename)
    path = user_dir / f"{uuid.uuid4().hex}_{filename}"
    path.write_bytes(content)

    text_content = None
    if is_text_upload(filename, file.content_type):
        text_content = content.decode("utf-8", errors="replace")[:MAX_FILE_CONTEXT_CHARS]

    item = db.add_file(user["id"], filename, file.content_type, path, text_content)
    return public_file(item)


@app.delete("/v1/files/{file_id}")
def delete_file(file_id: int, user=Depends(require_user)):
    item = db.get_owned("files", file_id, user["id"])
    if not item:
        raise HTTPException(status_code=404, detail="File not found.")
    Path(item["path"]).unlink(missing_ok=True)
    db.delete_owned("files", file_id, user["id"])
    return {"deleted": True}


@app.get("/v1/agents")
def list_agents(user=Depends(require_user)):
    return {"data": db.list_owned("agents", user["id"])}


@app.post("/v1/agents")
def create_agent(request: AgentRequest, user=Depends(require_user)):
    if not request.name.strip() or not request.instructions.strip():
        raise HTTPException(status_code=400, detail="Agent name and instructions are required.")
    return db.create_agent(
        user["id"],
        request.name.strip(),
        (request.description or "").strip(),
        request.instructions.strip(),
    )


@app.get("/v1/agents/{agent_id}")
def get_agent(agent_id: int, user=Depends(require_user)):
    agent = db.get_owned("agents", agent_id, user["id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return agent


@app.put("/v1/agents/{agent_id}")
def update_agent(agent_id: int, request: AgentRequest, user=Depends(require_user)):
    if not request.name.strip() or not request.instructions.strip():
        raise HTTPException(status_code=400, detail="Agent name and instructions are required.")
    agent = db.update_agent(
        agent_id,
        user["id"],
        request.name.strip(),
        (request.description or "").strip(),
        request.instructions.strip(),
    )
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return agent


@app.delete("/v1/agents/{agent_id}")
def delete_agent(agent_id: int, user=Depends(require_user)):
    agent = db.get_owned("agents", agent_id, user["id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    if agent["is_default"]:
        raise HTTPException(status_code=400, detail="Default Jarvis agent cannot be deleted.")
    db.delete_owned("agents", agent_id, user["id"])
    return {"deleted": True}


@app.get("/v1/workflows")
def list_workflows(user=Depends(require_user)):
    return {"data": db.normalize_workflows(db.list_owned("workflows", user["id"]))}


@app.post("/v1/workflows")
def create_workflow(request: WorkflowRequest, user=Depends(require_user)):
    steps = [step.strip() for step in request.steps if step.strip()]
    if not request.name.strip() or not steps:
        raise HTTPException(status_code=400, detail="Workflow name and at least one step are required.")
    return db.create_workflow(user["id"], request.name.strip(), (request.description or "").strip(), steps)


@app.get("/v1/workflows/{workflow_id}")
def get_workflow(workflow_id: int, user=Depends(require_user)):
    workflow = db.normalize_workflow(db.get_owned("workflows", workflow_id, user["id"]))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return workflow


@app.put("/v1/workflows/{workflow_id}")
def update_workflow(workflow_id: int, request: WorkflowRequest, user=Depends(require_user)):
    steps = [step.strip() for step in request.steps if step.strip()]
    if not request.name.strip() or not steps:
        raise HTTPException(status_code=400, detail="Workflow name and at least one step are required.")
    workflow = db.update_workflow(
        workflow_id,
        user["id"],
        request.name.strip(),
        (request.description or "").strip(),
        steps,
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return workflow


@app.delete("/v1/workflows/{workflow_id}")
def delete_workflow(workflow_id: int, user=Depends(require_user)):
    if not db.delete_owned("workflows", workflow_id, user["id"]):
        raise HTTPException(status_code=404, detail="Workflow not found.")
    return {"deleted": True}


@app.post("/v1/workflows/{workflow_id}/run")
def run_user_workflow(workflow_id: int, request: WorkflowRunRequest, user=Depends(require_user)):
    return run_workflow(user["id"], workflow_id, request)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
