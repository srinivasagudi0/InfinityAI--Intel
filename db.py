import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def configured_path(env_name, default_path):
    configured = os.getenv(env_name)
    path = Path(configured) if configured else default_path
    return path if path.is_absolute() else BASE_DIR / path


DB_PATH = configured_path("INFINITYAI_DB_PATH", Path("data/infinity.db"))
UPLOAD_DIR = configured_path("INFINITYAI_UPLOAD_DIR", Path("uploads"))


def now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def closing_connect():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_dict(row):
    return dict(row) if row else None


def rows_dict(rows):
    return [dict(row) for row in rows]


def init_db():
    with closing_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                external_user_id TEXT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                content_type TEXT,
                path TEXT NOT NULL,
                text_content TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT,
                instructions TEXT NOT NULL,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT,
                steps_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def create_user(email, password_hash, display_name=None):
    with closing_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (email, password_hash, display_name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (email.lower(), password_hash, display_name, now()),
        )
        user_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO agents
                (user_id, name, description, instructions, is_default, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (
                user_id,
                "Jarvis",
                "Your personal Infinity assistant.",
                "Act as the user's personal Jarvis: practical, proactive, concise, and aware of their saved memory, files, agents, and workflows.",
                now(),
            ),
        )
        return user_id


def get_user(user_id):
    with closing_connect() as conn:
        return row_dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def get_user_by_email(email):
    with closing_connect() as conn:
        return row_dict(
            conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        )


def public_user(user):
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }


def save_chat_message(user_id, external_user_id, session_id, role, content):
    with closing_connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages
                (user_id, external_user_id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, external_user_id, session_id, role, content, now()),
        )


def get_history(user_id, external_user_id, session_id, limit=8):
    with closing_connect() as conn:
        if user_id:
            rows = conn.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE user_id IS NULL AND external_user_id = ? AND session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (external_user_id, session_id, limit),
            ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def list_memories(user_id):
    with closing_connect() as conn:
        return rows_dict(
            conn.execute(
                "SELECT id, content, created_at FROM memories WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
            ).fetchall()
        )


def add_memory(user_id, content):
    with closing_connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories (user_id, content, created_at) VALUES (?, ?, ?)",
            (user_id, content, now()),
        )
        return row_dict(
            conn.execute(
                "SELECT id, content, created_at FROM memories WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        )


def delete_owned(table, item_id, user_id):
    with closing_connect() as conn:
        cur = conn.execute(f"DELETE FROM {table} WHERE id = ? AND user_id = ?", (item_id, user_id))
        return cur.rowcount > 0


def add_file(user_id, filename, content_type, path, text_content):
    with closing_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO files
                (user_id, filename, content_type, path, text_content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, filename, content_type, str(path), text_content, now()),
        )
        return row_dict(
            conn.execute(
                "SELECT * FROM files WHERE id = ? AND user_id = ?",
                (cur.lastrowid, user_id),
            ).fetchone()
        )


def get_owned(table, item_id, user_id):
    with closing_connect() as conn:
        return row_dict(
            conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND user_id = ?", (item_id, user_id)
            ).fetchone()
        )


def list_owned(table, user_id):
    with closing_connect() as conn:
        return rows_dict(
            conn.execute(
                f"SELECT * FROM {table} WHERE user_id = ? ORDER BY id DESC", (user_id,)
            ).fetchall()
        )


def get_files(user_id, file_ids):
    if not file_ids:
        return []
    placeholders = ",".join("?" for _ in file_ids)
    with closing_connect() as conn:
        return rows_dict(
            conn.execute(
                f"""
                SELECT * FROM files
                WHERE user_id = ? AND id IN ({placeholders})
                ORDER BY id DESC
                """,
                [user_id, *file_ids],
            ).fetchall()
        )


def create_agent(user_id, name, description, instructions):
    with closing_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO agents
                (user_id, name, description, instructions, is_default, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (user_id, name, description, instructions, now()),
        )
        return row_dict(
            conn.execute(
                "SELECT * FROM agents WHERE id = ? AND user_id = ?",
                (cur.lastrowid, user_id),
            ).fetchone()
        )


def update_agent(agent_id, user_id, name, description, instructions):
    with closing_connect() as conn:
        conn.execute(
            """
            UPDATE agents
            SET name = ?, description = ?, instructions = ?
            WHERE id = ? AND user_id = ?
            """,
            (name, description, instructions, agent_id, user_id),
        )
        return row_dict(
            conn.execute(
                "SELECT * FROM agents WHERE id = ? AND user_id = ?",
                (agent_id, user_id),
            ).fetchone()
        )


def create_workflow(user_id, name, description, steps):
    with closing_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO workflows
                (user_id, name, description, steps_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, name, description, json.dumps(steps), now()),
        )
        return normalize_workflow(
            row_dict(
                conn.execute(
                    "SELECT * FROM workflows WHERE id = ? AND user_id = ?",
                    (cur.lastrowid, user_id),
                ).fetchone()
            )
        )


def update_workflow(workflow_id, user_id, name, description, steps):
    with closing_connect() as conn:
        conn.execute(
            """
            UPDATE workflows
            SET name = ?, description = ?, steps_json = ?
            WHERE id = ? AND user_id = ?
            """,
            (name, description, json.dumps(steps), workflow_id, user_id),
        )
        return normalize_workflow(
            row_dict(
                conn.execute(
                    "SELECT * FROM workflows WHERE id = ? AND user_id = ?",
                    (workflow_id, user_id),
                ).fetchone()
            )
        )


def normalize_workflow(workflow):
    if workflow:
        workflow["steps"] = json.loads(workflow.pop("steps_json"))
    return workflow


def normalize_workflows(workflows):
    return [normalize_workflow(workflow) for workflow in workflows]
