import os
import tempfile
import unittest
import uuid


_tmp = tempfile.TemporaryDirectory()
os.environ["INFINITYAI_API_KEY"] = "test-key"
os.environ["INFINITYAI_DB_PATH"] = os.path.join(_tmp.name, "infinity-test.db")
os.environ["INFINITYAI_UPLOAD_DIR"] = os.path.join(_tmp.name, "uploads")

from fastapi.testclient import TestClient

import main


def fake_ask_model(model="infinity-1", messages=None):
    return f"FAKE: {messages[-1]['content']}"


main.ask_model = fake_ask_model


class InfinityMvpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.db.init_db()
        cls.client = TestClient(main.app)

    def workspace(self):
        return {"X-Workspace-ID": f"test-{uuid.uuid4().hex}"}

    def test_open_workspace_has_default_jarvis_and_no_login(self):
        headers = self.workspace()
        agents = self.client.get("/v1/agents", headers=headers).json()["data"]
        self.assertTrue(any(agent["name"] == "Jarvis" and agent["is_default"] for agent in agents))
        self.assertEqual(self.client.post("/auth/login", json={}).status_code, 404)
        self.assertEqual(self.client.post("/auth/register", json={}).status_code, 404)

    def test_chat_works_without_login(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "infinity-1",
                "session_id": "open-session",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "FAKE: Hello")

    def test_memory_file_agent_and_workflow(self):
        headers = self.workspace()

        chat = self.client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Remember my favorite stack is FastAPI"}]
            },
        )
        self.assertEqual(chat.status_code, 200, chat.text)

        memories = self.client.get("/v1/memories", headers=headers).json()["data"]
        self.assertTrue(any("FastAPI" in memory["content"] for memory in memories))

        upload = self.client.post(
            "/v1/files",
            headers=headers,
            files={"file": ("notes.txt", b"Infinity project notes", "text/plain")},
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertTrue(upload.json()["text_available"])

        agent = self.client.post(
            "/v1/agents",
            headers=headers,
            json={
                "name": "Coder",
                "description": "Writes code",
                "instructions": "Prefer short Python examples.",
            },
        )
        self.assertEqual(agent.status_code, 200, agent.text)

        workflow = self.client.post(
            "/v1/workflows",
            headers=headers,
            json={
                "name": "Two step",
                "description": "Test flow",
                "steps": ["Summarize {{input}}", "Turn into a task: {{previous}}"],
            },
        )
        self.assertEqual(workflow.status_code, 200, workflow.text)

        run = self.client.post(
            f"/v1/workflows/{workflow.json()['id']}/run",
            headers=headers,
            json={"input": "Build Jarvis"},
        )
        self.assertEqual(run.status_code, 200, run.text)
        self.assertIn("Turn into a task", run.json()["final_output"])


if __name__ == "__main__":
    unittest.main()
