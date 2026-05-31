import requests

BASE_URL = "http://192.168.1.7:8000"   # replace with your IP
API_KEY  = "my-secret-key-123"           # must match your .env

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Test 1 — list models
response = requests.get(f"{BASE_URL}/v1/models", headers=headers)
print("Models:", response.json())

# Test 2 — chat
payload = {
    "model": "infinity-1",
    "session_id": "test-session",
    "user_id": "test-user",
    "messages": [
        {"role": "user", "content": "Explain what an API is"}
    ]
}
response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers)
data = response.json()
print("Mode:", data["mode"])
print("Reply:", data["choices"][0]["message"]["content"])
