import uuid
from fastapi.testclient import TestClient
import src.services.auth_service as auth_service


class _FakeCollection:
    def __init__(self):
        self._docs = {}

    async def find_one(self, query):
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    async def insert_one(self, document):
        _id = str(uuid.uuid4())
        doc = dict(document)
        doc["_id"] = _id
        self._docs[_id] = doc

        class _Result:
            inserted_id = _id
        return _Result()


import pytest


@pytest.fixture
def client(monkeypatch):
    fake_users = _FakeCollection()
    monkeypatch.setattr(auth_service, "users_collection", fake_users)
    from src.main import app
    return TestClient(app)


def test_signup_persists_and_returns_curriculum(client):
    unique = uuid.uuid4().hex
    payload = {
        "email": f"curr-{unique}@example.com",
        "password": "password123",
        "name": "Curriculum Student",
        "curriculum": "jee-mains",
    }
    r = client.post("/auth/signup", json=payload)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["user"]["curriculum"] == "jee-mains"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert me.status_code == 200
    assert me.json()["curriculum"] == "jee-mains"


def test_signup_curriculum_is_optional(client):
    unique = uuid.uuid4().hex
    r = client.post("/auth/signup", json={
        "email": f"nocurr-{unique}@example.com",
        "password": "password123",
    })
    assert r.status_code == 201, r.text
    assert r.json()["user"]["curriculum"] is None
