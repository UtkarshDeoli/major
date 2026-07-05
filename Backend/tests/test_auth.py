import uuid

from fastapi.testclient import TestClient

# Must patch before importing the app so the module sees the mocked collection.
import src.services.auth_service as auth_service


class _FakeCollection:
    """Minimal async MongoDB collection mock backed by an in-memory dict."""

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
    # security module holds no direct collection reference, but ensure it reuses auth_service
    from src.main import app
    return TestClient(app)


def test_login_returns_user_profile(client):
    """Login must return the full user profile alongside the token."""
    unique = uuid.uuid4().hex
    payload = {
        "email": f"login-user-{unique}@example.com",
        "password": "password123",
        "name": "Login User",
    }
    signup = client.post("/auth/signup", json=payload)
    assert signup.status_code == 201

    response = client.post(
        "/auth/login",
        data={"username": payload["email"], "password": payload["password"]},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "user" in data
    assert data["user"]["email"] == payload["email"]
    assert data["user"]["role"] == "student"


def test_signup_defaults_to_student_role(client):
    """Public signup must always create a student, never a privileged role."""
    unique = uuid.uuid4().hex
    payload = {
        "email": f"public-signup-student-{unique}@example.com",
        "password": "password123",
        "name": "New Student",
    }
    response = client.post("/auth/signup", json=payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["email"] == payload["email"]
    assert "user" in data
    assert data["user"]["role"] == "student"
    assert data["user"]["onboarding_completed"] is False

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert me.status_code == 200
    assert me.json()["role"] == "student"
