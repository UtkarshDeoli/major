"""Rate-limit smoke tests for the slowapi limiter.

Uses httpx.AsyncClient + ASGITransport (same pattern as
tests/test_teacher_student.py) to avoid the session event-loop problems that
TestClient can cause when used without a context manager.
"""

import asyncio
import uuid

import httpx
from httpx import ASGITransport
import pytest

from src.core.data_store import users_collection
from src.main import app


pytestmark = pytest.mark.skipif(
    users_collection is None,
    reason="MongoDB connection unavailable",
)


_loop = asyncio.get_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


async def _post_login(client: httpx.AsyncClient, email: str):
    return await client.post(
        "/auth/login",
        data={"username": email, "password": "testpassword123"},
    )


async def _post_signup(client: httpx.AsyncClient, email: str):
    return await client.post(
        "/auth/signup",
        json={"email": email, "password": "testpassword123", "name": "Rate Test"},
    )


def _unique_email(prefix: str) -> str:
    return f"{prefix}.{uuid.uuid4().hex[:8]}@example.com"


def test_login_burst_throttled():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            codes = []
            for _ in range(8):
                r = await _post_login(client, _unique_email("rate"))
                codes.append(r.status_code)
            assert 429 in codes, f"expected at least one 429, got {codes}"
    _run(_test())


def test_signup_burst_throttled():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            codes = []
            for _ in range(8):
                r = await _post_signup(client, _unique_email("rate"))
                codes.append(r.status_code)
            assert 429 in codes, f"expected at least one 429, got {codes}"
    _run(_test())


async def _token_for(client: httpx.AsyncClient, email: str, password: str):
    """Create a user and return a bearer token."""
    signup_resp = await client.post(
        "/auth/signup",
        json={"email": email, "password": password, "name": "Rate Test"},
    )
    assert signup_resp.status_code in (200, 201), signup_resp.text

    login_resp = await client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    assert login_resp.status_code == 200, login_resp.text
    return login_resp.json()["access_token"]


def test_socratic_explain_rate_limited(monkeypatch):
    """Repeated calls to /socratic/explain should be throttled at SOCRATIC_LIMIT."""
    import importlib
    sr_module = importlib.import_module("src.routers.socratic_router")
    monkeypatch.setattr(sr_module, "explain_socratically", lambda **kwargs: {"explanation": "fake"})

    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            token = await _token_for(client, _unique_email("socratic"), "testpassword123")
            headers = {"Authorization": f"Bearer {token}"}
            codes = []
            for _ in range(35):
                r = await client.post(
                    "/socratic/explain",
                    json={"question": "What is 2+2?"},
                    headers=headers,
                )
                codes.append(r.status_code)
            assert 429 in codes, f"expected at least one 429, got {codes}"
    _run(_test())


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


def _fake_plan_json() -> str:
    import json
    weeks = []
    for w in range(1, 2):
        days = []
        for day_name in ["Monday"]:
            days.append({
                "day": day_name,
                "tasks": [{"subject": "Physics", "topic": "T", "activity": "Read", "minutes": 60, "resource": "Notes"}],
            })
        weeks.append({"week": w, "focus": f"Week {w}", "days": days})
    return json.dumps({"weeks": weeks})


def test_study_plans_rate_limited(monkeypatch):
    """Repeated calls to POST /study/plans should be throttled at GENERATION_LIMIT."""
    from src.services import gemini_service as gs

    monkeypatch.setattr(gs.gemini_service.model, "generate_content", lambda prompt: _FakeGeminiResponse(_fake_plan_json()))

    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            token = await _token_for(client, _unique_email("study_rate"), "testpassword123")
            headers = {"Authorization": f"Bearer {token}"}
            codes = []
            for _ in range(12):
                r = await client.post(
                    "/study/plans",
                    json={"title": "Rate test", "subjects": [], "weak_topics": [], "hours_per_day": 4, "num_weeks": 1},
                    headers=headers,
                )
                codes.append(r.status_code)
            assert any(c in (402, 429) for c in codes), f"expected throttling, got {codes}"
    _run(_test())
