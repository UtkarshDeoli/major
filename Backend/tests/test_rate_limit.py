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