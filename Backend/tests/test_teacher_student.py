import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


async def _ensure_user(client: httpx.AsyncClient, email: str, password: str, role: str = "student"):
    from src.services.auth_service import get_password_hash, get_user_by_email
    existing = await get_user_by_email(email)
    if existing:
        if "password_hash" not in existing:
            await users_collection.update_one(
                {"email": email},
                {"$set": {"password_hash": get_password_hash(password)}},
            )
        return

    if role == "student":
        signup_resp = await client.post(
            "/auth/signup",
            json={"email": email, "password": password},
        )
        assert signup_resp.status_code in (200, 201), signup_resp.text
        return

    await users_collection.insert_one({
        "email": email,
        "password_hash": get_password_hash(password),
        "role": role,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })


async def _token_for(client: httpx.AsyncClient, email: str, password: str, role: str = "student"):
    await _ensure_user(client, email, password, role=role)
    login_resp = await client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    assert login_resp.status_code == 200, login_resp.text
    return login_resp.json()["access_token"]


def _auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def test_teacher_can_list_students_in_class():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            teacher_email = "teacher.classonly@example.com"
            student_email = "student.classonly@example.com"
            password = "testpassword123"

            teacher_token = await _token_for(client, teacher_email, password, role="teacher")
            await _ensure_user(client, student_email, password, role="student")

            create_resp = await client.post(
                "/classes/",
                json={"name": "Test Class"},
                headers=_auth_headers(teacher_token),
            )
            assert create_resp.status_code == 201, create_resp.text
            enroll_code = create_resp.json()["enroll_code"]
            class_id = create_resp.json()["id"]

            student_token = await _token_for(client, student_email, password, role="student")
            join_resp = await client.post(
                "/classes/join",
                json={"enroll_code": enroll_code},
                headers=_auth_headers(student_token),
            )
            assert join_resp.status_code == 200, join_resp.text

            roster_resp = await client.get(
                "/teachers/students",
                headers=_auth_headers(teacher_token),
            )
            assert roster_resp.status_code == 200, roster_resp.text
            emails = {s["email"] for s in roster_resp.json()["students"]}
            assert student_email in emails

    _run(_test())


def test_teacher_analytics_includes_class_student():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            teacher_email = "teacher.analytics2@example.com"
            student_email = "student.analytics2@example.com"
            password = "testpassword123"

            teacher_token = await _token_for(client, teacher_email, password, role="teacher")
            await _ensure_user(client, student_email, password, role="student")

            create_resp = await client.post(
                "/classes/",
                json={"name": "Analytics Class"},
                headers=_auth_headers(teacher_token),
            )
            assert create_resp.status_code == 201, create_resp.text
            enroll_code = create_resp.json()["enroll_code"]

            student_token = await _token_for(client, student_email, password, role="student")
            await client.post(
                "/classes/join",
                json={"enroll_code": enroll_code},
                headers=_auth_headers(student_token),
            )

            analytics_resp = await client.get(
                "/teachers/analytics",
                headers=_auth_headers(teacher_token),
            )
            assert analytics_resp.status_code == 200, analytics_resp.text
            data = analytics_resp.json()
            assert data["total_students"] >= 1
            managed_emails = {s["email"] for s in data["student_analytics"]}
            assert student_email in managed_emails

    _run(_test())
