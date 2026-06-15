import asyncio

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


async def _token_for(client: httpx.AsyncClient, email: str, password: str, role: str = "student"):
    """Return a bearer token for the given email, signing up first if needed."""
    signup_resp = await client.post(
        "/auth/signup",
        json={"email": email, "password": password, "role": role},
    )
    if signup_resp.status_code not in (200, 201):
        # Assume the user already exists and log in instead.
        login_resp = await client.post(
            "/auth/login",
            data={"username": email, "password": password},
        )
        assert login_resp.status_code == 200, login_resp.text
        return login_resp.json()["access_token"]

    # After signup, log in to obtain the token.
    login_resp = await client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    assert login_resp.status_code == 200, login_resp.text
    return login_resp.json()["access_token"]


def _auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def test_teacher_can_manage_existing_student():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            teacher_email = "teacher.manage@example.com"
            student_email = "student.managed@example.com"
            password = "testpassword123"

            # Ensure both users exist.
            teacher_token = await _token_for(client, teacher_email, password, role="teacher")
            await _token_for(client, student_email, password, role="student")

            manage_resp = await client.post(
                "/teachers/students/manage",
                json={"student_email": student_email},
                headers=_auth_headers(teacher_token),
            )
            assert manage_resp.status_code == 200, manage_resp.text
            body = manage_resp.json()
            assert body["student_email"] == student_email
            assert body["teacher_id"] == teacher_email

            student_token = await _token_for(client, student_email, password, role="student")
            me_resp = await client.get("/auth/me", headers=_auth_headers(student_token))
            assert me_resp.status_code == 200, me_resp.text
            assert me_resp.json()["teacher_id"] == teacher_email

    _run(_test())


def test_teacher_cannot_manage_nonexistent_student():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            teacher_email = "teacher.nostudent@example.com"
            password = "testpassword123"
            teacher_token = await _token_for(client, teacher_email, password, role="teacher")

            resp = await client.post(
                "/teachers/students/manage",
                json={"student_email": "missing.student@example.com"},
                headers=_auth_headers(teacher_token),
            )
            assert resp.status_code == 404

    _run(_test())


def test_teacher_can_unmanage_student():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            teacher_email = "teacher.unmanage@example.com"
            student_email = "student.unmanaged@example.com"
            password = "testpassword123"

            teacher_token = await _token_for(client, teacher_email, password, role="teacher")
            await _token_for(client, student_email, password, role="student")

            # Link the student first.
            link_resp = await client.post(
                "/teachers/students/manage",
                json={"student_email": student_email},
                headers=_auth_headers(teacher_token),
            )
            assert link_resp.status_code == 200, link_resp.text

            # Then unlink them.
            unlink_resp = await client.request(
                "DELETE",
                "/teachers/students/manage",
                json={"student_email": student_email},
                headers=_auth_headers(teacher_token),
            )
            assert unlink_resp.status_code == 200, unlink_resp.text
            assert unlink_resp.json()["teacher_id"] is None

            student_token = await _token_for(client, student_email, password, role="student")
            me_resp = await client.get("/auth/me", headers=_auth_headers(student_token))
            assert me_resp.status_code == 200, me_resp.text
            assert me_resp.json()["teacher_id"] is None

    _run(_test())


def test_teacher_analytics_returns_managed_students():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            teacher_email = "teacher.analytics@example.com"
            student_email = "student.analytics@example.com"
            password = "testpassword123"

            teacher_token = await _token_for(client, teacher_email, password, role="teacher")
            await _token_for(client, student_email, password, role="student")

            link_resp = await client.post(
                "/teachers/students/manage",
                json={"student_email": student_email},
                headers=_auth_headers(teacher_token),
            )
            assert link_resp.status_code == 200, link_resp.text

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
