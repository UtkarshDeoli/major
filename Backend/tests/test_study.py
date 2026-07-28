"""Tests for the study router: focus sessions, study plans, and progress updates."""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from httpx import ASGITransport
import pytest
from bson import ObjectId

from src.core.data_store import (
    users_collection,
    focus_sessions_collection,
    study_plans_collection,
)
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

    signup_resp = await client.post(
        "/auth/signup",
        json={"email": email, "password": password},
    )
    assert signup_resp.status_code in (200, 201), signup_resp.text


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


def _unique(prefix: str) -> str:
    return f"{prefix}.{uuid.uuid4().hex[:8]}@example.com"


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


def _fake_plan_json(num_weeks: int = 2) -> str:
    weeks = []
    for w in range(1, num_weeks + 1):
        days = []
        for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            days.append({
                "day": day_name,
                "tasks": [
                    {
                        "subject": "Physics",
                        "topic": f"Topic {w}",
                        "activity": "Read and practice",
                        "minutes": 60,
                        "resource": "Notes",
                    }
                ],
            })
        weeks.append({"week": w, "focus": f"Week {w} focus", "days": days})
    import json
    return json.dumps({"weeks": weeks})


def test_focus_session_lifecycle():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("focus")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            # Start a session
            start_resp = await client.post(
                "/study/focus-sessions",
                json={"task": "Practice calculus", "duration_minutes": 25},
                headers=_auth_headers(token),
            )
            assert start_resp.status_code == 200, start_resp.text
            session = start_resp.json()
            assert session["task"] == "Practice calculus"
            assert session["duration_minutes"] == 25
            assert session["completed"] is False
            session_id = session["session_id"]

            # End the session
            end_resp = await client.patch(
                f"/study/focus-sessions/{session_id}",
                json={"completed": True, "notes": "Done"},
                headers=_auth_headers(token),
            )
            assert end_resp.status_code == 200, end_resp.text
            assert end_resp.json()["completed"] is True

            # Stats should reflect the completed session
            stats_resp = await client.get("/study/focus-stats", headers=_auth_headers(token))
            assert stats_resp.status_code == 200, stats_resp.text
            stats = stats_resp.json()
            assert stats["sessions_count"] >= 1
            assert stats["completed_sessions"] >= 1
            assert stats["total_minutes"] >= 0

    _run(_test())


def test_create_and_list_study_plan(monkeypatch):
    async def _test():
        from src.services import gemini_service as gs

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("plan")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            fake_text = _fake_plan_json(num_weeks=3)
            monkeypatch.setattr(gs.gemini_service.model, "generate_content", lambda prompt: _FakeGeminiResponse(fake_text))

            create_resp = await client.post(
                "/study/plans",
                json={
                    "title": "JEE Sprint",
                    "exam_date": "2026-08-15",
                    "subjects": ["Physics", "Chemistry", "Maths"],
                    "weak_topics": ["Calculus"],
                    "hours_per_day": 5,
                    "num_weeks": 3,
                },
                headers=_auth_headers(token),
            )
            assert create_resp.status_code == 200, create_resp.text
            created = create_resp.json()
            assert created["title"] == "JEE Sprint"
            assert created["num_weeks"] == 3
            assert len(created["plan"]["weeks"]) == 3
            plan_id = created["plan_id"]

            list_resp = await client.get("/study/plans", headers=_auth_headers(token))
            assert list_resp.status_code == 200, list_resp.text
            plans = list_resp.json()["plans"]
            assert any(p["plan_id"] == plan_id for p in plans)

    _run(_test())


def test_update_plan_progress_by_day_name():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("progress")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            # Seed a plan directly to avoid Gemini calls
            plan_doc = {
                "user_id": email,
                "title": "Progress Test",
                "exam_date": None,
                "subjects": ["Physics"],
                "weak_topics": [],
                "hours_per_day": 4,
                "num_weeks": 1,
                "plan": {
                    "weeks": [
                        {
                            "week": 1,
                            "focus": "Test week",
                            "days": [
                                {
                                    "day": "Monday",
                                    "tasks": [
                                        {
                                            "subject": "Physics",
                                            "topic": "Kinematics",
                                            "activity": "Solve problems",
                                            "minutes": 60,
                                            "completed": False,
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            result = await study_plans_collection.insert_one(plan_doc)
            plan_id = str(result.inserted_id)

            patch_resp = await client.patch(
                f"/study/plans/{plan_id}/progress",
                params={"week": 1, "day": "Monday", "task_index": 0, "completed": "true"},
                headers=_auth_headers(token),
            )
            assert patch_resp.status_code == 200, patch_resp.text

            updated = await study_plans_collection.find_one({"_id": ObjectId(plan_id)})
            assert updated["plan"]["weeks"][0]["days"][0]["tasks"][0]["completed"] is True

            # Invalid day should 400, not 500
            bad_resp = await client.patch(
                f"/study/plans/{plan_id}/progress",
                params={"week": 1, "day": "NotADay", "task_index": 0, "completed": "true"},
                headers=_auth_headers(token),
            )
            assert bad_resp.status_code == 400, bad_resp.text

    _run(_test())


def test_delete_study_plan():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("delete")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            plan_doc = {
                "user_id": email,
                "title": "Delete Me",
                "exam_date": None,
                "subjects": [],
                "weak_topics": [],
                "hours_per_day": 4,
                "num_weeks": 1,
                "plan": {"weeks": []},
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            result = await study_plans_collection.insert_one(plan_doc)
            plan_id = str(result.inserted_id)

            del_resp = await client.delete(f"/study/plans/{plan_id}", headers=_auth_headers(token))
            assert del_resp.status_code == 200, del_resp.text
            assert del_resp.json()["deleted"] is True

            gone = await study_plans_collection.find_one({"_id": ObjectId(plan_id)})
            assert gone is None

    _run(_test())


def test_study_plan_limit_enforced():
    async def _test():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("limit")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            # Seed 5 study plans to hit the Starter limit of 5
            for i in range(5):
                await study_plans_collection.insert_one({
                    "user_id": email,
                    "title": f"Plan {i}",
                    "exam_date": None,
                    "subjects": [],
                    "weak_topics": [],
                    "hours_per_day": 4,
                    "num_weeks": 1,
                    "plan": {"weeks": []},
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                })

            # The next creation attempt should be blocked with 402
            resp = await client.post(
                "/study/plans",
                json={
                    "title": "Over limit",
                    "subjects": [],
                    "weak_topics": [],
                    "hours_per_day": 4,
                    "num_weeks": 1,
                },
                headers=_auth_headers(token),
            )
            assert resp.status_code == 402, resp.text
            body = resp.json()
            assert body["detail"]["resource"] == "study_plan"

    _run(_test())


def test_study_plan_rate_limited(monkeypatch):
    async def _test():
        from src.services import gemini_service as gs

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("rate")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            fake_text = _fake_plan_json(num_weeks=1)
            monkeypatch.setattr(gs.gemini_service.model, "generate_content", lambda prompt: _FakeGeminiResponse(fake_text))

            codes = []
            for _ in range(12):
                r = await client.post(
                    "/study/plans",
                    json={
                        "title": "Rate test",
                        "subjects": [],
                        "weak_topics": [],
                        "hours_per_day": 4,
                        "num_weeks": 1,
                    },
                    headers=_auth_headers(token),
                )
                codes.append(r.status_code)

            assert any(c in (402, 429) for c in codes), f"expected throttling, got {codes}"

    _run(_test())
