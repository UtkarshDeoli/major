import math
from datetime import datetime, timezone

import pytest

import src.core.plan_enforcement as pe
import src.core.data_store as ds


class _FakeColl:
    """Minimal async Mongo collection mock backed by an in-memory dict."""

    def __init__(self):
        self.docs = {}
        self._i = 0

    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                return dict(d)
        return None

    async def find(self, q=None):
        q = q or {}
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]

    async def insert_one(self, doc):
        self._i += 1
        self.docs[str(self._i)] = dict(doc)

        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                if "$inc" in op:
                    for k, v in op["$inc"].items():
                        d[k] = d.get(k, 0) + v
                if "$set" in op:
                    d.update(op["$set"])

    async def count_documents(self, q):
        return len(await self.find(q))


@pytest.fixture
def isolated(monkeypatch):
    users = _FakeColl(); subs = _FakeColl(); orgs = _FakeColl()
    mt = _FakeColl(); subs_mt = _FakeColl(); fc = _FakeColl(); aim = _FakeColl()
    pdfs = _FakeColl(); classes = _FakeColl(); decks = _FakeColl(); usage = _FakeColl()
    for name, coll in [
        ("users_collection", users), ("subscriptions_collection", subs),
        ("organizations_collection", orgs), ("mock_tests_collection", mt),
        ("mock_test_submissions_collection", subs_mt), ("flashcards_collection", fc),
        ("ai_materials_collection", aim), ("pdfs_collection", pdfs),
        ("classes_collection", classes), ("flashcard_decks_collection", decks),
        ("usage_events_collection", usage),
    ]:
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(pe, name, coll)
    return dict(users=users, subs=subs, orgs=orgs, mt=mt, fc=fc, aim=aim,
                pdfs=pdfs, classes=classes, decks=decks, usage=usage)


async def test_free_plan_when_no_subscription_and_no_org(isolated):
    plan, source, org_id = await pe.get_effective_plan("a@x.com")
    assert plan == "starter" and source == "free" and org_id is None


async def test_own_subscription_wins_over_org(isolated):
    isolated["users"].docs["1"] = {"email": "a@x.com", "org_id": "org1"}
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "premium"}
    isolated["orgs"].docs["1"] = {"org_id": "org1", "tier": "pro", "status": "active",
                                   "owner_user_id": "b@x.com"}
    plan, source, org_id = await pe.get_effective_plan("a@x.com")
    assert (plan, source) == ("premium", "self")


async def test_org_tier_when_no_own_subscription(isolated):
    isolated["users"].docs["1"] = {"email": "a@x.com", "org_id": "org1"}
    isolated["orgs"].docs["1"] = {"org_id": "org1", "tier": "premium", "status": "active"}
    plan, source, org_id = await pe.get_effective_plan("a@x.com")
    assert (plan, source, org_id) == ("premium", "org", "org1")


async def test_inactive_subscription_falls_through_to_free(isolated):
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "cancelled", "plan": "pro"}
    plan, source, _ = await pe.get_effective_plan("a@x.com")
    assert (plan, source) == ("starter", "free")


async def test_usage_counts_mock_tests_this_month(isolated):
    now = datetime.now(timezone.utc)
    isolated["mt"].docs["1"] = {"user_id": "a@x.com", "created_at": now}
    isolated["mt"].docs["2"] = {"created_by": "a@x.com", "created_at": now}
    used = await pe.get_usage("a@x.com", "mock_test")
    assert used == 2


async def test_enforce_limit_402_when_over(isolated):
    from fastapi import HTTPException
    now = datetime.now(timezone.utc)
    for i in range(3):
        isolated["mt"].docs[str(i)] = {"user_id": "a@x.com", "created_at": now}
    isolated["users"].docs["1"] = {"email": "a@x.com"}
    dep = pe.enforce_limit("mock_test")
    user_info = {"email": "a@x.com", "user": {"email": "a@x.com"}}
    with pytest.raises(HTTPException) as exc:
        await dep(user_info=user_info)
    assert exc.value.status_code == 402
    assert exc.value.detail["resource"] == "mock_test"
    assert exc.value.detail["used"] == 3
    assert exc.value.detail["limit"] == 3
    assert exc.value.detail["plan"] == "starter"


async def test_enforce_limit_premium_no_402(isolated):
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "premium"}
    now = datetime.now(timezone.utc)
    for i in range(100):
        isolated["mt"].docs[str(i)] = {"user_id": "a@x.com", "created_at": now}
    dep = pe.enforce_limit("mock_test")
    user_info = {"email": "a@x.com", "user": {"email": "a@x.com"}}
    result = await dep(user_info=user_info)
    assert result["email"] == "a@x.com"


async def test_increment_usage_upserts_chat(isolated):
    await pe.increment_usage("a@x.com", "chat_message")
    await pe.increment_usage("a@x.com", "chat_message")
    used = await pe.get_usage("a@x.com", "chat_message")
    assert used == 2


async def test_increment_usage_noop_for_derived_resources(isolated):
    # mock_test usage is derived from the mock_tests collection, not usage_events
    await pe.increment_usage("a@x.com", "mock_test")
    assert await pe.get_usage("a@x.com", "chat_message") == 0


def test_mock_test_generate_402_on_free_limit(monkeypatch):
    """The wired /mock-tests/generate endpoint must 402 when the free limit is hit.

    The 402 fires in the enforce_limit dependency BEFORE the handler body runs,
    so no Gemini call is made.
    """
    from datetime import datetime, timezone
    from fastapi.testclient import TestClient
    from src.core.security import get_current_user, get_current_user_with_role
    from src.main import app

    now = datetime.now(timezone.utc)
    mt = _FakeColl(); users = _FakeColl(); subs = _FakeColl()
    for i in range(3):
        mt.docs[str(i)] = {"user_id": "u@x.com", "created_at": now}
    users.docs["1"] = {"email": "u@x.com", "role": "student"}

    import src.core.data_store as ds2
    import src.core.plan_enforcement as pe2
    for mod in (ds2, pe2):
        monkeypatch.setattr(mod, "mock_tests_collection", mt, raising=False)
        monkeypatch.setattr(mod, "users_collection", users, raising=False)
        monkeypatch.setattr(mod, "subscriptions_collection", subs, raising=False)
        monkeypatch.setattr(mod, "organizations_collection", _FakeColl(), raising=False)

    def _auth():
        return {"email": "u@x.com", "user": {"email": "u@x.com", "role": "student"}}
    app.dependency_overrides[get_current_user_with_role] = _auth
    app.dependency_overrides[get_current_user] = lambda: "u@x.com"
    try:
        client = TestClient(app)
        r = client.post("/mock-tests/generate", json={
            "syllabus_pdf_id": "x", "question_paper_pdf_ids": []})
        assert r.status_code == 402, r.text
        assert r.json()["detail"]["resource"] == "mock_test"
    finally:
        app.dependency_overrides.pop(get_current_user_with_role, None)
        app.dependency_overrides.pop(get_current_user, None)