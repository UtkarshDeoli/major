import sys
from datetime import datetime, timezone

import pytest

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
from src.main import app
from fastapi.testclient import TestClient

ar = sys.modules["src.routers.admin_router"]


class _FakeColl:
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

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return
        if upsert:
            self._i += 1
            doc = {k: v for k, v in q.items()}
            doc.update(op.get("$set", {}))
            self.docs[str(self._i)] = doc


def _set_auth(role: str, email: str):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def admin(monkeypatch):
    users = _FakeColl(); orgs = _FakeColl(); subs = _FakeColl(); pays = _FakeColl()
    for name, coll in [("users_collection", users), ("organizations_collection", orgs),
                       ("subscriptions_collection", subs), ("payments_collection", pays)]:
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(ar, name, coll)
    _set_auth("admin", "admin@x.com")
    client = TestClient(app)
    yield dict(client=client, users=users, orgs=orgs, subs=subs, pays=pays)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_non_admin_forbidden():
    _set_auth("student", "s@x.com")
    c = TestClient(app)
    try:
        assert c.get("/admin/users").status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user_with_role, None)


def test_list_users(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    r = admin["client"].get("/admin/users")
    assert r.status_code == 200
    assert any(u["email"] == "a@x.com" for u in r.json()["users"])


def test_list_users_filter_by_role(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    admin["users"].docs["2"] = {"email": "b@x.com", "role": "teacher"}
    r = admin["client"].get("/admin/users?role=teacher")
    assert r.status_code == 200
    emails = [u["email"] for u in r.json()["users"]]
    assert emails == ["b@x.com"]


def test_change_role(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    r = admin["client"].patch("/admin/users/a@x.com/role", json={"role": "teacher"})
    assert r.status_code == 200
    assert admin["users"].docs["1"]["role"] == "teacher"


def test_manual_activate(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    r = admin["client"].post("/admin/subscriptions/a@x.com/activate", json={"plan": "pro", "days": 30})
    assert r.status_code == 200
    sub = next(d for d in admin["subs"].docs.values() if d.get("user_id") == "a@x.com")
    assert sub["status"] == "active" and sub["plan"] == "pro"


def test_analytics(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    admin["users"].docs["2"] = {"email": "b@x.com", "role": "teacher"}
    admin["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "pro"}
    admin["pays"].docs["1"] = {"user_id": "a@x.com", "amount": 29900, "status": "captured"}
    admin["orgs"].docs["1"] = {"org_id": "o1", "status": "active"}
    r = admin["client"].get("/admin/analytics")
    assert r.status_code == 200
    a = r.json()
    assert a["totals"]["users_by_role"]["student"] == 1
    assert a["totals"]["active_subscriptions"] == 1
    assert a["totals"]["org_count"] == 1
    assert a["totals"]["mrr_paise"] == 29900


def test_suspend_org(admin):
    admin["orgs"].docs["1"] = {"org_id": "o1", "status": "active", "seats_total": 10}
    r = admin["client"].patch("/admin/orgs/o1", json={"status": "suspended"})
    assert r.status_code == 200
    assert admin["orgs"].docs["1"]["status"] == "suspended"