import sys

import pytest

import src.services.org_service as os_
import src.services.billing_service as bs
import src.core.data_store as ds
from src.core.security import get_current_user_with_role
from src.services.billing_service import FakeRazorpayClient
from src.main import app
from fastapi.testclient import TestClient


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
                if "$inc" in op:
                    for k, v in op["$inc"].items():
                        d[k] = d.get(k, 0) + v
                return
        if upsert:
            self._i += 1
            doc = {k: v for k, v in q.items()}
            doc.update(op.get("$set", {}))
            self.docs[str(self._i)] = doc


def _set_auth(role: str, email: str):
    """Override the auth dependency to return a user with the given role."""
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); orgs = _FakeColl(); invites = _FakeColl(); subs = _FakeColl()
    for name, coll in [("users_collection", users), ("organizations_collection", orgs),
                       ("org_invites_collection", invites), ("subscriptions_collection", subs)]:
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(os_, name, coll, raising=False)
    monkeypatch.setattr(bs, "get_client", lambda: FakeRazorpayClient(secret="k"))
    users.docs["1"] = {"email": "owner@x.com", "role": "subadmin"}
    _set_auth("subadmin", "owner@x.com")
    client = TestClient(app)
    yield dict(client=client, users=users, orgs=orgs, invites=invites)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_create_org(setup):
    r = setup["client"].post("/orgs", json={"name": "Acme Coaching", "brand_name": "Acme",
        "tier": "pro", "seats_total": 10, "billing_cycle": "monthly"})
    assert r.status_code == 201, r.text
    assert r.json()["org_id"]
    assert setup["users"].docs["1"]["role"] == "subadmin"
    assert setup["users"].docs["1"]["org_id"] == r.json()["org_id"]


def test_create_org_conflict_if_already_owner(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "brand_name": "Acme", "tier": "pro",
        "seats_total": 5, "billing_cycle": "monthly"})
    r = c.post("/orgs", json={"name": "Second", "brand_name": "Second", "tier": "pro",
        "seats_total": 5, "billing_cycle": "monthly"})
    assert r.status_code == 409


def test_invite_then_enroll_consumes_seat(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "brand_name": "Acme", "tier": "pro",
        "seats_total": 1, "billing_cycle": "monthly"})
    inv = c.post("/orgs/invite", json={"member_role": "student"}).json()
    code = inv["code"]

    setup["users"].docs["2"] = {"email": "stu@x.com", "role": "student"}
    _set_auth("student", "stu@x.com")
    r = c.post(f"/orgs/enroll/{code}")
    assert r.status_code == 200 and r.json()["member_role"] == "student"

    # second enroll should 402 (seat full)
    setup["users"].docs["3"] = {"email": "stu2@x.com", "role": "student"}
    _set_auth("student", "stu2@x.com")
    _set_auth("subadmin", "owner@x.com")
    inv2 = c.post("/orgs/invite", json={"member_role": "student"}).json()
    _set_auth("student", "stu2@x.com")
    r2 = c.post(f"/orgs/enroll/{inv2['code']}")
    assert r2.status_code == 402


def test_non_subadmin_cannot_create_org():
    _set_auth("student", "s@x.com")
    c = TestClient(app)
    try:
        r = c.post("/orgs", json={"name": "X", "brand_name": "X", "tier": "pro",
            "seats_total": 1, "billing_cycle": "monthly"})
        assert r.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user_with_role, None)


def test_remove_member_frees_seat(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "brand_name": "Acme", "tier": "pro",
        "seats_total": 2, "billing_cycle": "monthly"})
    inv = c.post("/orgs/invite", json={"member_role": "student"}).json()
    setup["users"].docs["2"] = {"email": "stu@x.com", "role": "student"}
    _set_auth("student", "stu@x.com")
    c.post(f"/orgs/enroll/{inv['code']}")
    _set_auth("subadmin", "owner@x.com")
    r = c.delete("/orgs/members/stu@x.com")
    assert r.status_code == 200 and r.json()["removed"] == "stu@x.com"
    assert setup["users"].docs["2"]["org_id"] is None