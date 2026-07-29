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


def _set_auth(role: str, email: str):
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
    yield dict(client=client, users=users, orgs=orgs)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_create_org_stores_tagline(setup):
    c = setup["client"]
    r = c.post("/orgs", json={"name": "Acme", "brand_name": "Acme",
        "tier": "pro", "seats_total": 5, "billing_cycle": "monthly", "tagline": "Best coaching"})
    assert r.status_code == 201, r.text
    org_id = r.json()["org_id"]
    org = setup["orgs"].docs["1"]
    assert org["tagline"] == "Best coaching"


def test_update_org_tagline_and_brand(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "tier": "pro", "seats_total": 5, "billing_cycle": "monthly"})
    r = c.patch("/orgs/", json={"brand_name": "Acme Coaching", "tagline": "We teach"})
    assert r.status_code == 200, r.text
    org = setup["orgs"].docs["1"]
    assert org["brand_name"] == "Acme Coaching"
    assert org["tagline"] == "We teach"


def test_branding_endpoint_requires_auth(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "tier": "pro", "seats_total": 5, "billing_cycle": "monthly"})
    org_id = setup["orgs"].docs["1"]["org_id"]
    app.dependency_overrides.pop(get_current_user_with_role, None)
    r = c.get(f"/orgs/{org_id}/branding")
    assert r.status_code == 401  # no token -> get_current_user_with_role rejects


def test_branding_endpoint_returns_public_fields(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "brand_name": "Acme",
        "tier": "pro", "seats_total": 5, "billing_cycle": "monthly", "tagline": "Hi"})
    org_id = setup["orgs"].docs["1"]["org_id"]
    r = c.get(f"/orgs/{org_id}/branding")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["name"] == "Acme"
    assert b["brand_name"] == "Acme"
    assert b["tagline"] == "Hi"
    assert "logo_url" in b
    assert "logo_file_path" not in b  # internal path must not leak
