import os
import pytest

import src.services.org_service as os_
import src.services.billing_service as bs
import src.core.data_store as ds
from src.core.security import get_current_user_with_role
from src.services.billing_service import FakeRazorpayClient
from src.main import app
from src.core import config
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


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch, tmp_path):
    users = _FakeColl(); orgs = _FakeColl(); invites = _FakeColl(); subs = _FakeColl()
    for name, coll in [("users_collection", users), ("organizations_collection", orgs),
                       ("org_invites_collection", invites), ("subscriptions_collection", subs)]:
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(os_, name, coll, raising=False)
    monkeypatch.setattr(bs, "get_client", lambda: FakeRazorpayClient(secret="k"))
    monkeypatch.setattr(config, "UPLOADS_DIR", str(tmp_path))
    monkeypatch.setattr(os_, "UPLOADS_DIR", str(tmp_path), raising=False)
    users.docs["1"] = {"email": "owner@x.com", "role": "subadmin"}
    _set_auth("subadmin", "owner@x.com")
    client = TestClient(app)
    yield dict(client=client, users=users, orgs=orgs, tmp=tmp_path)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_upload_and_fetch_logo(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "tier": "pro", "seats_total": 5, "billing_cycle": "monthly"})
    org_id = setup["orgs"].docs["1"]["org_id"]

    r = c.post(
        "/orgs/logo",
        files={"file": ("logo.png", b"\x89PNG\r\n\x1a\n fake", "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["logo_url"] == f"/orgs/{org_id}/logo"

    # file written to disk
    org = setup["orgs"].docs["1"]
    assert os.path.exists(org["logo_file_path"])

    # public fetch works without auth
    app.dependency_overrides.pop(get_current_user_with_role, None)
    img = c.get(f"/orgs/{org_id}/logo")
    assert img.status_code == 200
    assert img.headers["content-type"].startswith("image/")


def test_logo_404_when_none(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "tier": "pro", "seats_total": 5, "billing_cycle": "monthly"})
    org_id = setup["orgs"].docs["1"]["org_id"]
    app.dependency_overrides.pop(get_current_user_with_role, None)
    r = c.get(f"/orgs/{org_id}/logo")
    assert r.status_code == 404
