import uuid
import pytest
import importlib
from bson import ObjectId

import src.core.plan_enforcement as pe
import src.core.data_store as ds
cr = importlib.import_module("src.routers.class_router")
from src.core.security import get_current_user_with_role
from src.main import app
from fastapi.testclient import TestClient


class _FakeColl:
    def __init__(self):
        self.docs = {}
        self._i = 0

    async def find_one(self, q):
        def _match(doc_val, q_val, key):
            if doc_val == q_val:
                return True
            if key == "_id":
                return str(doc_val) == str(q_val)
            return False
        for d in self.docs.values():
            if all(_match(d.get(k), v, k) for k, v in q.items()):
                return dict(d)
        return None

    async def find(self, q=None):
        def _match(doc_val, q_val, key):
            if doc_val == q_val:
                return True
            if key == "_id":
                return str(doc_val) == str(q_val)
            return False
        return [dict(d) for d in self.docs.values() if all(_match(d.get(k), v, k) for k, v in (q or {}).items())]

    async def insert_one(self, doc):
        self._i += 1
        d = dict(doc)
        oid = str(ObjectId())
        d["_id"] = oid
        self.docs[str(self._i)] = d
        class R:
            inserted_id = oid
        return R()

    async def update_one(self, q, op, upsert=False):
        def _match(doc_val, q_val, key):
            if doc_val == q_val:
                return True
            if key == "_id":
                return str(doc_val) == str(q_val)
            return False
        for d in self.docs.values():
            if all(_match(d.get(k), v, k) for k, v in q.items()):
                d.update(op.get("$set", {}))
                if "$addToSet" in op:
                    for k, v in op["$addToSet"].items():
                        arr = d.setdefault(k, [])
                        if isinstance(v, dict) and "$each" in v:
                            for item in v["$each"]:
                                if item not in arr:
                                    arr.append(item)
                        elif v not in arr:
                            arr.append(v)
                if "$pull" in op:
                    for k, v in op["$pull"].items():
                        arr = d.get(k, [])
                        if v in arr:
                            arr.remove(v)
                return


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); classes = _FakeColl(); subs = _FakeColl(); orgs = _FakeColl()
    for name, coll in [
        ("users_collection", users), ("classes_collection", classes),
        ("subscriptions_collection", subs), ("organizations_collection", orgs),
        ("mock_test_submissions_collection", _FakeColl()),
    ]:
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(pe, name, coll, raising=False)
        monkeypatch.setattr(cr, name, coll, raising=False)
    orgs.docs["1"] = {"org_id": "org-9", "tier": "pro", "status": "active"}
    users.docs["1"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    users.docs["2"] = {"email": "t2@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")
    yield dict(users=users, classes=classes)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_create_class_sets_org_and_teacher_ids(setup):
    c = TestClient(app)
    r = c.post("/classes/", json={"name": "JEE Batch"})
    assert r.status_code == 201, r.text
    cls = setup["classes"].docs["1"]
    assert cls["org_id"] == "org-9"
    assert cls["teacher_ids"] == ["t1@x.com"]


def test_add_co_teacher(setup):
    c = TestClient(app)
    c.post("/classes/", json={"name": "JEE Batch"})
    cls_id = setup["classes"].docs["1"]["_id"]
    r = c.post(f"/classes/{cls_id}/teachers", json={"teacher_email": "t2@x.com"})
    assert r.status_code == 200, r.text
    assert "t2@x.com" in setup["classes"].docs["1"]["teacher_ids"]


def test_non_teacher_in_org_cannot_add_co_teacher(setup):
    setup["users"].docs["3"] = {"email": "outsider@x.com", "role": "teacher", "org_id": "org-other", "member_role": "teacher"}
    c = TestClient(app)
    c.post("/classes/", json={"name": "JEE Batch"})
    cls_id = setup["classes"].docs["1"]["_id"]
    _set_auth("teacher", "outsider@x.com")
    r = c.post(f"/classes/{cls_id}/teachers", json={"teacher_email": "t2@x.com"})
    assert r.status_code == 403
