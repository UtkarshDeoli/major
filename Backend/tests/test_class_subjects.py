import pytest
from bson import ObjectId

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
from src.main import app
from fastapi.testclient import TestClient


class _FakeCursor:
    """Async cursor-like wrapper over a list for Motor find().sort().to_list()."""

    def __init__(self, docs):
        self._docs = docs

    def sort(self, *args, **kwargs):
        return self

    async def to_list(self, length=None):
        return self._docs[:length] if length is not None else list(self._docs)


class _FakeColl:
    def __init__(self):
        self.docs = {}
        self._i = 0

    def _match(self, doc_val, q_val, key):
        if doc_val == q_val:
            return True
        if key == "_id":
            return str(doc_val) == str(q_val)
        return False

    async def find_one(self, q):
        for d in self.docs.values():
            if all(self._match(d.get(k), v, k) for k, v in q.items()):
                return dict(d)
        return None

    def find(self, q=None):
        q = q or {}
        results = [dict(d) for d in self.docs.values() if all(self._match(d.get(k), v, k) for k, v in q.items())]
        return _FakeCursor(results)

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
        for d in self.docs.values():
            if all(self._match(d.get(k), v, k) for k, v in q.items()):
                if "$set" in op:
                    d.update(op["$set"])
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

    async def delete_one(self, q):
        for k, d in list(self.docs.items()):
            if all(self._match(d.get(kk), v, kk) for kk, v in q.items()):
                del self.docs[k]
                return

    async def delete_many(self, q):
        for k, d in list(self.docs.items()):
            if all(self._match(d.get(kk), v, kk) for kk, v in q.items()):
                del self.docs[k]


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl()
    classes = _FakeColl()
    subjects = _FakeColl()
    materials = _FakeColl()
    pdfs = _FakeColl()
    chunks = _FakeColl()

    coll_map = {
        "users_collection": users,
        "classes_collection": classes,
        "class_subjects_collection": subjects,
        "class_materials_collection": materials,
        "pdfs_collection": pdfs,
        "document_chunks_collection": chunks,
    }
    for name, coll in coll_map.items():
        monkeypatch.setattr(ds, name, coll)

    svc = __import__("importlib").import_module("src.services.class_subject_service")
    for name, coll in coll_map.items():
        if hasattr(svc, name):
            monkeypatch.setattr(svc, name, coll)
    cms = __import__("importlib").import_module("src.services.class_material_service")
    for name, coll in coll_map.items():
        if hasattr(cms, name):
            monkeypatch.setattr(cms, name, coll)
    monkeypatch.setattr(cms.vector_store, "delete_document_chunks", lambda user_id, doc_id: None)

    _set_auth("teacher", "t1@x.com")
    c = TestClient(app)

    # seed a class with teacher_ids
    class_id = str(ObjectId())
    from datetime import datetime, timezone
    classes.docs["1"] = {
        "_id": class_id,
        "name": "JEE",
        "teacher_id": "t1@x.com",
        "teacher_ids": ["t1@x.com"],
        "student_emails": [],
        "subject_ids": [],
        "org_id": "org-9",
        "enroll_code": "JEE123",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    yield dict(users=users, classes=classes, subjects=subjects, class_id=class_id, client=c)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_create_subject(setup):
    r = setup["client"].post(f"/classes/{setup['class_id']}/subjects", json={"name": "Physics", "icon": "atom"})
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["name"] == "Physics"
    assert data["class_id"] == setup["class_id"]
    assert data["id"] in setup["classes"].docs["1"].get("subject_ids", [])


def test_list_subjects(setup):
    c = setup["client"]
    cid = setup["class_id"]
    c.post(f"/classes/{cid}/subjects", json={"name": "Physics"})
    c.post(f"/classes/{cid}/subjects", json={"name": "Chemistry"})
    r = c.get(f"/classes/{cid}/subjects")
    assert r.status_code == 200
    assert len(r.json()["subjects"]) == 2


def test_delete_subject(setup):
    c = setup["client"]
    cid = setup["class_id"]
    sub = c.post(f"/classes/{cid}/subjects", json={"name": "Physics"}).json()
    r = c.delete(f"/classes/{cid}/subjects/{sub['id']}")
    assert r.status_code == 200
    assert sub["id"] not in setup["classes"].docs["1"].get("subject_ids", [])


def test_outsider_cannot_create_subject(setup):
    setup["users"].docs["2"] = {"email": "bad@x.com", "role": "teacher", "org_id": "org-other", "member_role": "teacher"}
    _set_auth("teacher", "bad@x.com")
    r = setup["client"].post(f"/classes/{setup['class_id']}/subjects", json={"name": "X"})
    assert r.status_code == 403
