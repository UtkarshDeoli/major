import io
import pytest
from bson import ObjectId

import src.services.document_processor as dp

import src.core.data_store as ds
import src.core.plan_enforcement as pe
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

    async def insert_many(self, docs):
        inserted_ids = []
        for doc in docs:
            self._i += 1
            d = dict(doc)
            oid = str(ObjectId())
            d["_id"] = oid
            self.docs[str(self._i)] = d
            inserted_ids.append(oid)

        class R:
            pass
        result = R()
        result.inserted_ids = inserted_ids
        return result

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
def setup(monkeypatch, tmp_path):
    users = _FakeColl()
    classes = _FakeColl()
    subjects = _FakeColl()
    materials = _FakeColl()
    pdfs = _FakeColl()
    chunks = _FakeColl()
    subs = _FakeColl()
    orgs = _FakeColl()

    coll_map = {
        "users_collection": users,
        "classes_collection": classes,
        "class_subjects_collection": subjects,
        "class_materials_collection": materials,
        "pdfs_collection": pdfs,
        "document_chunks_collection": chunks,
        "subscriptions_collection": subs,
        "organizations_collection": orgs,
        "mock_test_submissions_collection": _FakeColl(),
    }

    # Patch data_store and plan_enforcement module-level references
    for name, coll in coll_map.items():
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(pe, name, coll, raising=False)

    # Patch module-level references inside routers/services that copy the globals at import time
    import importlib
    cr = importlib.import_module("src.routers.class_router")
    css = importlib.import_module("src.services.class_subject_service")
    cms = importlib.import_module("src.services.class_material_service")

    for name, coll in coll_map.items():
        if hasattr(cr, name):
            monkeypatch.setattr(cr, name, coll)
        if hasattr(css, name):
            monkeypatch.setattr(css, name, coll)
        if hasattr(cms, name):
            monkeypatch.setattr(cms, name, coll)

    orgs.docs["1"] = {"org_id": "org-9", "tier": "pro", "status": "active"}
    users.docs["1"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")

    c = TestClient(app)
    cid = c.post("/classes/", json={"name": "JEE"}).json()["id"]
    sid = c.post(f"/classes/{cid}/subjects", json={"name": "Physics"}).json()["id"]

    # Stub out real ChromaDB + embedding to avoid loading models in tests
    monkeypatch.setattr(cms.vector_store, "add_chunks", lambda user_id, chunks: None)
    monkeypatch.setattr(cms.vector_store, "delete_document_chunks", lambda user_id, doc_id: None)
    from src.services.vector_store import VectorStore
    monkeypatch.setattr(VectorStore, "get_embedding_model", lambda self=None: _FakeEncoder())

    # Stub PDF extraction + chunking so RAG path succeeds in tests
    monkeypatch.setattr(dp, "extract_text_from_pdf", lambda content: ("some text", 1))
    monkeypatch.setattr(dp, "chunk_document", lambda text, doc_type: [{"chunk_index": 0, "content": "chunk", "page": 1, "section": "s"}])
    monkeypatch.setattr(cms, "extract_text_from_pdf", lambda content: ("some text", 1))
    monkeypatch.setattr(cms, "chunk_document", lambda text, doc_type: [{"chunk_index": 0, "content": "chunk", "page": 1, "section": "s"}])

    yield dict(users=users, classes=classes, materials=materials, pdfs=pdfs, chunks=chunks, client=c, class_id=cid, subject_id=sid)
    app.dependency_overrides.pop(get_current_user_with_role, None)


class _FakeEmbedding:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeEncoder:
    def encode(self, text):
        return _FakeEmbedding([0.0] * 384)


def test_upload_class_material(setup):
    c = setup["client"]
    file = io.BytesIO(b"fake pdf content")
    r = c.post(
        f"/classes/{setup['class_id']}/subjects/{setup['subject_id']}/materials",
        files={"file": ("notes.pdf", file, "application/pdf")},
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["name"] == "notes.pdf"
    assert data["class_subject_id"] == setup["subject_id"]
    assert data["doc_id"] is not None
    assert data["rag_indexed"] is True
    stored = next(d for d in setup["materials"].docs.values() if d["_id"] == data["id"])
    assert stored["chunk_count"] == 1


def test_list_and_delete_class_material(setup):
    c = setup["client"]; cid = setup["class_id"]; sid = setup["subject_id"]
    up = c.post(f"/classes/{cid}/subjects/{sid}/materials", files={"file": ("a.pdf", io.BytesIO(b"x"), "application/pdf")}).json()
    lst = c.get(f"/classes/{cid}/subjects/{sid}/materials")
    assert lst.status_code == 200
    assert len(lst.json()["materials"]) == 1
    r = c.delete(f"/classes/{cid}/materials/{up['id']}")
    assert r.status_code == 200
