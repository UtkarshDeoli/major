import io
import uuid
import pytest
from bson import ObjectId

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


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); classes = _FakeColl(); subjects = _FakeColl(); materials = _FakeColl(); pdfs = _FakeColl(); decks = _FakeColl(); cards = _FakeColl(); tests = _FakeColl(); chunks = _FakeColl()
    coll_map = {
        "users_collection": users,
        "classes_collection": classes,
        "class_subjects_collection": subjects,
        "class_materials_collection": materials,
        "pdfs_collection": pdfs,
        "flashcard_decks_collection": decks,
        "flashcards_collection": cards,
        "mock_tests_collection": tests,
        "document_chunks_collection": chunks,
        "subscriptions_collection": _FakeColl(),
        "organizations_collection": _FakeColl(),
    }
    for n, c in coll_map.items():
        monkeypatch.setattr(ds, n, c)
        monkeypatch.setattr(pe, n, c, raising=False)

    import importlib
    cr = importlib.import_module("src.routers.class_router")
    css = importlib.import_module("src.services.class_subject_service")
    cms = importlib.import_module("src.services.class_material_service")
    for n, c in coll_map.items():
        if hasattr(cr, n):
            monkeypatch.setattr(cr, n, c)
        if hasattr(css, n):
            monkeypatch.setattr(css, n, c)
        if hasattr(cms, n):
            monkeypatch.setattr(cms, n, c)

    users.docs["1"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")

    monkeypatch.setattr(cms.vector_store, "add_chunks", lambda user_id, chunks: None)
    from src.services.vector_store import VectorStore
    monkeypatch.setattr(VectorStore, "get_embedding_model", lambda self=None: _FakeEncoder())

    c = TestClient(app)
    cid = c.post("/classes/", json={"name": "JEE"}).json()["id"]
    sid = c.post(f"/classes/{cid}/subjects", json={"name": "Physics"}).json()["id"]

    # upload a material with doc_id already set
    r = c.post(f"/classes/{cid}/subjects/{sid}/materials", files={"file": ("notes.pdf", io.BytesIO(b"x"), "application/pdf")})
    mat = r.json()

    # stub Gemini generation
    from src.services import gemini_service as gs
    async def fake_extract_text(pdf_path):
        return "some extracted text"
    monkeypatch.setattr(gs.gemini_service, "extract_text_from_pdf", fake_extract_text)

    cmr = importlib.import_module("src.routers.class_material_router")
    async def fake_generate_flashcards(content, num_cards, subject=None):
        return [{"front": "Q?", "back": "A."}] * 2
    monkeypatch.setattr(cmr, "generate_flashcards", fake_generate_flashcards)

    async def fake_generate(doc_ids, num_mcq, num_text, total_marks, difficulty_level, user_id, subject=None, class_id=None, class_subject_id=None, created_by=None, grading_mode="auto"):
        from datetime import datetime, timezone
        from src.core.models import MockTestResponse, MockTestQuestion
        return MockTestResponse(
            test_id="test-1", title="T", total_marks=30, time_limit=30, user_id=user_id,
            questions=[MockTestQuestion(id="1", type="mcq", question="Q", options=["A) a"], correctAnswer="A) a", marks=2)],
            created_at=datetime.now(timezone.utc),
            created_by=created_by, class_id=class_id, class_subject_id=class_subject_id,
        )
    monkeypatch.setattr(cmr, "generate_mock_test_from_docs_service", fake_generate)

    yield dict(client=c, class_id=cid, subject_id=sid, material_id=mat["id"], decks=decks, tests=tests)
    app.dependency_overrides.pop(get_current_user_with_role, None)


class _FakeEncoder:
    def encode(self, text):
        return [0.0] * 384


def test_generate_flashcards_from_class_material(setup):
    r = setup["client"].post(f"/classes/{setup['class_id']}/materials/{setup['material_id']}/generate-flashcards")
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["card_count"] == 2
    # stored deck has class_id
    deck = list(setup["decks"].docs.values())[0]
    assert deck["class_id"] == setup["class_id"]


def test_generate_mock_test_from_class_material(setup):
    r = setup["client"].post(f"/classes/{setup['class_id']}/materials/{setup['material_id']}/generate-mock-test")
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["test_id"] == "test-1"
    assert data["class_id"] == setup["class_id"]
