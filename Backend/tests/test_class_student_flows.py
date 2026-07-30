import pytest
import importlib
from bson import ObjectId

import src.core.data_store as ds
cr = importlib.import_module("src.routers.class_router")
from src.core.security import get_current_user_with_role, get_current_user
from src.main import app
from fastapi.testclient import TestClient


class _FakeCursor:
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
        if key == "student_emails":
            return q_val in (doc_val or [])
        if isinstance(q_val, dict) and "$in" in q_val:
            return doc_val in q_val["$in"]
        return False

    def _matches_query(self, doc, q):
        if "$or" in q:
            return any(self._matches_query(doc, sub) for sub in q["$or"])
        return all(self._match(doc.get(k), v, k) for k, v in q.items())

    async def find_one(self, q):
        for d in self.docs.values():
            if self._matches_query(d, q):
                return dict(d)
        return None

    def find(self, q=None):
        q = q or {}
        results = [dict(d) for d in self.docs.values() if self._matches_query(d, q)]
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
            if self._matches_query(d, q):
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

    async def find_one_and_update(self, q, op, upsert=False, return_document=None):
        for d in self.docs.values():
            if self._matches_query(d, q):
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
                return dict(d)
        return None


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth
    app.dependency_overrides[get_current_user] = lambda: email


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl()
    classes = _FakeColl()
    submissions = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "classes_collection", classes)
    monkeypatch.setattr(ds, "mock_test_submissions_collection", submissions)
    monkeypatch.setattr(cr, "users_collection", users)
    monkeypatch.setattr(cr, "mock_test_submissions_collection", submissions)

    svc = __import__("importlib").import_module("src.services.class_service")
    monkeypatch.setattr(svc, "classes_collection", classes)

    from datetime import datetime, timezone
    class_id = str(ObjectId())
    classes.docs["1"] = {
        "_id": class_id,
        "name": "JEE",
        "teacher_id": "t1@x.com",
        "teacher_ids": ["t1@x.com"],
        "student_emails": [],
        "org_id": "org-9",
        "enroll_code": "JEE123",
        "subject_ids": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    users.docs["1"] = {"email": "s1@x.com", "role": "student", "org_id": "org-9", "member_role": "student"}
    _set_auth("student", "s1@x.com")
    c = TestClient(app)
    yield dict(users=users, classes=classes, class_id=class_id, client=c)
    app.dependency_overrides.pop(get_current_user_with_role, None)
    app.dependency_overrides.pop(get_current_user, None)


def test_student_joins_class_by_enroll_code(setup):
    c = setup["client"]
    r = c.post("/classes/join", json={"enroll_code": "JEE123"})
    assert r.status_code == 200, r.text
    assert r.json()["enrolled"] is True
    assert "s1@x.com" in setup["classes"].docs["1"]["student_emails"]


def test_student_lists_their_classes(setup):
    c = setup["client"]
    c.post("/classes/join", json={"enroll_code": "JEE123"})
    r = c.get("/classes/me")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["classes"]) == 1
    assert data["classes"][0]["enroll_code"] == "JEE123"


def test_student_can_view_enrolled_class(setup):
    c = setup["client"]
    c.post("/classes/join", json={"enroll_code": "JEE123"})
    r = c.get(f"/classes/{setup['class_id']}")
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "JEE"


def test_non_member_student_cannot_view_class(setup):
    setup["classes"].docs["2"] = {
        "_id": str(ObjectId()),
        "name": "NEET",
        "teacher_id": "t2@x.com",
        "teacher_ids": ["t2@x.com"],
        "student_emails": [],
        "org_id": "org-9",
        "enroll_code": "NEET99",
        "subject_ids": [],
        "created_at": setup["classes"].docs["1"]["created_at"],
        "updated_at": setup["classes"].docs["1"]["updated_at"],
    }
    c = setup["client"]
    r = c.get(f"/classes/{setup['classes'].docs['2']['_id']}")
    assert r.status_code == 403


def test_student_gets_class_study_content(setup, monkeypatch):
    from datetime import datetime, timezone
    c = setup["client"]
    cid = setup["class_id"]
    c.post("/classes/join", json={"enroll_code": "JEE123"})

    # stub class content collections
    subjects = _FakeColl()
    subjects.docs["1"] = {"_id": str(ObjectId()), "class_id": cid, "name": "Physics", "created_at": datetime.now(timezone.utc)}
    decks = _FakeColl()
    decks.docs["1"] = {"_id": str(ObjectId()), "class_id": cid, "title": "F1", "card_count": 5, "created_at": datetime.now(timezone.utc)}
    tests = _FakeColl()
    tests.docs["1"] = {"_id": str(ObjectId()), "class_id": cid, "title": "T1", "total_marks": 30, "created_at": datetime.now(timezone.utc)}
    monkeypatch.setattr(ds, "class_subjects_collection", subjects)
    monkeypatch.setattr(ds, "flashcard_decks_collection", decks)
    monkeypatch.setattr(ds, "mock_tests_collection", tests)

    r = c.get(f"/classes/{cid}/content")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["class_id"] == cid
    assert len(data["subjects"]) == 1
    assert len(data["decks"]) == 1
    assert len(data["tests"]) == 1


def test_teacher_gets_class_students(setup):
    c = setup["client"]
    setup["users"].docs["2"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")
    cid = setup["class_id"]
    setup["classes"].docs["1"]["student_emails"] = ["s1@x.com"]
    setup["users"].docs["1"]["name"] = "Student One"

    r = c.get(f"/classes/{cid}/students")
    assert r.status_code == 200, r.text
    assert len(r.json()["students"]) == 1
    assert r.json()["students"][0]["email"] == "s1@x.com"


def test_list_teacher_students_across_classes(setup):
    c = setup["client"]
    from bson import ObjectId
    from datetime import datetime, timezone

    setup["users"].docs["2"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")
    cid = setup["class_id"]
    setup["classes"].docs["1"]["student_emails"] = ["s1@x.com"]

    r = c.get("/classes/students")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["students"]) == 1
    assert data["students"][0]["email"] == "s1@x.com"


def test_class_students_analytics(setup):
    c = setup["client"]
    setup["users"].docs["2"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")
    cid = setup["class_id"]
    setup["classes"].docs["1"]["student_emails"] = ["s1@x.com"]

    r = c.get(f"/classes/{cid}/students/analytics")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["students"]) == 1
    assert data["students"][0]["email"] == "s1@x.com"
    assert data["students"][0]["tests_taken"] == 0


def test_teacher_gets_class_tests(setup, monkeypatch):
    from datetime import datetime, timezone
    c = setup["client"]
    setup["users"].docs["2"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")
    cid = setup["class_id"]

    tests = _FakeColl()
    tests.docs["1"] = {"_id": str(ObjectId()), "class_id": cid, "title": "T1", "total_marks": 30, "created_at": datetime.now(timezone.utc)}
    monkeypatch.setattr(ds, "mock_tests_collection", tests)

    r = c.get(f"/classes/{cid}/tests")
    assert r.status_code == 200, r.text
    assert len(r.json()["tests"]) == 1
    assert r.json()["tests"][0]["title"] == "T1"


def test_student_can_access_class_flashcard_deck(setup, monkeypatch):
    import uuid
    from datetime import datetime, timezone
    c = setup["client"]
    cid = setup["class_id"]
    c.post("/classes/join", json={"enroll_code": "JEE123"})

    deck_id = str(uuid.uuid4())
    decks = _FakeColl()
    decks.docs["1"] = {
        "_id": ObjectId(), "id": deck_id, "class_id": cid, "user_id": "t1@x.com",
        "created_by": "t1@x.com", "title": "F1", "card_count": 1,
        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
    }
    cards = _FakeColl()
    cards.docs["1"] = {
        "_id": ObjectId(), "id": str(ObjectId()), "deck_id": deck_id,
        "front": "Q", "back": "A", "ease": 2, "interval_days": 0, "reps": 0,
        "due_at": datetime.now(timezone.utc), "created_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(ds, "flashcard_decks_collection", decks)
    monkeypatch.setattr(ds, "flashcards_collection", cards)

    r = c.get(f"/flashcards/decks/{deck_id}")
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "F1"


def test_student_can_access_class_mock_test(setup, monkeypatch):
    from datetime import datetime, timezone
    from src.core.models import MockTestQuestion
    c = setup["client"]
    cid = setup["class_id"]
    c.post("/classes/join", json={"enroll_code": "JEE123"})

    test_id = str(ObjectId())
    tests = _FakeColl()
    tests.docs["1"] = {
        "_id": test_id, "test_id": test_id, "class_id": cid, "user_id": "t1@x.com",
        "created_by": "t1@x.com", "title": "T1", "total_marks": 2, "time_limit": 10,
        "difficulty_level": "medium", "grading_mode": "auto", "status": "ready",
        "questions": [{
            "id": "1", "type": "mcq", "question": "Q", "options": ["A) a"],
            "correctAnswer": "A) a", "marks": 2,
        }],
        "created_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(ds, "mock_tests_collection", tests)

    r = c.get(f"/mock-tests/{test_id}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["title"] == "T1"
    assert data["questions"][0]["correctAnswer"] is None
