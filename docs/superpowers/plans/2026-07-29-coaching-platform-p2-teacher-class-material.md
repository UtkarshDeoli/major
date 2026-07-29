# Orbit Coaching-Platform Reshape — Phase 2 (Teacher Class/Subject/Material + AI Generation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the teacher-side class management experience: dedicated `/classes` and `/classes/[id]` pages, ClassSubject CRUD, class-scoped material upload with full RAG indexing, and one-click AI generation of flashcards + mock tests from any class material — all visible to the class's students in later phases.

**Architecture:** Extend the P1 foundation. Backend adds focused services/routers for class subjects and class materials; material upload reuses the existing extract/chunk/embed/index pipeline from `material_router.py` but stores a `ClassMaterial` record linked to `(class_id, class_subject_id)`. AI generation endpoints read the class material's `doc_id`, call the existing Gemini services, and store the resulting flashcard deck / mock test with `class_id` + `class_subject_id` so they are class-scoped. Frontend reuses the `ai-materials-sidebar.tsx` action pattern (Summary / Flashcards / Mock Test buttons) inside a new `/classes/[id]` tabbed page.

**Tech Stack:** FastAPI, Motor (MongoDB), Pydantic v2, pytest; Next.js 16 App Router, TypeScript, Tailwind, shadcn/ui, axios.

## Global Constraints

- Backend runs from `Backend/` with the venv active: `source venv/bin/activate`. Tests: `pytest tests/ -v`. App: `python -m uvicorn src.main:app --reload --port 8001`.
- Frontend runs from `Frontend/`: `npm run dev` (port 3000), `npm run lint`, `npm run build`. No frontend test runner — frontend tasks are verified with `npm run lint` + `npm run build` + manual browser checks.
- All new Mongo fields are optional with defaults so existing documents keep working — no destructive schema changes.
- Existing collections extended in place, never renamed.
- Roles stay `student | teacher | subadmin | admin`. `subadmin` = coaching admin.
- A class is multi-teacher: `Class.teacher_ids` contains the creator + co-teachers. Any teacher in the class's org may add subjects and materials. Class-material ownership is checked via `teacher_email in cls["teacher_ids"]`.
- Class materials are stored on disk under `uploads/{teacher_email}/` (same as personal materials), indexed into ChromaDB and `document_chunks_collection` keyed by `doc_id`.
- Commit after every task. Branch off `master` first if not already on a feature branch.

## Scope of this plan

Phase 2 covers teacher class management only. Student visibility of class materials/tests is Phase 4.

## File Structure

**Backend — create:**
- `Backend/src/services/class_subject_service.py` — ClassSubject CRUD + ownership checks.
- `Backend/src/services/class_material_service.py` — class-scoped material upload + RAG + list + delete.
- `Backend/src/routers/class_subject_router.py` — `POST/GET/DELETE /classes/{class_id}/subjects`.
- `Backend/src/routers/class_material_router.py` — `POST/GET /classes/{class_id}/subjects/{subject_id}/materials`, `DELETE /classes/{class_id}/materials/{material_id}`, `POST /classes/{class_id}/materials/{material_id}/generate-{flashcards,mock-test}`.

**Backend — modify:**
- `Backend/src/core/data_store.py` — add CRUD helpers for `class_subjects_collection`, `class_materials_collection`.
- `Backend/src/core/models.py` — add `class_id`/`class_subject_id` to `FlashcardDeck` and `MockTestResponse`.
- `Backend/src/routers/class_router.py` — set `org_id`/`teacher_ids` on create; add `POST /classes/{class_id}/teachers` (co-teacher).
- `Backend/src/routers/flashcard_router.py` — accept optional `class_id`/`class_subject_id` in `GenerateFlashcardsRequest` and store them on the deck.
- `Backend/src/routers/mock_test_router.py` — accept optional `class_id`/`class_subject_id` in `MockTestFromDocRequest` and store them on the test.
- `Backend/src/services/mock_test_service.py` — `generate_mock_test_from_docs_service` accepts and stores `class_id`/`class_subject_id`/`created_by`.
- `Backend/src/main.py` — include the two new routers.

**Frontend — create:**
- `Frontend/app/(dashboard)/classes/page.tsx` — teacher class list + create.
- `Frontend/app/(dashboard)/classes/[id]/page.tsx` — class detail tabs (Subjects, Materials, Tests, Students, Analytics). Phase 2 implements Subjects and Materials fully, plus simple Tests/Students/Analytics placeholders or existing data.

**Frontend — modify:**
- `Frontend/lib/api.ts` — extend `classAPI`; add `classSubjectAPI`, `classMaterialAPI`.
- `Frontend/components/dashboard/app-shell.tsx` — teacher nav: remove Focus/Plans, add Classes.
- `Frontend/app/(dashboard)/teacher/page.tsx` — replace embedded `TeacherClassesPanel` with a link to `/classes` (or keep panel as a summary + link).

**Tests — create:**
- `Backend/tests/test_class_subjects.py`
- `Backend/tests/test_class_materials.py`
- `Backend/tests/test_class_material_generation.py`

---

### Task 1: Make class creation org-aware + add co-teacher endpoint

**Files:**
- Modify: `Backend/src/routers/class_router.py:29-112,155-168,230-255`
- Test: `Backend/tests/test_class_router_p2.py`

**Interfaces:**
- Consumes: existing `Class` model (now has `org_id`, `teacher_ids`, `subject_ids` from P1), `users_collection`, `classes_collection`.
- Produces: `create_class` sets `org_id` and `teacher_ids` from the teacher's user doc; `POST /classes/{class_id}/teachers` adds a co-teacher (same org, `member_role="teacher"`). `get_class_detail` and `remove_student` authorize via `teacher_email in cls["teacher_ids"]` instead of only `teacher_id`. Later tasks rely on `teacher_ids` for ownership checks.

- [ ] **Step 1: Write the failing test**

Create `Backend/tests/test_class_router_p2.py`:

```python
import uuid
import pytest

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
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
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in (q or {}).items())]

    async def insert_one(self, doc):
        self._i += 1
        d = dict(doc)
        d["_id"] = str(self._i)
        self.docs[str(self._i)] = d
        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
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
                return


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); classes = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "classes_collection", classes)
    monkeypatch.setattr(ds, "mock_test_submissions_collection", _FakeColl())
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Backend && source venv/bin/activate && pytest tests/test_class_router_p2.py -v`
Expected: FAIL — `org_id`/`teacher_ids` not set on create; `/classes/{id}/teachers` 404.

- [ ] **Step 3: Update `create_class` to set `org_id` and `teacher_ids`**

In `Backend/src/routers/class_router.py`, modify `create_class` (around line 87-112). Add an import for `users_collection` if not already present (it is imported at line 17). Read the teacher's user doc to get `org_id`:

```python
@router.post("/", response_model=ClassSummary, status_code=status.HTTP_201_CREATED)
async def create_class(
    request: ClassCreateRequest,
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("class_count")),
):
    """Create a new class/batch and get a shareable enroll code."""
    teacher_email = teacher["email"]
    now = datetime.now(timezone.utc)
    enroll_code = _gen_enroll_code()

    teacher_user = None
    if users_collection is not None:
        teacher_user = await users_collection.find_one({"email": teacher_email})
    org_id = teacher_user.get("org_id") if teacher_user else None

    doc = {
        "teacher_id": teacher_email,
        "name": request.name,
        "description": request.description,
        "exam_preset": request.exam_preset,
        "enroll_code": enroll_code,
        "student_emails": [],
        "org_id": org_id,
        "teacher_ids": [teacher_email],
        "subject_ids": [],
        "created_at": now,
        "updated_at": now,
    }
    class_id = await store_class(doc)
    return ClassSummary(
        id=class_id, name=request.name, description=request.description,
        exam_preset=request.exam_preset, enroll_code=enroll_code,
        student_count=0, created_at=now,
    )
```

- [ ] **Step 4: Update `get_class_detail` and `remove_student` to use `teacher_ids`**

Change the authorization check in `get_class_detail` (line 161) from:
```python
    if cls.get("teacher_id") != teacher_email:
        raise HTTPException(status_code=403, detail="Not authorized to view this class")
```
to:
```python
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to view this class")
```

Do the same in `remove_student` (line 240):
```python
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to modify this class")
```

- [ ] **Step 5: Add the co-teacher endpoint**

Add a request model after `ClassCreateRequest`:

```python
class AddTeacherRequest(BaseModel):
    teacher_email: str
```

Add the endpoint after `create_class`:

```python
@router.post("/{class_id}/teachers", status_code=status.HTTP_200_OK)
async def add_teacher(
    request: AddTeacherRequest,
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    """Add a co-teacher to a class. Both teachers must belong to the same org."""
    teacher_email = teacher["email"]
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to modify this class")

    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    target = await users_collection.find_one({"email": request.teacher_email})
    if not target:
        raise HTTPException(status_code=404, detail="Teacher not found")
    if target.get("role") != "teacher" or target.get("member_role") != "teacher":
        raise HTTPException(status_code=400, detail="User is not a teacher in an organization")
    if target.get("org_id") != cls.get("org_id"):
        raise HTTPException(status_code=403, detail="Teacher must belong to the same organization")

    from src.core.data_store import classes_collection
    if classes_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$addToSet": {"teacher_ids": request.teacher_email}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
    return {"class_id": class_id, "teacher_email": request.teacher_email, "added": True}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_class_router_p2.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Run existing class tests for regressions**

Run: `pytest tests/test_class_router_p2.py tests/test_orgs.py tests/test_auth.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Backend/src/routers/class_router.py Backend/tests/test_class_router_p2.py
git commit -m "feat(class): org-aware creation, teacher_ids auth, add co-teacher endpoint"
```

---

### Task 2: ClassSubject CRUD backend

**Files:**
- Create: `Backend/src/services/class_subject_service.py`, `Backend/src/routers/class_subject_router.py`
- Modify: `Backend/src/core/data_store.py:563-613` (add helpers), `Backend/src/main.py:84` (include router)
- Test: `Backend/tests/test_class_subjects.py`

**Interfaces:**
- Consumes: `Class` (with `teacher_ids`, `org_id`), `class_subjects_collection`, `users_collection`.
- Produces: `POST /classes/{class_id}/subjects` (teacher in class) → `{id, class_id, name, icon, created_by}`; `GET /classes/{class_id}/subjects` → list; `DELETE /classes/{class_id}/subjects/{subject_id}` (creator or any class teacher). Updates `Class.subject_ids` on create/delete.

- [ ] **Step 1: Add data_store helpers for class_subjects**

In `Backend/src/core/data_store.py`, after the existing class helpers section (after `add_student_to_class`, before `get_user_by_email`), add:

```python
# --- Class subject helpers --------------------------------------------------
async def store_class_subject(subject_data: Dict[str, Any]) -> str:
    if class_subjects_collection is None:
        raise Exception("Database connection not available")
    result = await class_subjects_collection.insert_one(subject_data)
    return str(result.inserted_id)


async def get_class_subject_by_id(subject_id: str) -> Optional[Dict[str, Any]]:
    if class_subjects_collection is None:
        raise Exception("Database connection not available")
    sub = await class_subjects_collection.find_one({"_id": ObjectId(subject_id)})
    return object_id_to_str(sub) if sub else None


async def list_class_subjects(class_id: str) -> List[Dict[str, Any]]:
    if class_subjects_collection is None:
        raise Exception("Database connection not available")
    cursor = class_subjects_collection.find({"class_id": class_id}).sort("created_at", 1)
    subs = await cursor.to_list(length=None)
    return [object_id_to_str(s) for s in subs]


async def delete_class_subject(subject_id: str):
    if class_subjects_collection is None:
        raise Exception("Database connection not available")
    await class_subjects_collection.delete_one({"_id": ObjectId(subject_id)})
```

- [ ] **Step 2: Create the service**

Create `Backend/src/services/class_subject_service.py`:

```python
"""ClassSubject CRUD + ownership checks."""
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException

from src.core.data_store import (
    classes_collection,
    class_subjects_collection,
    get_class_by_id,
    list_class_subjects,
    store_class_subject,
    get_class_subject_by_id,
    delete_class_subject,
)


async def _require_class_teacher(class_id: str, teacher_email: str):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to manage this class")
    return cls


async def create_class_subject(class_id: str, name: str, icon: Optional[str], teacher_email: str) -> dict:
    if classes_collection is None or class_subjects_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    cls = await _require_class_teacher(class_id, teacher_email)
    now = datetime.now(timezone.utc)
    subject_doc = {
        "class_id": class_id,
        "name": name,
        "icon": icon,
        "created_by": teacher_email,
        "created_at": now,
        "updated_at": now,
    }
    subject_id = await store_class_subject(subject_doc)

    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$addToSet": {"subject_ids": subject_id}, "$set": {"updated_at": now}},
    )

    return {"id": subject_id, **subject_doc}


async def list_subjects(class_id: str, teacher_email: str) -> List[dict]:
    await _require_class_teacher(class_id, teacher_email)
    return await list_class_subjects(class_id)


async def remove_class_subject(class_id: str, subject_id: str, teacher_email: str) -> dict:
    if classes_collection is None or class_subjects_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    cls = await _require_class_teacher(class_id, teacher_email)
    subject = await get_class_subject_by_id(subject_id)
    if not subject or subject.get("class_id") != class_id:
        raise HTTPException(status_code=404, detail="Subject not found in this class")

    await delete_class_subject(subject_id)
    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$pull": {"subject_ids": subject_id}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
    return {"subject_id": subject_id, "deleted": True}
```

- [ ] **Step 3: Create the router**

Create `Backend/src/routers/class_subject_router.py`:

```python
"""ClassSubject endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from src.core.security import require_role
from src.services import class_subject_service as svc

router = APIRouter(prefix="/classes/{class_id}/subjects", tags=["Class Subjects"])


class SubjectCreateRequest(BaseModel):
    name: str
    icon: Optional[str] = None


class SubjectResponse(BaseModel):
    id: str
    class_id: str
    name: str
    icon: Optional[str] = None
    created_by: str
    created_at: datetime


class SubjectListResponse(BaseModel):
    subjects: List[SubjectResponse]


@router.post("/", response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
async def create_subject(
    request: SubjectCreateRequest,
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    result = await svc.create_class_subject(class_id, request.name, request.icon, teacher["email"])
    return SubjectResponse(**result)


@router.get("/", response_model=SubjectListResponse)
async def list_subjects(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    subs = await svc.list_subjects(class_id, teacher["email"])
    return SubjectListResponse(subjects=[SubjectResponse(**s) for s in subs])


@router.delete("/{subject_id}", status_code=status.HTTP_200_OK)
async def delete_subject(
    subject_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    return await svc.remove_class_subject(class_id, subject_id, teacher["email"])
```

Add the missing imports at the top of the file:
```python
from datetime import datetime
from typing import List
```

- [ ] **Step 4: Wire the router into `main.py`**

In `Backend/src/main.py`, add the import near the other router imports (line 31):
```python
from src.routers import (
    ...
    class_subject_router,
)
```

And add `app.include_router(class_subject_router)` after `app.include_router(class_router)` at line 84.

- [ ] **Step 5: Write the failing test**

Create `Backend/tests/test_class_subjects.py`:

```python
import uuid
import pytest

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
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
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in (q or {}).items())]

    async def insert_one(self, doc):
        self._i += 1
        d = dict(doc)
        d["_id"] = str(self._i)
        self.docs[str(self._i)] = d
        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
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

    async def delete_one(self, q):
        for k, d in list(self.docs.items()):
            if all(d.get(kk) == v for kk, v in q.items()):
                del self.docs[k]
                return


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); classes = _FakeColl(); subjects = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "classes_collection", classes)
    monkeypatch.setattr(ds, "class_subjects_collection", subjects)
    users.docs["1"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")
    c = TestClient(app)
    # seed a class with teacher_ids
    r = c.post("/classes/", json={"name": "JEE"})
    class_id = r.json()["id"]
    yield dict(users=users, classes=classes, subjects=subjects, class_id=class_id, client=c)
    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_create_subject(setup):
    r = setup["client"].post(f"/classes/{setup['class_id']}/subjects", json={"name": "Physics", "icon": "atom"})
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["name"] == "Physics"
    assert data["class_id"] == setup["class_id"]
    assert "Physics" in setup["classes"].docs["1"].get("subject_ids", [])


def test_list_subjects(setup):
    c = setup["client"]
    cid = setup["class_id"]
    c.post(f"/classes/{cid}/subjects", json={"name": "Physics"})
    c.post(f"/classes/{cid}/subjects", json={"name": "Chemistry"})
    r = c.get(f"/classes/{cid}/subjects")
    assert r.status_code == 200
    assert len(r.json()["subjects"]) == 2


def test_delete_subject(setup):
    c = setup["client"]; cid = setup["class_id"]
    sub = c.post(f"/classes/{cid}/subjects", json={"name": "Physics"}).json()
    r = c.delete(f"/classes/{cid}/subjects/{sub['id']}")
    assert r.status_code == 200
    assert sub["id"] not in setup["classes"].docs["1"].get("subject_ids", [])


def test_outsider_cannot_create_subject(setup):
    setup["users"].docs["2"] = {"email": "bad@x.com", "role": "teacher", "org_id": "org-other", "member_role": "teacher"}
    _set_auth("teacher", "bad@x.com")
    r = setup["client"].post(f"/classes/{setup['class_id']}/subjects", json={"name": "X"})
    assert r.status_code == 403
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_class_subjects.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add Backend/src/services/class_subject_service.py Backend/src/routers/class_subject_router.py Backend/src/core/data_store.py Backend/src/main.py Backend/tests/test_class_subjects.py
git commit -m "feat(class-subjects): CRUD backend for class subjects"
```

---

### Task 3: ClassMaterial upload + RAG backend

**Files:**
- Create: `Backend/src/services/class_material_service.py`, `Backend/src/routers/class_material_router.py`
- Modify: `Backend/src/core/data_store.py` (add helpers), `Backend/src/main.py` (include router)
- Test: `Backend/tests/test_class_materials.py`

**Interfaces:**
- Consumes: `Class`, `ClassSubject`, `pdfs_collection`, `document_chunks_collection`, ChromaDB `VectorStore`, existing `store_pdf_metadata`, `update_pdf_metadata`, `store_document_chunks`.
- Produces: `POST /classes/{class_id}/subjects/{subject_id}/materials` (multipart, teacher in class) → `ClassMaterialResponse`; `GET /classes/{class_id}/subjects/{subject_id}/materials` → list; `DELETE /classes/{class_id}/materials/{material_id}` → deleted. Material record has `doc_id` + `rag_indexed=True` when indexing succeeds.

- [ ] **Step 1: Add data_store helpers for class_materials**

In `Backend/src/core/data_store.py`, after the class_subjects helpers added in Task 2, add:

```python
# --- Class material helpers -------------------------------------------------
async def store_class_material(material_data: Dict[str, Any]) -> str:
    if class_materials_collection is None:
        raise Exception("Database connection not available")
    result = await class_materials_collection.insert_one(material_data)
    return str(result.inserted_id)


async def get_class_material_by_id(material_id: str) -> Optional[Dict[str, Any]]:
    if class_materials_collection is None:
        raise Exception("Database connection not available")
    mat = await class_materials_collection.find_one({"_id": ObjectId(material_id)})
    return object_id_to_str(mat) if mat else None


async def list_class_materials(class_id: str, class_subject_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if class_materials_collection is None:
        raise Exception("Database connection not available")
    q: Dict[str, Any] = {"class_id": class_id}
    if class_subject_id:
        q["class_subject_id"] = class_subject_id
    cursor = class_materials_collection.find(q).sort("created_at", -1)
    mats = await cursor.to_list(length=None)
    return [object_id_to_str(m) for m in mats]


async def delete_class_material(material_id: str):
    if class_materials_collection is None:
        raise Exception("Database connection not available")
    await class_materials_collection.delete_one({"_id": ObjectId(material_id)})
```

- [ ] **Step 2: Create the service**

Create `Backend/src/services/class_material_service.py`. This is adapted from `material_router.py:123-264` but scoped to a class subject.

```python
"""Class-scoped material upload + RAG indexing."""
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, UploadFile

from src.core.data_store import (
    classes_collection,
    class_subjects_collection,
    class_materials_collection,
    get_class_by_id,
    get_class_subject_by_id,
    get_class_material_by_id,
    list_class_materials,
    store_class_material,
    delete_class_material,
    store_pdf_metadata,
    update_pdf_metadata,
    store_document_chunks,
)
from src.services.vector_store import VectorStore
from src.services.document_processor import detect_doc_type, extract_text_from_pdf, chunk_document

vector_store = VectorStore()


async def _require_class_teacher(class_id: str, teacher_email: str) -> dict:
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to manage this class")
    return cls


async def upload_class_material(
    class_id: str,
    class_subject_id: str,
    file: UploadFile,
    teacher_email: str,
    _plan: dict,  # passed from router dependency for limit enforcement
) -> dict:
    if any(c is None for c in (classes_collection, class_subjects_collection, class_materials_collection)):
        raise HTTPException(status_code=503, detail="Database connection not available")

    cls = await _require_class_teacher(class_id, teacher_email)
    subject = await get_class_subject_by_id(class_subject_id)
    if not subject or subject.get("class_id") != class_id:
        raise HTTPException(status_code=404, detail="Subject not found in this class")

    subject_name = subject.get("name")
    tags = [cls.get("name", ""), subject_name] if cls.get("name") else [subject_name]

    doc_type = detect_doc_type(file.filename)
    if doc_type == "unknown":
        doc_type = "text"

    content = await file.read()
    size = len(content)

    user_dir = os.path.join("uploads", teacher_email)
    os.makedirs(user_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(user_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(content)

    file_type = "pdf" if file.filename.lower().endswith(".pdf") else "text"

    now = datetime.now(timezone.utc)
    material_doc = {
        "class_id": class_id,
        "class_subject_id": class_subject_id,
        "teacher_id": teacher_email,
        "name": file.filename,
        "type": file_type,
        "size": size,
        "doc_id": None,
        "rag_indexed": False,
        "page_count": None,
        "created_at": now,
        "updated_at": now,
    }
    material_id = await store_class_material(material_doc)

    doc_id = None
    chunk_count = 0
    rag_indexed = False
    page_count = None

    try:
        pdf_meta = await store_pdf_metadata(
            filename=file.filename,
            size=size,
            user_id=teacher_email,
            file_path=file_path,
            title=file.filename,
            tags=tags,
        )
        doc_id = pdf_meta["id"]

        if doc_type == "pdf":
            text, page_count = extract_text_from_pdf(content)
        else:
            text = content.decode("utf-8", errors="ignore")
        chunks_data = chunk_document(text, doc_type=doc_type)

        model = VectorStore.get_embedding_model()
        chroma_chunks = []
        for chunk in chunks_data:
            chroma_id = str(uuid.uuid4())
            embedding = model.encode(chunk["content"]).tolist()
            chroma_chunks.append({
                "chroma_id": chroma_id,
                "user_id": teacher_email,
                "doc_id": doc_id,
                "doc_name": file.filename,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "embedding": embedding,
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "doc_type": doc_type,
                "subject": subject_name,
                "tags": tags,
                "material_id": material_id,
            })

        if chroma_chunks:
            vector_store.add_chunks(teacher_email, chroma_chunks)
            await store_document_chunks(chroma_chunks)
            chunk_count = len(chroma_chunks)
            rag_indexed = True

        await update_pdf_metadata(doc_id, {
            "processed": True,
            "chunk_count": chunk_count,
            "doc_type": doc_type,
            "subject": subject_name,
            "tags": tags,
            "page_count": page_count,
            "material_id": material_id,
        })
    except Exception as index_err:
        print(f"Class material RAG indexing failed for {material_id}: {index_err}")

    await class_materials_collection.update_one(
        {"_id": __import__("bson").ObjectId(material_id)},
        {"$set": {
            "doc_id": doc_id,
            "rag_indexed": rag_indexed,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "updated_at": datetime.now(timezone.utc),
        }}
    )

    created = await get_class_material_by_id(material_id)
    return created


async def list_materials(class_id: str, class_subject_id: Optional[str], teacher_email: str) -> List[dict]:
    await _require_class_teacher(class_id, teacher_email)
    return await list_class_materials(class_id, class_subject_id)


async def remove_class_material(class_id: str, material_id: str, teacher_email: str) -> dict:
    if class_materials_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    await _require_class_teacher(class_id, teacher_email)
    mat = await get_class_material_by_id(material_id)
    if not mat or mat.get("class_id") != class_id:
        raise HTTPException(status_code=404, detail="Material not found in this class")
    await delete_class_material(material_id)
    return {"material_id": material_id, "deleted": True}


async def get_class_material(class_id: str, material_id: str) -> Optional[dict]:
    mat = await get_class_material_by_id(material_id)
    if not mat or mat.get("class_id") != class_id:
        return None
    return mat
```

- [ ] **Step 3: Create the router**

Create `Backend/src/routers/class_material_router.py`:

```python
"""ClassMaterial endpoints."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from pydantic import BaseModel

from src.core.security import require_role
from src.core.plan_enforcement import enforce_limit
from src.services import class_material_service as svc

router = APIRouter(prefix="/classes/{class_id}", tags=["Class Materials"])


class ClassMaterialResponse(BaseModel):
    id: str
    class_id: str
    class_subject_id: str
    teacher_id: str
    name: str
    type: str
    size: int
    doc_id: Optional[str] = None
    rag_indexed: bool = False
    page_count: Optional[int] = None
    created_at: datetime


class ClassMaterialListResponse(BaseModel):
    materials: List[ClassMaterialResponse]


@router.post("/subjects/{subject_id}/materials", response_model=ClassMaterialResponse, status_code=status.HTTP_201_CREATED)
async def upload_material(
    file: UploadFile = File(...),
    class_id: str = Path(...),
    subject_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("doc_storage")),
):
    result = await svc.upload_class_material(class_id, subject_id, file, teacher["email"], _plan)
    return ClassMaterialResponse(**result)


@router.get("/subjects/{subject_id}/materials", response_model=ClassMaterialListResponse)
async def list_materials(
    class_id: str = Path(...),
    subject_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    mats = await svc.list_materials(class_id, subject_id, teacher["email"])
    return ClassMaterialListResponse(materials=[ClassMaterialResponse(**m) for m in mats])


@router.delete("/materials/{material_id}", status_code=status.HTTP_200_OK)
async def delete_material(
    material_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    return await svc.remove_class_material(class_id, material_id, teacher["email"])
```

- [ ] **Step 4: Wire the router into `main.py`**

In `Backend/src/main.py`, add `class_material_router` to the import list and `app.include_router(class_material_router)` after the class_subject_router include.

- [ ] **Step 5: Write the failing test**

Create `Backend/tests/test_class_materials.py`:

```python
import io
import uuid
import pytest

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
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
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in (q or {}).items())]

    async def insert_one(self, doc):
        self._i += 1
        d = dict(doc)
        d["_id"] = str(self._i)
        self.docs[str(self._i)] = d
        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return

    async def delete_one(self, q):
        for k, d in list(self.docs.items()):
            if all(d.get(kk) == v for kk, v in q.items()):
                del self.docs[k]
                return


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch, tmp_path):
    users = _FakeColl(); classes = _FakeColl(); subjects = _FakeColl(); materials = _FakeColl(); pdfs = _FakeColl(); chunks = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "classes_collection", classes)
    monkeypatch.setattr(ds, "class_subjects_collection", subjects)
    monkeypatch.setattr(ds, "class_materials_collection", materials)
    monkeypatch.setattr(ds, "pdfs_collection", pdfs)
    monkeypatch.setattr(ds, "document_chunks_collection", chunks)
    users.docs["1"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")

    c = TestClient(app)
    cid = c.post("/classes/", json={"name": "JEE"}).json()["id"]
    sid = c.post(f"/classes/{cid}/subjects", json={"name": "Physics"}).json()["id"]

    # Stub out real ChromaDB + embedding to avoid loading models in tests
    from src.services import class_material_service as cms
    monkeypatch.setattr(cms.vector_store, "add_chunks", lambda user_id, chunks: None)
    from src.services.vector_store import VectorStore
    monkeypatch.setattr(VectorStore, "get_embedding_model", lambda self=None: _FakeEncoder())

    yield dict(users=users, classes=classes, materials=materials, pdfs=pdfs, chunks=chunks, client=c, class_id=cid, subject_id=sid)
    app.dependency_overrides.pop(get_current_user_with_role, None)


class _FakeEncoder:
    def encode(self, text):
        return [0.0] * 384


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


def test_list_and_delete_class_material(setup):
    c = setup["client"]; cid = setup["class_id"]; sid = setup["subject_id"]
    up = c.post(f"/classes/{cid}/subjects/{sid}/materials", files={"file": ("a.pdf", io.BytesIO(b"x"), "application/pdf")}).json()
    lst = c.get(f"/classes/{cid}/subjects/{sid}/materials")
    assert lst.status_code == 200
    assert len(lst.json()["materials"]) == 1
    r = c.delete(f"/classes/{cid}/materials/{up['id']}")
    assert r.status_code == 200
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_class_materials.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add Backend/src/services/class_material_service.py Backend/src/routers/class_material_router.py Backend/src/core/data_store.py Backend/src/main.py Backend/tests/test_class_materials.py
git commit -m "feat(class-materials): class-scoped material upload + RAG backend"
```

---

### Task 4: Generate flashcards/mock tests from a class material

**Files:**
- Modify: `Backend/src/core/models.py:174-189,414-424`, `Backend/src/services/mock_test_service.py:137-215`, `Backend/src/routers/flashcard_router.py:122-169`, `Backend/src/routers/mock_test_router.py:127-165`, `Backend/src/routers/class_material_router.py`
- Test: `Backend/tests/test_class_material_generation.py`

**Interfaces:**
- Consumes: `ClassMaterial` with `doc_id`, existing `generate_flashcards`, `generate_mock_test_from_docs_service`, flashcard/mock storage helpers.
- Produces: `POST /classes/{class_id}/materials/{material_id}/generate-flashcards` → `{deck_id, card_count}`; `POST /classes/{class_id}/materials/{material_id}/generate-mock-test` → `MockTestResponse`. Both store `class_id` + `class_subject_id` + `created_by` so the outputs are class-scoped. `FlashcardDeck` and `MockTestResponse` models gain optional `class_id`/`class_subject_id` fields.

- [ ] **Step 1: Extend the data models**

In `Backend/src/core/models.py`:

Add to `MockTestResponse` (after `created_by`, around line 183):
```python
    class_id: Optional[str] = None
    class_subject_id: Optional[str] = None
```

Add to `FlashcardDeck` (after `created_by`, around line 421):
```python
    class_id: Optional[str] = None
    class_subject_id: Optional[str] = None
```

- [ ] **Step 2: Extend flashcard generation to store class scope**

In `Backend/src/routers/flashcard_router.py`, add to `GenerateFlashcardsRequest` (around line 436 in models.py — actually this model is in models.py, not flashcard_router.py; modify it there):

In `Backend/src/core/models.py` `GenerateFlashcardsRequest`:
```python
    class_id: Optional[str] = None
    class_subject_id: Optional[str] = None
```

In `Backend/src/routers/flashcard_router.py`, modify `generate_deck` deck_doc (around line 142-153) to include class scope:
```python
    deck_doc = {
        "id": deck_id,
        "user_id": user_id,
        "title": title,
        "subject": req.subject,
        "source_material_ids": req.material_ids,
        "source_type": "ai",
        "created_by": None,
        "card_count": len(cards_data),
        "class_id": req.class_id,
        "class_subject_id": req.class_subject_id,
        "created_at": now,
        "updated_at": now,
    }
```

- [ ] **Step 3: Extend mock-test generation to store class scope**

In `Backend/src/services/mock_test_service.py`, modify `generate_mock_test_from_docs_service` signature and storage:

Change signature (line 137) to add:
```python
    class_id: Optional[str] = None,
    class_subject_id: Optional[str] = None,
    created_by: Optional[str] = None,
```

When constructing `MockTestResponse` (around line 192-203), add:
```python
        mock_test = MockTestResponse(
            ...
            created_by=created_by or user_id,
            class_id=class_id,
            class_subject_id=class_subject_id,
            ...
        )
```

In `Backend/src/routers/mock_test_router.py`, modify `MockTestFromDocRequest` (in models.py) to add:
```python
    class_id: Optional[str] = None
    class_subject_id: Optional[str] = None
```

And in `generate_mock_test_from_doc` (around line 148), pass them:
```python
        mock_test = await generate_mock_test_from_docs_service(
            doc_ids=req.doc_ids,
            num_mcq=req.num_mcq,
            num_text=req.num_text,
            total_marks=req.total_marks,
            difficulty_level=req.difficulty_level,
            user_id=user_id,
            subject=req.subject,
            class_id=req.class_id,
            class_subject_id=req.class_subject_id,
            created_by=user_id,
        )
```

- [ ] **Step 4: Add the class-material generation endpoints**

In `Backend/src/routers/class_material_router.py`, add imports at the top:
```python
import uuid

from src.services.gemini_service import gemini_service, generate_flashcards
from src.services.mock_test_service import generate_mock_test_from_docs_service
from src.core.models import MockTestResponse
from src.core.data_store import (
    store_flashcard_deck,
    store_flashcards,
)
```

Add response models:
```python
class GenerateFlashcardsFromMaterialResponse(BaseModel):
    deck_id: str
    card_count: int


class GenerateMockTestFromMaterialResponse(MockTestResponse):
    pass
```

Add a helper to read doc content (reuses flashcard `_gather_content` pattern):
```python
async def _get_doc_content(doc_id: str, teacher_email: str) -> str:
    from src.core.data_store import get_pdf_metadata
    pdf = await get_pdf_metadata(doc_id)
    if not pdf or pdf.get("user_id") != teacher_email:
        raise HTTPException(status_code=404, detail="Source document not found")
    file_path = pdf.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="Source document has no file")
    try:
        if file_path.lower().endswith(".pdf"):
            text = await gemini_service.extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read source document: {e}")
    return text[:8000]
```

Add the two endpoints:
```python
@router.post("/materials/{material_id}/generate-flashcards", response_model=GenerateFlashcardsFromMaterialResponse, status_code=status.HTTP_201_CREATED)
async def generate_flashcards_from_material(
    material_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("flashcard")),
):
    mat = await svc.get_class_material(class_id, material_id)
    if not mat or not mat.get("doc_id"):
        raise HTTPException(status_code=400, detail="Material is not indexed for AI generation")

    content = await _get_doc_content(mat["doc_id"], teacher["email"])
    cards_data = await generate_flashcards(content, num_cards=15, subject=mat.get("name"))
    if not cards_data:
        raise HTTPException(status_code=502, detail="Failed to generate flashcards")

    deck_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    deck_doc = {
        "id": deck_id,
        "user_id": teacher["email"],
        "title": f"Flashcards — {mat['name']}",
        "subject": mat.get("name"),
        "source_material_ids": [material_id],
        "source_type": "ai",
        "created_by": teacher["email"],
        "card_count": len(cards_data),
        "class_id": class_id,
        "class_subject_id": mat["class_subject_id"],
        "created_at": now,
        "updated_at": now,
    }
    await store_flashcard_deck(deck_doc)
    card_docs = [{
        "id": str(uuid.uuid4()),
        "deck_id": deck_id,
        "front": c["front"],
        "back": c["back"],
        "ease": 2,
        "interval_days": 0,
        "reps": 0,
        "due_at": now,
        "created_at": now,
    } for c in cards_data]
    await store_flashcards(card_docs)
    return GenerateFlashcardsFromMaterialResponse(deck_id=deck_id, card_count=len(card_docs))


@router.post("/materials/{material_id}/generate-mock-test", response_model=GenerateMockTestFromMaterialResponse, status_code=status.HTTP_201_CREATED)
async def generate_mock_test_from_material(
    material_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("mock_test")),
):
    mat = await svc.get_class_material(class_id, material_id)
    if not mat or not mat.get("doc_id"):
        raise HTTPException(status_code=400, detail="Material is not indexed for AI generation")

    mock_test = await generate_mock_test_from_docs_service(
        doc_ids=[mat["doc_id"]],
        num_mcq=10,
        num_text=3,
        total_marks=30,
        difficulty_level="medium",
        user_id=teacher["email"],
        subject=mat.get("name"),
        class_id=class_id,
        class_subject_id=mat["class_subject_id"],
        created_by=teacher["email"],
    )
    return mock_test
```

Note: `svc.get_class_material` does not exist yet — add it to the service:
```python
async def get_class_material(class_id: str, material_id: str) -> Optional[dict]:
    mat = await get_class_material_by_id(material_id)
    if not mat or mat.get("class_id") != class_id:
        return None
    return mat
```

Also add `uuid` import to the router file.

- [ ] **Step 5: Write the failing test**

Create `Backend/tests/test_class_material_generation.py`. This tests the schema + endpoint wiring with mocked generation.

```python
import io
import uuid
import pytest

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
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
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in (q or {}).items())]

    async def insert_one(self, doc):
        self._i += 1
        d = dict(doc)
        d["_id"] = str(self._i)
        self.docs[str(self._i)] = d
        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return

    async def delete_one(self, q):
        for k, d in list(self.docs.items()):
            if all(d.get(kk) == v for kk, v in q.items()):
                del self.docs[k]
                return


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); classes = _FakeColl(); subjects = _FakeColl(); materials = _FakeColl(); pdfs = _FakeColl(); decks = _FakeColl(); cards = _FakeColl(); tests = _FakeColl(); chunks = _FakeColl()
    for n, c in [("users_collection", users), ("classes_collection", classes), ("class_subjects_collection", subjects),
                 ("class_materials_collection", materials), ("pdfs_collection", pdfs), ("flashcard_decks_collection", decks),
                 ("flashcards_collection", cards), ("mock_tests_collection", tests), ("document_chunks_collection", chunks)]:
        monkeypatch.setattr(ds, n, c)
    users.docs["1"] = {"email": "t1@x.com", "role": "teacher", "org_id": "org-9", "member_role": "teacher"}
    _set_auth("teacher", "t1@x.com")

    from src.services import class_material_service as cms
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
    monkeypatch.setattr(gs, "generate_flashcards", lambda content, num_cards, subject=None: [{"front": "Q?", "back": "A."}] * 2)
    from src.services import mock_test_service as mts
    async def fake_generate(doc_ids, num_mcq, num_text, total_marks, difficulty_level, user_id, subject=None, class_id=None, class_subject_id=None, created_by=None, grading_mode="auto"):
        from src.core.models import MockTestResponse, MockTestQuestion
        return MockTestResponse(
            test_id="test-1", title="T", total_marks=30, time_limit=30, user_id=user_id,
            questions=[MockTestQuestion(id="1", type="mcq", question="Q", options=["A) a"], correctAnswer="A) a", marks=2)],
            created_by=created_by, class_id=class_id, class_subject_id=class_subject_id,
        )
    monkeypatch.setattr(mts, "generate_mock_test_from_docs_service", fake_generate)
    monkeypatch.setattr(mts, "store_mock_test", lambda data: None)

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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_class_material_generation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Run focused backend suites together**

Run: `pytest tests/test_class_router_p2.py tests/test_class_subjects.py tests/test_class_materials.py tests/test_class_material_generation.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Backend/src/core/models.py Backend/src/services/mock_test_service.py Backend/src/routers/flashcard_router.py Backend/src/routers/mock_test_router.py Backend/src/routers/class_material_router.py Backend/src/services/class_material_service.py Backend/tests/test_class_material_generation.py
git commit -m "feat(class-materials): generate flashcards + mock tests from class material"
```

---

### Task 5: Frontend API helpers

**Files:**
- Modify: `Frontend/lib/api.ts:653-678` (classAPI) + new helpers after line 678.

**Interfaces:**
- Consumes: the new backend endpoints from Tasks 1–4.
- Produces: `classAPI` extended with `addTeacher`, `getClassDetail`; new `classSubjectAPI` (create/list/delete); new `classMaterialAPI` (upload/list/delete, generate flashcards/mock test).

- [ ] **Step 1: Extend `classAPI`**

In `Frontend/lib/api.ts`, after `removeStudent` inside `classAPI` (line 677), add:

```ts
  async addTeacher(classId: string, teacherEmail: string): Promise<any> {
    const res = await api.post(`/classes/${classId}/teachers`, { teacher_email: teacherEmail });
    return res.data;
  },
```

- [ ] **Step 2: Add `classSubjectAPI`**

After the closing `};` of `classAPI` (around line 679), add:

```ts
// ─── Class subjects ──────────────────────────────────────────────────────────
export const classSubjectAPI = {
  async create(classId: string, req: { name: string; icon?: string }): Promise<{ id: string; class_id: string; name: string; icon?: string; created_by: string }> {
    const res = await api.post(`/classes/${classId}/subjects`, req);
    return res.data;
  },
  async list(classId: string): Promise<{ subjects: any[] }> {
    const res = await api.get(`/classes/${classId}/subjects`);
    return res.data;
  },
  async delete(classId: string, subjectId: string): Promise<any> {
    const res = await api.delete(`/classes/${classId}/subjects/${subjectId}`);
    return res.data;
  },
};
```

- [ ] **Step 3: Add `classMaterialAPI`**

After `classSubjectAPI`, add:

```ts
// ─── Class materials ─────────────────────────────────────────────────────────
export const classMaterialAPI = {
  async upload(classId: string, subjectId: string, file: File): Promise<any> {
    const form = new FormData();
    form.append("file", file);
    const res = await api.post(`/classes/${classId}/subjects/${subjectId}/materials`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
  async list(classId: string, subjectId: string): Promise<{ materials: any[] }> {
    const res = await api.get(`/classes/${classId}/subjects/${subjectId}/materials`);
    return res.data;
  },
  async delete(classId: string, materialId: string): Promise<any> {
    const res = await api.delete(`/classes/${classId}/materials/${materialId}`);
    return res.data;
  },
  async generateFlashcards(classId: string, materialId: string): Promise<{ deck_id: string; card_count: number }> {
    const res = await api.post(`/classes/${classId}/materials/${materialId}/generate-flashcards`);
    return res.data;
  },
  async generateMockTest(classId: string, materialId: string): Promise<any> {
    const res = await api.post(`/classes/${classId}/materials/${materialId}/generate-mock-test`);
    return res.data;
  },
};
```

- [ ] **Step 4: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add Frontend/lib/api.ts
git commit -m "feat(frontend): classSubjectAPI and classMaterialAPI helpers"
```

---

### Task 6: Frontend `/classes` page (teacher)

**Files:**
- Create: `Frontend/app/(dashboard)/classes/page.tsx`
- Modify: `Frontend/components/dashboard/teacher/teacher-classes-panel.tsx` (add "Open class" link), `Frontend/app/(dashboard)/teacher/page.tsx` (replace panel with link or keep summary).

**Interfaces:**
- Consumes: `classAPI`.
- Produces: A dedicated `/classes` page listing all classes for the teacher with create-class dialog + enroll code + "Open" link to `/classes/{id}`. The teacher dashboard panel becomes a summary + link.

- [ ] **Step 1: Create `/classes` page**

Create `Frontend/app/(dashboard)/classes/page.tsx`:

```tsx
"use client"

import { useCallback, useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import { Copy, Loader2, Plus, Users } from "lucide-react"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogTrigger,
} from "@/components/ui/dialog"
import { RoleGuard } from "@/components/auth/route-protection/role-guard"

interface ClassItem {
  id: string
  name: string
  description?: string
  exam_preset?: string
  enroll_code: string
  student_count: number
}

export default function ClassesPage() {
  const { toast } = useToast()
  const [classes, setClasses] = useState<ClassItem[]>([])
  const [loading, setLoading] = useState(true)
  const [createOpen, setCreateOpen] = useState(false)
  const [newName, setNewName] = useState("")
  const [newDesc, setNewDesc] = useState("")
  const [newPreset, setNewPreset] = useState("")
  const [isCreating, setIsCreating] = useState(false)

  const fetchClasses = useCallback(async () => {
    setLoading(true)
    try {
      const list = await classAPI.listClasses()
      setClasses((list || []) as ClassItem[])
    } catch (e) {
      toast({ title: "Couldn't load classes", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { fetchClasses() }, [fetchClasses])

  const handleCreate = async () => {
    if (!newName.trim()) return
    setIsCreating(true)
    try {
      await classAPI.createClass({ name: newName.trim(), description: newDesc.trim() || undefined, exam_preset: newPreset.trim() || undefined })
      setCreateOpen(false); setNewName(""); setNewDesc(""); setNewPreset("")
      fetchClasses()
      toast({ title: "Class created", description: "Share the enroll code with students." })
    } catch (e) {
      toast({ title: "Couldn't create class", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setIsCreating(false)
    }
  }

  const copyCode = (code: string) => {
    navigator.clipboard?.writeText(code)
    toast({ title: "Enroll code copied", description: code })
  }

  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Classes</h1>
            <p className="text-sm text-muted-foreground">Manage your batches, subjects, materials, and tests.</p>
          </div>
          <Dialog open={createOpen} onOpenChange={setCreateOpen}>
            <DialogTrigger asChild>
              <Button><Plus className="h-4 w-4 mr-2" />New Class</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>New Class</DialogTitle></DialogHeader>
              <div className="space-y-3">
                <div className="space-y-1.5"><Label htmlFor="cn">Name</Label><Input id="cn" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. JEE 2026 Batch" /></div>
                <div className="space-y-1.5"><Label htmlFor="cd">Description</Label><Input id="cd" value={newDesc} onChange={(e) => setNewDesc(e.target.value)} placeholder="Optional" /></div>
                <div className="space-y-1.5"><Label htmlFor="cp">Exam preset</Label><Input id="cp" value={newPreset} onChange={(e) => setNewPreset(e.target.value)} placeholder="e.g. jee-mains" /></div>
              </div>
              <DialogFooter>
                <Button disabled={!newName.trim() || isCreating} onClick={handleCreate}>{isCreating ? <Loader2 className="h-4 w-4 animate-spin" /> : "Create"}</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>

        {loading ? (
          <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
        ) : classes.length === 0 ? (
          <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No classes yet. Create one to group your students.</div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {classes.map((c) => (
              <div key={c.id} className="rounded-lg border bg-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">{c.name}</h3>
                  <span className="text-xs text-muted-foreground flex items-center gap-1"><Users className="h-3 w-3" />{c.student_count}</span>
                </div>
                {c.description && <p className="text-xs text-muted-foreground">{c.description}</p>}
                <div className="flex items-center gap-2">
                  <code className="rounded bg-secondary px-2 py-1 text-xs tracking-wider font-mono">{c.enroll_code}</code>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => copyCode(c.enroll_code)}><Copy className="h-3.5 w-3.5" /></Button>
                </div>
                <Button asChild variant="outline" className="w-full">
                  <Link href={`/classes/${c.id}`}>Open class</Link>
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </RoleGuard>
  )
}
```

- [ ] **Step 2: Update `TeacherClassesPanel` to link to `/classes`**

In `Frontend/components/dashboard/teacher/teacher-classes-panel.tsx`, add `Link` import at the top:
```tsx
import Link from "next/link"
```

Replace the roster button on the class card (around line 149) with:
```tsx
                <Button variant="outline" size="sm" className="ml-auto h-7 text-[12px]" asChild>
                  <Link href={`/classes/${c.id}`}>Open</Link>
                </Button>
```

Keep the roster dialog for now as a quick view, or remove it to keep the panel simple. The plan recommends keeping it as a summary + link; leave the roster dialog intact.

- [ ] **Step 3: Update teacher dashboard to link to `/classes`**

In `Frontend/app/(dashboard)/teacher/page.tsx`, after the `TeacherClassesPanel` (around line 455), add a prominent "Manage classes →" link, or replace the panel header with a link. Minimal change: add a link below the panel.

Find where `TeacherClassesPanel` is rendered (lines 455-462) and wrap or add:
```tsx
<div className="flex items-center justify-between mt-4">
  <p className="text-sm text-muted-foreground">View all classes, subjects, materials, and tests.</p>
  <Button asChild variant="outline" size="sm"><Link href="/classes">Manage classes →</Link></Button>
</div>
```

Add `Link` import if not present.

- [ ] **Step 4: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add Frontend/app/(dashboard)/classes/page.tsx Frontend/components/dashboard/teacher/teacher-classes-panel.tsx Frontend/app/(dashboard)/teacher/page.tsx
git commit -m "feat(frontend): dedicated /classes page for teachers"
```

---

### Task 7: Frontend `/classes/[id]` page with Subjects + Materials + AI actions

**Files:**
- Create: `Frontend/app/(dashboard)/classes/[id]/page.tsx`
- Modify: `Frontend/lib/api.ts` (already done in Task 5)

**Interfaces:**
- Consumes: `classAPI.getClass`, `classSubjectAPI`, `classMaterialAPI`.
- Produces: A tabbed class-detail page for teachers. Subjects tab: list/add/delete subjects. Materials tab (inside a selected subject): upload/list/delete materials + Flashcards/Mock Test AI action buttons. Reuses the `ai-materials-sidebar.tsx` action pattern.

- [ ] **Step 1: Create the page**

Create `Frontend/app/(dashboard)/classes/[id]/page.tsx`:

```tsx
"use client"

import { useCallback, useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI, classSubjectAPI, classMaterialAPI } from "@/lib/api"
import { Loader2, Plus, Trash2, FileText, Layers, ClipboardList } from "lucide-react"
import { RoleGuard } from "@/components/auth/route-protection/role-guard"

interface Subject {
  id: string
  name: string
  icon?: string
}

interface Material {
  id: string
  name: string
  type: string
  doc_id?: string
  rag_indexed: boolean
}

export default function ClassDetailPage() {
  const { id } = useParams() as { id: string }
  const { toast } = useToast()
  const [cls, setCls] = useState<any>(null)
  const [subjects, setSubjects] = useState<Subject[]>([])
  const [activeSubject, setActiveSubject] = useState<string | null>(null)
  const [materials, setMaterials] = useState<Material[]>([])
  const [loading, setLoading] = useState(true)
  const [newSubject, setNewSubject] = useState("")
  const [uploading, setUploading] = useState(false)
  const [generating, setGenerating] = useState<string | null>(null)

  const loadClass = useCallback(async () => {
    try {
      const c = await classAPI.getClass(id)
      setCls(c)
    } catch (e) {
      toast({ title: "Couldn't load class", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, toast])

  const loadSubjects = useCallback(async () => {
    try {
      const res = await classSubjectAPI.list(id)
      const list = (res.subjects || []) as Subject[]
      setSubjects(list)
      if (list.length > 0 && !activeSubject) setActiveSubject(list[0].id)
    } catch (e) {
      toast({ title: "Couldn't load subjects", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, activeSubject, toast])

  const loadMaterials = useCallback(async () => {
    if (!activeSubject) return
    try {
      const res = await classMaterialAPI.list(id, activeSubject)
      setMaterials((res.materials || []) as Material[])
    } catch (e) {
      toast({ title: "Couldn't load materials", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, activeSubject, toast])

  useEffect(() => { setLoading(true); Promise.all([loadClass(), loadSubjects()]).finally(() => setLoading(false)) }, [loadClass, loadSubjects])
  useEffect(() => { loadMaterials() }, [loadMaterials])

  const handleAddSubject = async () => {
    if (!newSubject.trim()) return
    try {
      await classSubjectAPI.create(id, { name: newSubject.trim() })
      setNewSubject("")
      loadSubjects()
      toast({ title: "Subject added" })
    } catch (e) {
      toast({ title: "Couldn't add subject", description: getErrorMessage(e), variant: "destructive" })
    }
  }

  const handleDeleteSubject = async (subjectId: string) => {
    try {
      await classSubjectAPI.delete(id, subjectId)
      if (activeSubject === subjectId) setActiveSubject(null)
      loadSubjects()
      toast({ title: "Subject deleted" })
    } catch (e) {
      toast({ title: "Couldn't delete subject", description: getErrorMessage(e), variant: "destructive" })
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f || !activeSubject) return
    setUploading(true)
    try {
      await classMaterialAPI.upload(id, activeSubject, f)
      loadMaterials()
      toast({ title: "Material uploaded" })
    } catch (err) {
      toast({ title: "Upload failed", description: getErrorMessage(err), variant: "destructive" })
    } finally {
      setUploading(false)
    }
  }

  const handleGenerateFlashcards = async (material: Material) => {
    setGenerating(`flash-${material.id}`)
    try {
      const res = await classMaterialAPI.generateFlashcards(id, material.id)
      toast({ title: "Flashcards ready", description: `${res.card_count} cards generated.` })
    } catch (err) {
      toast({ title: "Flashcards failed", description: getErrorMessage(err), variant: "destructive" })
    } finally {
      setGenerating(null)
    }
  }

  const handleGenerateMockTest = async (material: Material) => {
    setGenerating(`mock-${material.id}`)
    try {
      const res = await classMaterialAPI.generateMockTest(id, material.id)
      toast({ title: "Mock test ready", description: `Test ID: ${res.test_id}` })
    } catch (err) {
      toast({ title: "Mock test failed", description: getErrorMessage(err), variant: "destructive" })
    } finally {
      setGenerating(null)
    }
  }

  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        {loading ? (
          <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
        ) : (
          <>
            <div>
              <h1 className="text-xl font-semibold">{cls?.name}</h1>
              {cls?.description && <p className="text-sm text-muted-foreground">{cls.description}</p>}
            </div>

            <div className="grid gap-6 lg:grid-cols-[240px,1fr]">
              <div className="space-y-4">
                <h2 className="text-sm font-semibold">Subjects</h2>
                <div className="flex gap-2">
                  <Input value={newSubject} onChange={(e) => setNewSubject(e.target.value)} placeholder="New subject" />
                  <Button size="icon" onClick={handleAddSubject} disabled={!newSubject.trim()}><Plus className="h-4 w-4" /></Button>
                </div>
                <div className="space-y-1">
                  {subjects.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => setActiveSubject(s.id)}
                      className={`w-full flex items-center justify-between rounded-md px-3 py-2 text-sm ${activeSubject === s.id ? "bg-secondary text-foreground" : "hover:bg-muted/50 text-muted-foreground"}`}
                    >
                      <span>{s.name}</span>
                      <Trash2 className="h-3.5 w-3.5 opacity-0 group-hover:opacity-100" onClick={(e) => { e.stopPropagation(); handleDeleteSubject(s.id) }} />
                    </button>
                  ))}
                  {subjects.length === 0 && <p className="text-xs text-muted-foreground px-1">No subjects yet.</p>}
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold">Materials</h2>
                  <div className="flex items-center gap-2">
                    {uploading && <Loader2 className="h-4 w-4 animate-spin" />}
                    <Label htmlFor="cm-upload" className="cursor-pointer">
                      <div className="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90">Upload material</div>
                      <input id="cm-upload" type="file" className="hidden" onChange={handleUpload} disabled={!activeSubject || uploading} />
                    </Label>
                  </div>
                </div>

                {!activeSubject ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Select a subject to view/upload materials.</div>
                ) : materials.length === 0 ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No materials yet for this subject. Upload one.</div>
                ) : (
                  <div className="grid gap-3">
                    {materials.map((m) => (
                      <div key={m.id} className="rounded-md border p-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <div className="text-sm font-medium">{m.name}</div>
                            <div className="text-xs text-muted-foreground">{m.rag_indexed ? "AI-ready" : "Not indexed"}</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <Button size="sm" variant="outline" disabled={!m.rag_indexed || !!generating} onClick={() => handleGenerateFlashcards(m)}>
                            {generating === `flash-${m.id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Layers className="h-3.5 w-3.5 mr-1" />}
                            Flashcards
                          </Button>
                          <Button size="sm" variant="outline" disabled={!m.rag_indexed || !!generating} onClick={() => handleGenerateMockTest(m)}>
                            {generating === `mock-${m.id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ClipboardList className="h-3.5 w-3.5 mr-1" />}
                            Mock Test
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </RoleGuard>
  )
}
```

- [ ] **Step 2: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/(dashboard)/classes/[id]/page.tsx
git commit -m "feat(frontend): class detail page with subjects, materials, and AI actions"
```

---

### Task 8: Teacher sidebar — remove Focus/Plans, add Classes

**Files:**
- Modify: `Frontend/components/dashboard/app-shell.tsx:40-49,145-155`

**Interfaces:**
- Consumes: `user.role` from `useAuth`.
- Produces: Teachers see Dashboard, Classes, Chat, Analysis, Mock Tests, Flashcards, Analytics, Billing, Settings. Students keep Focus/Plans.

- [ ] **Step 1: Add role-specific base nav**

In `Frontend/components/dashboard/app-shell.tsx`, replace the `baseNav` definition (lines 40-49) with:

```tsx
const studentNav = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/analysis", label: "Analysis", icon: FileBarChart },
  { href: "/mock-tests", label: "Mock Tests", icon: BookOpen },
  { href: "/flashcards", label: "Flashcards", icon: Sparkles },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/focus", label: "Focus", icon: Focus },
  { href: "/plans", label: "Plans", icon: Calendar },
]

const teacherNav = [
  { href: "/teacher", label: "Dashboard", icon: LayoutDashboard },
  { href: "/classes", label: "Classes", icon: Users },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/analysis", label: "Analysis", icon: FileBarChart },
  { href: "/mock-tests", label: "Mock Tests", icon: BookOpen },
  { href: "/flashcards", label: "Flashcards", icon: Sparkles },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
]
```

Add `Users` to the lucide imports at the top (line 6 currently imports many icons; add `Users` if missing).

- [ ] **Step 2: Use role-specific base nav in composition and update the nav-item type**

Replace the `allNav` composition (lines 145-155) with:

```tsx
  const allNav = (() => {
    const base = user?.role === "teacher" ? teacherNav : studentNav
    const list = [...base]
    list.push(billingNav)
    if (user?.role === "subadmin") {
      list.push(orgNav)
      list.push(adminNav)
    } else if (user?.role === "admin") {
      list.push(adminNav)
    }
    return list
  })()
```

The `SidebarNavItem` component in the same file likely uses `item: typeof baseNav[0]`. Since `baseNav` is replaced by `studentNav`/`teacherNav`, change the prop type to an inline shape:

```tsx
type NavItem = { href: string; label: string; icon: React.ComponentType<{ className?: string }> }
```

and update `SidebarNavItem`'s `item` prop to `NavItem`. If `React` is not already imported at the top of `app-shell.tsx`, add `import React from "react"`.

- [ ] **Step 3: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/dashboard/app-shell.tsx
git commit -m "feat(frontend): teacher sidebar gets Classes, loses Focus/Plans"
```

---

## Phase 2 completion checklist

- [ ] Backend tests pass: `pytest tests/test_class_router_p2.py tests/test_class_subjects.py tests/test_class_materials.py tests/test_class_material_generation.py -v`
- [ ] Frontend lint + build pass: `npm run lint && npm run build`
- [ ] A teacher can create a class, see it in `/classes`, open it, add subjects, upload materials to a subject, and generate flashcards/mock tests from a material.
- [ ] Co-teacher can be added and can access the same class.
- [ ] Generated flashcard decks and mock tests carry `class_id`/`class_subject_id`/`created_by` so Phase 4 can show them to students.

## Non-blocking follow-ups for later phases

- Extract the file-save/chunk/embed/index logic shared between `material_router.py` and `class_material_service.py` into a single helper.
- Add file-size and content-type validation on class-material upload.
- Implement the Tests/Students/Analytics tabs on `/classes/[id]` (Phase 3/4).
- Add a student view of `/classes/[id]` (Phase 4).