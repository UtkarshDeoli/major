# Orbit Coaching-Platform Reshape — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay the data-model + branding + curriculum foundation for the coaching-platform reshape: add a `curriculum` field to students at signup, extend the `Class` model for multi-teacher/org/subjects, add new collections + Pydantic models for class subjects/materials/invites, backfill existing data, and ship org logo/branding APIs.

**Architecture:** Extend existing FastAPI/MongoDB layers in place — no renames, all new fields optional with defaults. Add fields to `models.py`, collections to `data_store.py`, logic to `org_service.py` + `org_router.py`, a one-off migration in a new `src/core/migrations.py`. Frontend adds a curriculum select at signup and logo/tagline UI in the org onboarding + org pages.

**Tech Stack:** FastAPI, Motor (MongoDB), Pydantic v2, pytest (asyncio_mode=auto), Next.js 16 App Router, TypeScript, Tailwind, axios.

## Scope of this plan

This is **Phase 1 of 5**. Phases 2–5 (teacher class/subject/material + AI generation; actual tests + scheduling; student enrolled experience + branding display; AI chat fix) each get their own plan once the prior phase lands, because their tasks depend on the real interfaces this phase produces. Test-schema extensions (`mode`, `start_at`, etc.) are intentionally deferred to Phase 3.

## Global Constraints

- Backend runs from `Backend/` with the venv active: `source venv/bin/activate`. Tests: `pytest tests/ -v`. App: `python -m uvicorn src.main:app --reload --port 8001`.
- Frontend runs from `Frontend/`: `npm run dev` (port 3000), `npm run lint`, `npm run build`. There is **no frontend test runner** — frontend tasks are verified with `npm run lint` + `npm run build` + manual browser checks against the running dev server. Backend tasks use TDD with the existing fake-collection pattern.
- All new Mongo fields are optional with defaults so existing documents keep working — no destructive schema changes.
- Existing collections are extended in place, never renamed (`mock_tests_collection`, `classes_collection`, `organizations_collection`, etc.).
- Roles stay `student | teacher | subadmin | admin`. `subadmin` = coaching admin.
- Commit after every task. Branch off `master` first if not already on a feature branch.
- Logo files are stored on disk under `{UPLOADS_DIR}/orgs/{org_id}/`; `UPLOADS_DIR` defaults to `./uploads` (configurable). Logos are served by a public (no-auth) endpoint so `<img>` tags can render them without a Bearer token.

## File Structure

**Backend — create:**
- `Backend/src/core/migrations.py` — one-off P1 backfill (`run_coaching_p1_migration`) + future migrations.
- `Backend/scripts/run_migration.py` — CLI entrypoint that runs the migration against the real DB.

**Backend — modify:**
- `Backend/src/core/config.py` — add `UPLOADS_DIR`.
- `Backend/src/core/models.py` — add `curriculum` to `User`; extend `Class`; add `ClassSubject`, `ClassMaterial`, `ClassInvite` models.
- `Backend/src/core/data_store.py` — add `class_subjects_collection`, `class_materials_collection`, `class_invites_collection` (+ None fallbacks + indexes).
- `Backend/src/services/auth_service.py` — `create_user` accepts `curriculum`.
- `Backend/src/routers/auth_router.py` — `UserCreate`/`UserResponse`/`_build_user_response`/`/auth/me`/`signup` carry `curriculum`.
- `Backend/src/services/org_service.py` — `create_org` accepts `tagline`; add `update_org`, `get_org_by_org_id`, `upload_logo`, `get_logo_path`.
- `Backend/src/routers/org_router.py` — `OrgCreateRequest.tagline`; `PATCH /orgs/`, `POST /orgs/logo`, `GET /orgs/{org_id}/logo`, `GET /orgs/{org_id}/branding`.

**Frontend — modify:**
- `Frontend/lib/context/auth-context.tsx` — `User.curriculum`, `SignupPayload.curriculum`, `signup` posts it.
- `Frontend/components/auth/auth-form.tsx` — curriculum `<select>` (PRESET_EXAMS) on signup.
- `Frontend/lib/api.ts` — `orgAPI.uploadLogo`, `orgAPI.updateOrg`, `orgAPI.getBranding`.
- `Frontend/app/onboarding/org/page.tsx` — tagline field + logo upload on org creation.
- `Frontend/app/(dashboard)/org/page.tsx` — render logo + tagline; add edit-logo/upload control.

**Tests — create:**
- `Backend/tests/test_curriculum.py`
- `Backend/tests/test_migrations.py`
- `Backend/tests/test_org_branding.py`

---

### Task 1: Add `curriculum` field to the user stack (backend)

**Files:**
- Modify: `Backend/src/core/models.py:316` (User), `Backend/src/services/auth_service.py:74-128` (create_user), `Backend/src/routers/auth_router.py:36-156`
- Test: `Backend/tests/test_curriculum.py`

**Interfaces:**
- Consumes: existing `User` model, `create_user`, `UserResponse`, `/auth/me`.
- Produces: `User.curriculum: Optional[str]`; `create_user(..., curriculum=None)`; `UserResponse.curriculum`; `/auth/me` and `/auth/signup` responses include `curriculum`. Later phases read `curriculum` from the user doc.

- [ ] **Step 1: Write the failing test**

Create `Backend/tests/test_curriculum.py`:

```python
import uuid
from fastapi.testclient import TestClient
import src.services.auth_service as auth_service


class _FakeCollection:
    def __init__(self):
        self._docs = {}

    async def find_one(self, query):
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    async def insert_one(self, document):
        _id = str(uuid.uuid4())
        doc = dict(document)
        doc["_id"] = _id
        self._docs[_id] = doc

        class _Result:
            inserted_id = _id
        return _Result()


import pytest


@pytest.fixture
def client(monkeypatch):
    fake_users = _FakeCollection()
    monkeypatch.setattr(auth_service, "users_collection", fake_users)
    from src.main import app
    return TestClient(app)


def test_signup_persists_and_returns_curriculum(client):
    unique = uuid.uuid4().hex
    payload = {
        "email": f"curr-{unique}@example.com",
        "password": "password123",
        "name": "Curriculum Student",
        "curriculum": "jee-mains",
    }
    r = client.post("/auth/signup", json=payload)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["user"]["curriculum"] == "jee-mains"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert me.status_code == 200
    assert me.json()["curriculum"] == "jee-mains"


def test_signup_curriculum_is_optional(client):
    unique = uuid.uuid4().hex
    r = client.post("/auth/signup", json={
        "email": f"nocurr-{unique}@example.com",
        "password": "password123",
    })
    assert r.status_code == 201, r.text
    assert r.json()["user"]["curriculum"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd Backend && source venv/bin/activate && pytest tests/test_curriculum.py -v`
Expected: FAIL — `KeyError: 'curriculum'` (response model drops unknown field) or `TypeError` (create_user doesn't accept curriculum).

- [ ] **Step 3: Add `curriculum` to the `User` model**

In `Backend/src/core/models.py`, inside `class User` (after the `institute` field at line 316), add:

```python
    curriculum: Optional[str] = None      # exam-preset key (e.g. "jee-mains") chosen at signup
```

- [ ] **Step 4: Make `create_user` accept `curriculum`**

In `Backend/src/services/auth_service.py`, change the `create_user` signature (line 74) to add the param and set it on the `User`:

```python
async def create_user(
    email: str,
    password: str,
    name: Optional[str] = None,
    role: str = "student",
    institute: Optional[str] = None,
    preferred_language: Optional[str] = None,
    curriculum: Optional[str] = None,
):
```

And in the `User(...)` construction (around line 108), add `curriculum=curriculum,`:

```python
        user = User(
            email=email,
            password_hash=hashed_password,
            name=name,
            role=role,  # type: ignore[arg-type]
            institute=institute,
            preferred_language=preferred_language or "en",
            auth_provider="email",
            curriculum=curriculum,
        )
```

- [ ] **Step 5: Carry `curriculum` through the auth router**

In `Backend/src/routers/auth_router.py`:

Add to `class UserCreate` (line 36):
```python
    curriculum: Optional[str] = None
```

Add to `class UserResponse` (line 42), after `institute`:
```python
    curriculum: Optional[str] = None
```

In `_build_user_response` (line 71), add to the `UserResponse(...)` call:
```python
        curriculum=user.get("curriculum"),
```

In `signup` (line 90), pass it to `create_user`:
```python
    user = await create_user(
        email=user_data.email,
        password=user_data.password,
        name=user_data.name,
        curriculum=user_data.curriculum,
    )
```

In `get_me` (line 133), add to the returned `UserResponse(...)`:
```python
        curriculum=user.get("curriculum"),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd Backend && source venv/bin/activate && pytest tests/test_curriculum.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Run the full auth suite to confirm no regression**

Run: `pytest tests/test_auth.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Backend/src/core/models.py Backend/src/services/auth_service.py Backend/src/routers/auth_router.py Backend/tests/test_curriculum.py
git commit -m "feat(auth): add curriculum field to user signup and profile"
```

---

### Task 2: Extend `Class` model + add class-subject/material/invite models & collections

**Files:**
- Modify: `Backend/src/core/models.py:336-347` (Class), `Backend/src/core/data_store.py:33-91,432-465`
- Test: `Backend/tests/test_class_models.py`

**Interfaces:**
- Consumes: existing `Class` model, `data_store` collection pattern.
- Produces: `Class.org_id`, `Class.teacher_ids`, `Class.subject_ids`; new models `ClassSubject`, `ClassMaterial`, `ClassInvite`; new collections `class_subjects_collection`, `class_materials_collection`, `class_invites_collection` (importable from `src.core.data_store`). Phase 2 uses these.

- [ ] **Step 1: Write the failing test**

Create `Backend/tests/test_class_models.py`:

```python
import pytest
from src.core.models import Class, ClassSubject, ClassMaterial, ClassInvite
from src.core import data_store as ds


def test_class_supports_multi_teacher_and_org():
    c = Class(
        teacher_id="t@x.com",
        name="JEE 2026",
        enroll_code="ABC123",
        org_id="org-1",
        teacher_ids=["t@x.com", "t2@x.com"],
        subject_ids=[],
    )
    assert c.org_id == "org-1"
    assert c.teacher_ids == ["t@x.com", "t2@x.com"]
    assert c.subject_ids == []


def test_class_subject_model():
    s = ClassSubject(class_id="c1", name="Physics", created_by="t@x.com")
    assert s.class_id == "c1"
    assert s.name == "Physics"


def test_class_material_model():
    m = ClassMaterial(
        class_id="c1", class_subject_id="s1", teacher_id="t@x.com",
        name="notes.pdf", type="pdf", size=1024, doc_id="doc-1", rag_indexed=True,
    )
    assert m.doc_id == "doc-1" and m.rag_indexed is True


def test_class_invite_model():
    inv = ClassInvite(class_id="c1", email="stu@x.com", token="tok", status="pending")
    assert inv.status == "pending"


def test_new_collections_exist():
    # They are None only if MongoDB never connected; in the test env they may be None,
    # so we only assert the attributes are present (not left undefined).
    assert hasattr(ds, "class_subjects_collection")
    assert hasattr(ds, "class_materials_collection")
    assert hasattr(ds, "class_invites_collection")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_class_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'ClassSubject'` / `AttributeError`.

- [ ] **Step 3: Extend the `Class` model**

In `Backend/src/core/models.py`, replace the `Class` body (lines 336-347) with:

```python
# Teacher Class / Batch model — a teacher groups students (e.g. "JEE 2026 Batch")
class Class(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    teacher_id: str                      # teacher email who owns the class (creator)
    name: str                            # e.g. "JEE 2026 Batch"
    description: Optional[str] = None
    exam_preset: Optional[str] = None    # e.g. "jee-mains" — drives default subjects
    enroll_code: str                     # short shareable code
    student_emails: List[str] = []
    # Coaching-platform fields (Phase 1)
    org_id: Optional[str] = None         # coaching this class belongs to
    teacher_ids: List[str] = []          # creator + co-teachers (all same org)
    subject_ids: List[str] = []          # references to class_subjects
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# A subject inside a class — any teacher in the class's org may add one.
class ClassSubject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    class_id: str
    name: str
    icon: Optional[str] = None
    created_by: str                      # teacher email
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# A material uploaded to a class subject. doc_id links into pdfs_collection + RAG.
class ClassMaterial(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    class_id: str
    class_subject_id: str
    teacher_id: str                      # uploader
    name: str
    type: Literal["pdf", "image", "text"] = "pdf"
    size: int = 0
    page_count: Optional[int] = None
    doc_id: Optional[str] = None         # links to pdfs_collection for RAG
    rag_indexed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Pending invite for a student who does not yet have an account.
class ClassInvite(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    class_id: str
    email: str
    token: str
    status: Literal["pending", "used"] = "pending"
    created_by: str                      # teacher email
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    used_at: Optional[datetime] = None
```

- [ ] **Step 4: Add the new collections to `data_store.py`**

In `Backend/src/core/data_store.py`, in the `try` block after `classes_collection = db.classes` (line 50), add:

```python
    class_subjects_collection = db.class_subjects
    class_materials_collection = db.class_materials
    class_invites_collection = db.class_invites
```

In the `except` block after `classes_collection = None` (line 82), add:

```python
    class_subjects_collection = None
    class_materials_collection = None
    class_invites_collection = None
```

- [ ] **Step 5: Add indexes for the new collections**

In `ensure_indexes()` (after the `classes_collection` index block, around line 445), add:

```python
    if class_subjects_collection is not None:
        await class_subjects_collection.create_index([("class_id", 1), ("name", 1)])
    if class_materials_collection is not None:
        await class_materials_collection.create_index([("class_id", 1), ("class_subject_id", 1)])
    if class_invites_collection is not None:
        await class_invites_collection.create_index([("class_id", 1), ("email", 1)])
        await class_invites_collection.create_index("token", unique=True)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_class_models.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Run a broad smoke check**

Run: `pytest tests/test_orgs.py tests/test_auth.py -v`
Expected: PASS (no regression from new model fields).

- [ ] **Step 8: Commit**

```bash
git add Backend/src/core/models.py Backend/src/core/data_store.py Backend/tests/test_class_models.py
git commit -m "feat(model): extend Class for org/multi-teacher/subjects + add ClassSubject/ClassMaterial/ClassInvite"
```

---

### Task 3: P1 migration — backfill existing classes

**Files:**
- Create: `Backend/src/core/migrations.py`, `Backend/scripts/run_migration.py`
- Test: `Backend/tests/test_migrations.py`

**Interfaces:**
- Consumes: `classes_collection`, `users_collection` (passed in, or imported from data_store at runtime).
- Produces: `async def run_coaching_p1_migration(classes_coll, users_coll=None) -> dict` returning `{"classes_backfilled": int}`. Idempotent: only sets fields that are missing.

- [ ] **Step 1: Write the failing test**

Create `Backend/tests/test_migrations.py`:

```python
import pytest
from src.core.migrations import run_coaching_p1_migration


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._docs):
            raise StopAsyncIteration
        d = dict(self._docs[self._i])
        self._i += 1
        return d


class _FakeColl:
    def __init__(self, docs):
        self.docs = docs

    def find(self, q=None):
        return _AsyncCursor(self.docs.values())

    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                return dict(d)
        return None

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return


@pytest.mark.asyncio
async def test_backfill_adds_teacher_ids_subject_ids_org_id():
    classes = _FakeColl({
        "1": {"_id": "1", "teacher_id": "t@x.com", "enroll_code": "X1"},
    })
    users = _FakeColl({
        "t": {"email": "t@x.com", "org_id": "org-9"},
    })
    res = await run_coaching_p1_migration(classes, users)
    assert res["classes_backfilled"] == 1
    assert classes.docs["1"]["teacher_ids"] == ["t@x.com"]
    assert classes.docs["1"]["subject_ids"] == []
    assert classes.docs["1"]["org_id"] == "org-9"


@pytest.mark.asyncio
async def test_backfill_is_idempotent():
    classes = _FakeColl({
        "1": {"_id": "1", "teacher_id": "t@x.com", "teacher_ids": ["t@x.com"],
              "subject_ids": [], "org_id": "org-9"},
    })
    res = await run_coaching_p1_migration(classes, _FakeColl({}))
    assert res["classes_backfilled"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrations.py -v`
Expected: FAIL — `ModuleNotFoundError: src.core.migrations`.

- [ ] **Step 3: Create the migration module**

Create `Backend/src/core/migrations.py`:

```python
"""One-off data migrations for the coaching-platform reshape.

Each function is idempotent: it only sets fields that are missing, so it is
safe to run repeatedly. Run via `python scripts/run_migration.py`.
"""
from typing import Optional


async def run_coaching_p1_migration(classes_coll, users_coll=None) -> dict:
    """Backfill existing classes with Phase 1 fields.

    - teacher_ids: [teacher_id] if missing
    - subject_ids: [] if missing
    - org_id: derived from the teacher's user doc if missing
    """
    cursor = classes_coll.find({})
    count = 0
    async for cls in cursor:
        update = {}
        if not cls.get("teacher_ids"):
            tid = cls.get("teacher_id")
            update["teacher_ids"] = [tid] if tid else []
        if "subject_ids" not in cls:
            update["subject_ids"] = []
        if not cls.get("org_id") and users_coll is not None:
            teacher = await users_coll.find_one({"email": cls.get("teacher_id")})
            update["org_id"] = teacher.get("org_id") if teacher else None
        if update:
            await classes_coll.update_one({"_id": cls["_id"]}, {"$set": update})
            count += 1
    return {"classes_backfilled": count}
```

- [ ] **Step 4: Create the CLI runner**

Create `Backend/scripts/run_migration.py`:

```python
"""Run coaching-platform migrations against the real database.

Usage:  source venv/bin/activate && python scripts/run_migration.py
"""
import asyncio

from src.core.data_store import classes_collection, users_collection
from src.core.migrations import run_coaching_p1_migration


async def main():
    if classes_collection is None:
        raise SystemExit("MongoDB not connected; cannot run migration.")
    res = await run_coaching_p1_migration(classes_collection, users_collection)
    print("Migration result:", res)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_migrations.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add Backend/src/core/migrations.py Backend/scripts/run_migration.py Backend/tests/test_migrations.py
git commit -m "feat(migrations): idempotent P1 backfill for class org/teacher_ids/subject_ids"
```

---

### Task 4: Org branding — tagline on create + update + branding endpoint

**Files:**
- Modify: `Backend/src/services/org_service.py:20-51` (create_org) + new functions, `Backend/src/routers/org_router.py:12-69`
- Test: `Backend/tests/test_org_branding.py`

**Interfaces:**
- Consumes: `organizations_collection`, `get_current_user_with_role`.
- Produces: `create_org(..., tagline=None)`; `async def update_org(owner_id, brand_name, tagline)`; `async def get_org_by_org_id(org_id)`; `GET /orgs/{org_id}/branding` → `{org_id, name, brand_name, logo_url, tagline}` (authenticated, any logged-in user); `PATCH /orgs/` → `{updated: True}`.

- [ ] **Step 1: Write the failing test**

Create `Backend/tests/test_org_branding.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_org_branding.py -v`
Expected: FAIL — `tagline` ignored on create; `PATCH /orgs/` and `/orgs/{org_id}/branding` return 405/404.

- [ ] **Step 3: Extend `create_org` and add service functions**

In `Backend/src/services/org_service.py`, change `create_org` signature (line 20) to accept `tagline` and store it:

```python
async def create_org(owner_id: str, name: str, brand_name: Optional[str],
                     tier: str, seats_total: int, billing_cycle: str,
                     tagline: Optional[str] = None) -> dict:
```

In the `doc = {...}` dict (around line 29), add `"tagline": tagline,`:

```python
    doc = {
        "org_id": org_id, "name": name, "brand_name": brand_name,
        "tagline": tagline, "logo_url": None, "logo_file_path": None,
        "owner_user_id": owner_id, "tier": tier, "seats_total": seats_total,
        "seats_used": 0, "status": "active", "billing_cycle": billing_cycle,
        "created_at": now, "updated_at": now,
    }
```

Append these functions at the end of `org_service.py`:

```python
async def get_org_by_org_id(org_id: str) -> Optional[dict]:
    if organizations_collection is None:
        return None
    return await organizations_collection.find_one({"org_id": org_id})


async def update_org(owner_id: str, brand_name: Optional[str], tagline: Optional[str]) -> dict:
    if organizations_collection is None:
        raise HTTPException(503, "Database connection not available")
    org = await get_org_by_owner(owner_id)
    if not org:
        raise HTTPException(404, "No organization found")
    update: dict = {"updated_at": datetime.now(timezone.utc)}
    if brand_name is not None:
        update["brand_name"] = brand_name
    if tagline is not None:
        update["tagline"] = tagline
    await organizations_collection.update_one(
        {"org_id": org["org_id"]}, {"$set": update})
    return {"updated": True}


def public_branding(org: dict) -> dict:
    return {
        "org_id": org.get("org_id"),
        "name": org.get("name"),
        "brand_name": org.get("brand_name"),
        "logo_url": org.get("logo_url"),
        "tagline": org.get("tagline"),
    }
```

- [ ] **Step 4: Add the router endpoints**

In `Backend/src/routers/org_router.py`:

Add `tagline` to `OrgCreateRequest` (line 12):

```python
class OrgCreateRequest(BaseModel):
    name: str
    brand_name: Optional[str] = None
    tagline: Optional[str] = None
    tier: Literal["pro", "premium"]
    seats_total: int = 1
    billing_cycle: Literal["monthly", "yearly"] = "monthly"
```

Add a new request model near the others:

```python
class OrgUpdateRequest(BaseModel):
    brand_name: Optional[str] = None
    tagline: Optional[str] = None
```

Pass `tagline` in the `create_org` call (line 31):

```python
    return await svc.create_org(user["email"], req.name, req.brand_name,
                                req.tier, req.seats_total, req.billing_cycle, req.tagline)
```

Add the two new endpoints (after `create_org`, before `GET /me`):

```python
@router.patch("/")
async def update_org(req: OrgUpdateRequest, user=Depends(require_role("subadmin"))):
    return await svc.update_org(user["email"], req.brand_name, req.tagline)


@router.get("/{org_id}/branding")
async def get_branding(org_id: str = Path(...), user=Depends(get_current_user_with_role)):
    org = await svc.get_org_by_org_id(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    return svc.public_branding(org)
```

Ensure `Path` is imported (it already is at line 3: `from fastapi import APIRouter, Depends, Path, HTTPException, status`).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_org_branding.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the org suite for regressions**

Run: `pytest tests/test_orgs.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Backend/src/services/org_service.py Backend/src/routers/org_router.py Backend/tests/test_org_branding.py
git commit -m "feat(org): tagline on create/update + public branding endpoint"
```

---

### Task 5: Org logo upload + logo serving endpoint

**Files:**
- Modify: `Backend/src/core/config.py` (add `UPLOADS_DIR`), `Backend/src/services/org_service.py` (logo functions), `Backend/src/routers/org_router.py` (logo endpoints)
- Test: `Backend/tests/test_org_logo.py`

**Interfaces:**
- Consumes: `UPLOADS_DIR` from config, `organizations_collection`, `require_role("subadmin")`, `get_current_user_with_role`.
- Produces: `async def upload_logo(owner_id, file, uploads_dir=None) -> {"logo_url": str}` (defaults to the module-level `UPLOADS_DIR`); `async def get_logo_path(org_id) -> str | None`; `POST /orgs/logo` (multipart, subadmin); `GET /orgs/{org_id}/logo` (public FileResponse). `logo_url` is stored as `/orgs/{org_id}/logo`. The router calls `svc.upload_logo(user["email"], file)` and does **not** import or pass `UPLOADS_DIR` — the service reads its own module global, so tests monkeypatch `src.services.org_service.UPLOADS_DIR`.

- [ ] **Step 1: Add `UPLOADS_DIR` to config**

In `Backend/src/core/config.py`, add to `Settings` (after `CHROMA_DB_PATH`, line 18):

```python
    UPLOADS_DIR: str = "./uploads"
```

And export it at the bottom (after `CHROMA_DB_PATH = settings.CHROMA_DB_PATH`, line 53):

```python
UPLOADS_DIR = settings.UPLOADS_DIR
```

- [ ] **Step 2: Write the failing test**

Create `Backend/tests/test_org_logo.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_org_logo.py -v`
Expected: FAIL — `POST /orgs/logo` returns 404/405.

- [ ] **Step 4: Add logo service functions**

At the top of `Backend/src/services/org_service.py`, update imports:

```python
"""Organization (coaching/school) + seat license logic."""
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, UploadFile

from src.core.data_store import (
    users_collection, organizations_collection, org_invites_collection,
)
from src.services import billing_service
from src.core.config import RAZORPAY_KEY_ID, UPLOADS_DIR
from src.core.plans import PLAN_PRICES
```

Append at the end of `org_service.py`:

```python
async def upload_logo(owner_id: str, file: UploadFile, uploads_dir: Optional[str] = None) -> dict:
    if organizations_collection is None:
        raise HTTPException(503, "Database connection not available")
    org = await get_org_by_owner(owner_id)
    if not org:
        raise HTTPException(404, "No organization found for your account")
    content = await file.read()
    uploads_dir = uploads_dir or UPLOADS_DIR
    org_dir = os.path.join(uploads_dir, "orgs", org["org_id"])
    os.makedirs(org_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "logo.png")[1] or ".png"
    filename = f"logo{ext}"
    path = os.path.join(org_dir, filename)
    with open(path, "wb") as f:
        f.write(content)
    logo_url = f"/orgs/{org['org_id']}/logo"
    await organizations_collection.update_one(
        {"org_id": org["org_id"]},
        {"$set": {"logo_url": logo_url, "logo_file_path": path,
                  "updated_at": datetime.now(timezone.utc)}},
    )
    return {"logo_url": logo_url}


async def get_logo_path(org_id: str) -> Optional[str]:
    if organizations_collection is None:
        return None
    org = await organizations_collection.find_one({"org_id": org_id})
    if not org:
        return None
    return org.get("logo_file_path")
```

- [ ] **Step 5: Add the logo router endpoints**

In `Backend/src/routers/org_router.py`, update imports:

```python
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Path, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.core.security import require_role, get_current_user_with_role
from src.services import org_service as svc
```

Add the endpoints (place them after `GET /orgs/{org_id}/branding`):

```python
@router.post("/logo")
async def upload_logo(file: UploadFile = File(...), user=Depends(require_role("subadmin"))):
    return await svc.upload_logo(user["email"], file)


@router.get("/{org_id}/logo")
async def get_logo(org_id: str = Path(...)):
    path = await svc.get_logo_path(org_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No logo on file")
    return FileResponse(path, media_type="image/*")
```

Add `import os` at the top of `org_router.py` (after the other imports).

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_org_logo.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Run the whole org + branding suites**

Run: `pytest tests/test_org_logo.py tests/test_org_branding.py tests/test_orgs.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Backend/src/core/config.py Backend/src/services/org_service.py Backend/src/routers/org_router.py Backend/tests/test_org_logo.py
git commit -m "feat(org): logo upload (multipart) + public logo serving endpoint"
```

---

### Task 6: Frontend — curriculum select at signup

**Files:**
- Modify: `Frontend/lib/context/auth-context.tsx:17-42,66-70,216-233`, `Frontend/components/auth/auth-form.tsx:25-58`
- Verify: `npm run lint && npm run build` + manual browser check.

**Interfaces:**
- Consumes: `PRESET_EXAMS` from `@/lib/constants/exams`; `signup` from `useAuth()`.
- Produces: `User.curriculum?: string`; `SignupPayload.curriculum?: string`; the signup form posts `curriculum` (a preset id or omitted). Backend (Task 1) stores + returns it.

- [ ] **Step 1: Add `curriculum` to the `User` and `SignupPayload` types**

In `Frontend/lib/context/auth-context.tsx`, add to `interface User` (after `institute?: string;` around line 30):

```ts
  curriculum?: string;
```

Add to `SignupPayload` (lines 66-70):

```ts
export interface SignupPayload {
  email: string;
  password: string;
  name?: string;
  curriculum?: string;
}
```

- [ ] **Step 2: Post `curriculum` from the `signup` function**

The `signup` function (line 216) already posts `payload` straight to `/auth/signup`, so no change is needed there — `payload` now may include `curriculum`. Verify line 220 reads `const response = await api.post("/auth/signup", payload);` (it does).

- [ ] **Step 3: Add a curriculum `<select>` to the signup form**

In `Frontend/components/auth/auth-form.tsx`:

Add the import near the top (after the existing imports):

```tsx
import { PRESET_EXAMS } from "@/lib/constants/exams"
```

Add state (after line 27, next to the other `useState` calls):

```tsx
  const [curriculum, setCurriculum] = useState('')
```

In `handleSubmit` (lines 37-41), pass `curriculum`:

```tsx
        await signup({
          email,
          password,
          name: name || undefined,
          curriculum: curriculum || undefined,
        });
```

Add the select inside the form, just before the submit button (render it only for signup — `type === 'signup'`). Use the same label/input styling as the surrounding fields:

```tsx
      {type === 'signup' && (
        <div className="grid gap-2">
          <label htmlFor="curriculum" className="text-sm font-medium">What are you preparing for? (optional)</label>
          <select
            id="curriculum"
            value={curriculum}
            onChange={(e) => setCurriculum(e.target.value)}
            className="border rounded-md px-3 py-2 bg-background"
          >
            <option value="">Select an exam / curriculum</option>
            {PRESET_EXAMS.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>
      )}
```

- [ ] **Step 4: Lint + build**

Run:
```bash
cd Frontend && npm run lint && npm run build
```
Expected: no errors.

- [ ] **Step 5: Manual verification**

Run `npm run dev` (frontend) and `python -m uvicorn src.main:app --reload --port 8001` (backend, venv active). Open `http://localhost:3000/signup`:
- The curriculum `<select>` appears with all preset exams.
- Sign up choosing "JEE" — request body to `/auth/signup` includes `curriculum: "jee-mains"` (check the Network tab).
- After signup, `localStorage` user / `/auth/me` shows `curriculum: "jee-mains"`.

- [ ] **Step 6: Commit**

```bash
git add Frontend/lib/context/auth-context.tsx Frontend/components/auth/auth-form.tsx
git commit -m "feat(frontend): curriculum select on signup"
```

---

### Task 7: Frontend — org logo upload + tagline UI

**Files:**
- Modify: `Frontend/lib/api.ts:719-765` (orgAPI), `Frontend/app/onboarding/org/page.tsx:21-60`, `Frontend/app/(dashboard)/org/page.tsx:41-159`
- Verify: `npm run lint && npm run build` + manual browser check.

**Interfaces:**
- Consumes: backend `POST /orgs/logo` (multipart), `PATCH /orgs/`, `GET /orgs/{org_id}/branding`, `GET /orgs/{org_id}/logo`. The `User` type has `org_id`.
- Produces: `orgAPI.uploadLogo(file)`, `orgAPI.updateOrg({brand_name?, tagline?})`, `orgAPI.getBranding(orgId)`. Org onboarding + org pages can upload a logo and show it.

- [ ] **Step 1: Add the new `orgAPI` methods**

In `Frontend/lib/api.ts`, inside the `orgAPI` object (after `previewEnroll`, before the closing `};` at line 765), add:

```ts
  async uploadLogo(file: File): Promise<{ logo_url: string }> {
    const form = new FormData();
    form.append("file", file);
    const res = await api.post("/orgs/logo", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  async updateOrg(payload: { brand_name?: string; tagline?: string }): Promise<any> {
    const res = await api.patch("/orgs/", payload);
    return res.data;
  },

  async getBranding(orgId: string): Promise<{
    org_id: string; name: string; brand_name?: string | null; logo_url?: string | null; tagline?: string | null;
  }> {
    const res = await api.get(`/orgs/${encodeURIComponent(orgId)}/branding`);
    return res.data;
  },
```

- [ ] **Step 2: Add a logo `<img>` helper that prefixes the API base**

The logo URL returned by the backend is a relative path like `/orgs/{org_id}/logo`. The axios instance already has `baseURL` set (from `NEXT_PUBLIC_API_URL`). For an `<img>`, build the full URL from the same env var. At the top of `lib/api.ts` (near the axios instance creation), confirm the base is exported; if not, expose it. Add near the `api` instance:

```ts
export const API_BASE_URL = (api.defaults.baseURL || "").replace(/\/$/, "");
```

(If `api.defaults.baseURL` is unset, instead use `export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";`.)

- [ ] **Step 3: Add tagline + logo upload to org onboarding**

In `Frontend/app/onboarding/org/page.tsx`:

Add state (after line 25):

```tsx
  const [tagline, setTagline] = useState("");
  const [logoFile, setLogoFile] = useState<File | null>(null);
```

Add imports:

```tsx
import { API_BASE_URL, orgAPI } from "@/lib/api";
```

In `handleSubmit` (lines 42-60), include `tagline` in `orgAPI.create` and upload the logo after the org is created. Replace the body of the `try` block:

```tsx
    try {
      const created: any = await orgAPI.create({
        name,
        brand_name: brandName || undefined,
        tagline: tagline || undefined,
        tier,
        seats_total: seats,
        billing_cycle: billingCycle,
      });
      if (logoFile) {
        try {
          await orgAPI.uploadLogo(logoFile);
        } catch (e) {
          // non-fatal: org is created; logo can be added later from /org
        }
      }
      toast({ title: "Organization created", description: "You can now manage seats and invites from /org." });
      await refreshUser();
      router.push("/admin");
    } catch (error) {
      toast({ title: "Could not create organization", description: getErrorMessage(error), variant: "destructive" });
      setIsSubmitting(false);
    }
```

Note: `orgAPI.create` currently returns the backend `{org_id, checkout}` object (see `lib/api.ts:726`), so `created` is available if needed; the logo upload uses the authenticated subadmin identity server-side, so no org_id is required in the request.

Add the two form fields inside the form (before the submit button, around line 147): a tagline text input and a logo file input:

```tsx
          <div className="grid gap-2">
            <label htmlFor="tagline" className="text-sm font-medium">Tagline (optional)</label>
            <input id="tagline" value={tagline} onChange={(e) => setTagline(e.target.value)}
              className="border rounded-md px-3 py-2" placeholder="e.g. Dream. Prepare. Achieve." />
          </div>
          <div className="grid gap-2">
            <label htmlFor="logo" className="text-sm font-medium">Coaching logo (optional)</label>
            <input id="logo" type="file" accept="image/*"
              onChange={(e) => setLogoFile(e.target.files?.[0] || null)}
              className="border rounded-md px-3 py-2" />
            {logoFile && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={URL.createObjectURL(logoFile)} alt="logo preview" className="h-12 w-12 object-contain" />
            )}
          </div>
```

- [ ] **Step 4: Show the logo + tagline on the `/org` page**

In `Frontend/app/(dashboard)/org/page.tsx`, the `org` state already holds the full org doc returned by `orgAPI.getMe()`, which now includes `logo_url` and `tagline`. Update the org card header (lines 154-159) to render the logo and tagline. Replace that block with:

```tsx
                    <div className="flex items-center gap-3">
                      {org.logo_url && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={`${API_BASE_URL}${org.logo_url}`} alt={org.brand_name || org.name}
                             className="h-10 w-10 rounded-md object-contain border" />
                      )}
                      <div>
                        <CardTitle className="text-base">{org.brand_name || org.name}</CardTitle>
                        <CardDescription className="capitalize">{org.tier} tier · {org.status}</CardDescription>
                        {org.tagline && <p className="text-xs text-muted-foreground mt-1">{org.tagline}</p>}
                      </div>
                    </div>
```

Add the import at the top:

```tsx
import { API_BASE_URL } from "@/lib/api";
```

Add an "Upload logo" control inside the org card (after the add-seats form, before the invite card). A simple file input that calls `orgAPI.uploadLogo` then reloads:

```tsx
                  <div className="mt-4 grid gap-2">
                    <label htmlFor="org-logo" className="text-sm font-medium">Logo</label>
                    <input id="org-logo" type="file" accept="image/*"
                      onChange={async (e) => {
                        const f = e.target.files?.[0];
                        if (!f) return;
                        try {
                          await orgAPI.uploadLogo(f);
                          await loadOrg();
                          toast({ title: "Logo updated" });
                        } catch (err) {
                          toast({ title: "Logo upload failed", description: getErrorMessage(err), variant: "destructive" });
                        }
                      }}
                      className="border rounded-md px-3 py-2" />
                  </div>
```

- [ ] **Step 5: Lint + build**

Run:
```bash
cd Frontend && npm run lint && npm run build
```
Expected: no errors. (If `@next/next/no-img-element` lint warnings appear, the inline disables handle them; ensure they don't fail the build.)

- [ ] **Step 6: Manual end-to-end verification**

Run both servers. As a subadmin (create one via the org onboarding flow):
- On `/onboarding/org`, fill name + tagline + pick a logo file → submit → org created → logo uploaded.
- On `/org`, the org card shows the logo image, brand name, and tagline.
- Replacing the logo via the `/org` file input updates the displayed image.
- In the Network tab, `GET /orgs/me` returns `logo_url` and `tagline` on the org; `GET /orgs/{org_id}/logo` returns the image with 200.

- [ ] **Step 7: Commit**

```bash
git add Frontend/lib/api.ts Frontend/app/onboarding/org/page.tsx "Frontend/app/(dashboard)/org/page.tsx"
git commit -m "feat(frontend): org logo upload + tagline UI in onboarding and /org"
```

---

## Phase 1 completion checklist

- [ ] `pytest tests/ -v` passes (new + existing suites green).
- [ ] `npm run lint && npm run build` pass.
- [ ] A student can sign up with a curriculum and `/auth/me` returns it.
- [ ] A subadmin can create an org with a tagline, upload a logo, and the logo renders on `/org`.
- [ ] `GET /orgs/{org_id}/branding` returns public branding (no internal `logo_file_path`).
- [ ] `GET /orgs/{org_id}/logo` serves the image publicly; 404 when none.
- [ ] `python scripts/run_migration.py` backfills existing classes (`teacher_ids`, `subject_ids`, `org_id`) and is idempotent.

## What the next phases will build (not in this plan)

- **Phase 2:** teacher `/classes` + `/classes/[id]` pages, `ClassSubject` CRUD, class-scoped material upload + RAG, generate flashcards/mock from a class material.
- **Phase 3:** `mode=actual` tests with `start_at`/`end_at`, timed runner, no student feedback, teacher attempts/marks view (+ test/submission `mode` schema).
- **Phase 4:** student enrolled experience — coaching banner/logo on home, classes section, class detail (subjects/materials/tests), org auto-enroll on class join; coaching-admin monitoring tabs.
- **Phase 5:** AI chat fix — thread conversation history into Gemini + free (no-material) chat + visible session history.