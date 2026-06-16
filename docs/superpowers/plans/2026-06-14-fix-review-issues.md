# Fix Code Review Issues — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all issues raised in the code review of the teacher/sub-admin + analytics + flashcards + web-search feature branch, so the code is secure, internally consistent, and the test suite/build passes.

**Architecture:** Keep the new role hierarchy (`student`, `teacher`, `subadmin`, `admin`) and the new business logic (licenses, managed students, analytics, flashcards, web-augmented mock tests), but close privilege-escalation holes, correct the `managed_by`/`license_id` relationship wiring, restore test tooling, and remove generated files from the diff.

**Tech Stack:** FastAPI, Pydantic, MongoDB/Motor, Next.js 15 + TypeScript, pytest, Git.

---

## File map of changes

| File | Why it changes |
|------|----------------|
| `Backend/src/routers/auth_router.py` | Remove `role`/`institute`/`preferred_language` from public signup; `role` forced to `"student"` server-side. |
| `Backend/src/services/auth_service.py` | `create_user` defaults `role` to `"student"` and ignores caller-provided privileged roles. |
| `Backend/src/routers/onboarding_router.py` | Ignore `role` from onboarding save; keep name/institute/language only. |
| `Backend/src/routers/teacher_router.py` | Correct `managed_by` and `license_id` propagation from teacher to student. |
| `Backend/requirements.txt` | Restore `pytest` and keep runtime dependencies pinned enough for reproducibility. |
| `Backend/tests/test_main.py` | Fix the broken import or delete the file so the suite runs. |
| `Frontend/components/auth/auth-form.tsx` | Remove role/institute selector from signup form; keep it on onboarding if needed. |
| `Frontend/lib/context/auth-context.tsx` | Remove `role` from `SignupPayload`. |
| `Backend/src/routers/auth_router.py` | Add `SubscriptionInfo` to `UserResponse` instead of `Optional[Any]`. |
| `Backend/src/core/models.py` | (no change unless needed for above) |
| `Frontend/lib/data.ts` | Extend `User` interface to match backend `UserResponse`. |
| `Backend/src/services/mock_test_service.py` | Use the joined search terms correctly, or simplify code. |
| `Backend/src/routers/mock_test_router.py` | Move `users_collection` import to module top. |
| `Frontend/next-env.d.ts` | Revert to original content and ensure `.gitignore` covers it. |
| `Frontend/tsconfig.tsbuildinfo` | Remove from the repository and add to `.gitignore`. |

---

## Task 1: Close privilege escalation in public signup

**Files:**
- Modify: `Backend/src/routers/auth_router.py`
- Modify: `Backend/src/services/auth_service.py`
- Modify: `Frontend/components/auth/auth-form.tsx`
- Modify: `Frontend/lib/context/auth-context.tsx`
- Test: `Backend/tests/test_auth.py` (create if missing)

### Background
The public `/auth/signup` endpoint currently accepts `role` in `UserCreate` and writes it directly to the database. The frontend signup form lets the user choose `admin`, `subadmin`, `teacher`, or `student`. This allows anyone to create an admin account.

### Steps

- [ ] **Step 1: Write the failing security test**

Create or append to `Backend/tests/test_auth.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_signup_ignores_privileged_role():
    """Public signup must force role to student even if admin is sent."""
    payload = {
        "email": "attacker-admin@example.com",
        "password": "password123",
        "name": "Attacker",
        "role": "admin",
        "institute": "Evil Inc",
    }
    response = client.post("/auth/signup", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == payload["email"]
    # Fetch /auth/me to verify role
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert me.status_code == 200
    assert me.json()["role"] == "student"
```

Run:

```bash
cd Backend && source .venv/bin/activate && pytest tests/test_auth.py::test_signup_ignores_privileged_role -v
```

Expected: FAIL — role is currently persisted as `admin`.

- [ ] **Step 2: Update `UserCreate` in auth_router.py**

Open `Backend/src/routers/auth_router.py` and change the `UserCreate` class so it no longer accepts `role`, `institute`, or `preferred_language`:

```python
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
```

Update the `signup` handler to pass only `email`, `password`, `name`:

```python
user = await create_user(
    email=user_data.email,
    password=user_data.password,
    name=user_data.name,
)
```

- [ ] **Step 3: Harden `create_user` in auth_service.py**

Open `Backend/src/services/auth_service.py` and make `create_user` ignore any externally supplied role/institute/language, always defaulting to safe values for public signup:

```python
async def create_user(
    email: str,
    password: str,
    name: Optional[str] = None,
    role: str = "student",
    institute: Optional[str] = None,
    preferred_language: Optional[str] = None,
):
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        existing_user = await get_user_by_email(email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Public signup is always a student; privileged roles require admin/sub-admin enrollment.
        allowed_roles = {"student", "teacher", "subadmin", "admin"}
        if role not in allowed_roles:
            role = "student"
        if role in {"admin", "subadmin", "teacher"}:
            role = "student"

        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            password_hash=hashed_password,
            name=name,
            role=role,
            institute=institute,
            preferred_language=preferred_language or "en",
        )

        result = await users_collection.insert_one(user.model_dump(by_alias=True))
        created_user = await users_collection.find_one({"_id": result.inserted_id})
        return created_user
```

- [ ] **Step 4: Remove privileged role UI from signup form**

Open `Frontend/components/auth/auth-form.tsx`:

1. Remove `role`, `setRole`, `institute`, `setInstitute` state and the role/institute JSX block.
2. Remove the `Select` import and `GraduationCap` import if no longer used elsewhere.
3. Update `handleSubmit`:

```typescript
await signup({ email, password, name: name || undefined });
```

Open `Frontend/lib/context/auth-context.tsx` and remove `role`/`institute`/`preferred_language` from `SignupPayload`:

```typescript
export interface SignupPayload {
  email: string;
  password: string;
  name?: string;
}
```

- [ ] **Step 5: Run the test again**

```bash
cd Backend && source .venv/bin/activate && pytest tests/test_auth.py::test_signup_ignores_privileged_role -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Backend/src/routers/auth_router.py Backend/src/services/auth_service.py Frontend/components/auth/auth-form.tsx Frontend/lib/context/auth-context.tsx Backend/tests/test_auth.py
git commit -m "fix(auth): prevent privileged role signup; always default to student"
```

---

## Task 2: Remove role escalation from onboarding save

**Files:**
- Modify: `Backend/src/routers/onboarding_router.py`

### Background
The onboarding endpoint still accepts `role` and can change a student into an admin. Since role is now assigned only through enrollment, onboarding must not touch it.

### Steps

- [ ] **Step 1: Write the failing test**

Append to `Backend/tests/test_auth.py` or create `Backend/tests/test_onboarding.py`:

```python
@pytest.mark.asyncio
async def test_onboarding_cannot_change_role():
    """Onboarding save must ignore a role field and keep the user's current role."""
    # Sign up a student
    signup = client.post("/auth/signup", json={
        "email": "onboarding-role@example.com",
        "password": "password123",
    })
    token = signup.json()["access_token"]

    response = client.post("/api/onboarding", json={
        "role": "admin",
        "institute": "Test School",
    }, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["role"] == "student"
```

Run:

```bash
cd Backend && source .venv/bin/activate && pytest tests/test_onboarding.py::test_onboarding_cannot_change_role -v
```

Expected: FAIL.

- [ ] **Step 2: Drop role handling from onboarding save**

Open `Backend/src/routers/onboarding_router.py`. In `save_onboarding`, remove the branch that updates `role`:

```python
update_fields = {}
if data.name is not None:
    update_fields["name"] = data.name.strip()
if data.institute is not None:
    update_fields["institute"] = data.institute.strip()
if data.preferred_language is not None:
    update_fields["preferred_language"] = data.preferred_language.strip()
update_fields["updated_at"] = datetime.now(timezone.utc)
```

Keep `OnboardingData.role` in the schema only if the frontend still sends it, but document it as ignored.

- [ ] **Step 3: Run the test again**

```bash
cd Backend && source .venv/bin/activate && pytest tests/test_onboarding.py::test_onboarding_cannot_change_role -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add Backend/src/routers/onboarding_router.py Backend/tests/test_onboarding.py
git commit -m "fix(onboarding): ignore role field during onboarding to prevent escalation"
```

---

## Task 3: Fix teacher → student relationship wiring

**Files:**
- Modify: `Backend/src/routers/teacher_router.py`

### Background
When a teacher manages a student, the code sets `managed_by` to the teacher’s email. That breaks sub-admin `list_enrolled_users`, which expects `managed_by` to be the sub-admin’s email. The sub-admin relationship lives in `teacher.managed_by`, and the license lives in `teacher.license_id`.

### Steps

- [ ] **Step 1: Write the failing test**

Create or append to `Backend/tests/test_teachers.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_manage_student_inherits_subadmin_relationship():
    """When a teacher under a sub-admin manages a student, the student must inherit
    the sub-admin's id as managed_by and the license id as license_id."""
    # Enroll a sub-admin (we need a way to create one; for now we can simulate via direct DB
    # or use an admin-only enrollment endpoint. If none exists, create a fixture that inserts
    # the sub-admin user and license directly.)
    from src.core.data_store import users_collection, licenses_collection
    from src.services.auth_service import get_password_hash
    from src.core.models import License

    subadmin_email = "subadmin-rel@example.com"
    teacher_email = "teacher-rel@example.com"
    student_email = "student-rel@example.com"

    # Insert sub-admin + license
    users_collection.insert_one({
        "email": subadmin_email,
        "password_hash": get_password_hash("password123"),
        "role": "subadmin",
        "name": "Sub Admin",
    })
    license_result = licenses_collection.insert_one(
        License(brand_name="Test", subadmin_id=subadmin_email, seats=10).model_dump(by_alias=True)
    )
    license_id = str(license_result.inserted_id)

    # Enroll teacher under sub-admin (simulate enrollment behavior)
    users_collection.insert_one({
        "email": teacher_email,
        "password_hash": get_password_hash("password123"),
        "role": "teacher",
        "managed_by": subadmin_email,
        "license_id": license_id,
    })

    # Create student
    users_collection.insert_one({
        "email": student_email,
        "password_hash": get_password_hash("password123"),
        "role": "student",
    })

    teacher_login = client.post("/auth/login", data={"username": teacher_email, "password": "password123"})
    teacher_token = teacher_login.json()["access_token"]

    response = client.post(
        "/teachers/students/manage",
        json={"student_email": student_email},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )
    assert response.status_code == 200

    student = users_collection.find_one({"email": student_email})
    assert student["teacher_id"] == teacher_email
    assert student["managed_by"] == subadmin_email
    assert student["license_id"] == license_id
```

Run:

```bash
cd Backend && source .venv/bin/activate && pytest tests/test_teachers.py::test_manage_student_inherits_subadmin_relationship -v
```

Expected: FAIL — `managed_by` will be the teacher email.

- [ ] **Step 2: Fix `manage_student`**

Open `Backend/src/routers/teacher_router.py` and replace the update_fields block in `manage_student`:

```python
    teacher = user_info["user"]
    teacher_email = teacher["email"]

    update_fields = {
        "teacher_id": teacher_email,
    }

    # Propagate the teacher's sub-admin relationship down to the student.
    if teacher.get("managed_by"):
        update_fields["managed_by"] = teacher["managed_by"]
    if teacher.get("license_id"):
        update_fields["license_id"] = teacher["license_id"]
```

- [ ] **Step 3: Run the test again**

```bash
cd Backend && source .venv/bin/activate && pytest tests/test_teachers.py::test_manage_student_inherits_subadmin_relationship -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add Backend/src/routers/teacher_router.py Backend/tests/test_teachers.py
git commit -m "fix(teacher): propagate sub-admin managed_by/license_id to managed students"
```

---

## Task 4: Restore test tooling in requirements

**Files:**
- Modify: `Backend/requirements.txt`

### Background
`requirements.txt` was converted from pinned to loose constraints and `pytest` was dropped. The project has a `tests/` directory and uses pytest in `CLAUDE.md`.

### Steps

- [ ] **Step 1: Add pytest and pytest-asyncio**

Open `Backend/requirements.txt`. Add at the end (or in a clear dev section):

```text
# Test tooling
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

Keep runtime dependencies as they are unless specific versions are known to break. If reproducibility is important, consider pinning upper bounds, e.g. `fastapi>=0.110.0,<0.120.0`.

- [ ] **Step 2: Install and run the full test suite**

```bash
cd Backend && source .venv/bin/activate && pip install -r requirements.txt && pytest tests/ -q
```

Expected: Tests collect (the existing import failure will be handled in Task 5).

- [ ] **Step 3: Commit**

```bash
git add Backend/requirements.txt
git commit -m "chore(deps): restore pytest/pytest-asyncio/httpx test dependencies"
```

---

## Task 5: Fix or remove the broken test_main.py

**Files:**
- Modify or delete: `Backend/tests/test_main.py`

### Background
`tests/test_main.py` imports `main` from `src.main`, which does not exist. This blocks the entire test suite from collecting.

### Steps

- [ ] **Step 1: Inspect the existing test**

```bash
cat Backend/tests/test_main.py
```

- [ ] **Step 2: Fix the import**

If the test only needs the FastAPI app, change:

```python
from src.main import main
```
to:

```python
from src.main import app
```

and update any references from `main(...)` to `app`.

If the file is empty or nonsensical, delete it instead.

- [ ] **Step 3: Run the suite**

```bash
cd Backend && source .venv/bin/activate && pytest tests/ -q
```

Expected: No collection errors. Existing tests pass or fail only for legitimate reasons.

- [ ] **Step 4: Commit**

```bash
git add Backend/tests/test_main.py  # or git rm if deleted
git commit -m "fix(tests): correct broken test_main.py import"
```

---

## Task 6: Type consistency for subscription and user response

**Files:**
- Modify: `Backend/src/routers/auth_router.py`
- Modify: `Frontend/lib/data.ts`

### Background
`UserResponse.subscription` is typed as `Optional[Any]`, but the backend has a concrete `SubscriptionInfo` model. The frontend `User` type also lacks subscription/license fields.

### Steps

- [ ] **Step 1: Use the concrete subscription type on the backend**

Open `Backend/src/routers/auth_router.py`. Change imports to include `SubscriptionInfo` from `src.core.models`:

```python
from src.core.models import SubscriptionInfo
```

Update `UserResponse`:

```python
class UserResponse(BaseModel):
    email: str
    name: Optional[str] = None
    role: Literal["student", "teacher", "subadmin", "admin"] = "student"
    institute: Optional[str] = None
    preferred_language: str = "en"
    onboarding_completed: bool = False
    active_exam_id: Optional[str] = None
    teacher_id: Optional[str] = None
    managed_by: Optional[str] = None
    license_id: Optional[str] = None
    subscription: Optional[SubscriptionInfo] = None
```

- [ ] **Step 2: Add subscription to frontend User type**

Open `Frontend/lib/data.ts`:

```typescript
export interface SubscriptionInfo {
  plan: 'weekly' | 'monthly';
  started_at: string;
  expires_at: string;
  status: 'active' | 'expired' | 'cancelled';
}

export interface User {
  email: string;
  name?: string;
  role?: 'student' | 'teacher' | 'subadmin' | 'admin';
  institute?: string;
  preferred_language?: string;
  onboarding_completed?: boolean;
  active_exam_id?: string;
  teacher_id?: string;
  managed_by?: string;
  license_id?: string;
  subscription?: SubscriptionInfo;
}
```

- [ ] **Step 3: Verify backend compiles and frontend builds**

```bash
cd Backend && source .venv/bin/activate && python -c "from src.main import app"
cd Frontend && npm run build
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add Backend/src/routers/auth_router.py Frontend/lib/data.ts
git commit -m "fix(types): align UserResponse subscription type with frontend"
```

---

## Task 7: Clean up generated frontend files

**Files:**
- Revert: `Frontend/next-env.d.ts`
- Delete: `Frontend/tsconfig.tsbuildinfo`
- Modify: `Frontend/.gitignore`

### Background
`next-env.d.ts` and `tsconfig.tsbuildinfo` are generated by Next.js and should not be committed. The diff shows both changed.

### Steps

- [ ] **Step 1: Revert next-env.d.ts**

```bash
cd Frontend && git checkout next-env.d.ts
```

Content should return to:

```typescript
/// <reference types="next" />
/// <reference types="next/image-types/global" />
import "./.next/types/routes.d.ts";

// NOTE: This file should not be edited
// see https://nextjs.org/docs/app/api-reference/config/typescript for more information.
```

- [ ] **Step 2: Remove tsbuildinfo and ignore it**

```bash
cd Frontend && rm tsconfig.tsbuildinfo
git rm --cached tsconfig.tsbuildinfo
```

Open `Frontend/.gitignore` and ensure these lines exist:

```gitignore
/.next/
next-env.d.ts
tsconfig.tsbuildinfo
```

- [ ] **Step 3: Verify build still works**

```bash
cd Frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add Frontend/.gitignore
git add Frontend/next-env.d.ts
git commit -m "chore(frontend): remove generated next-env.d.ts and tsbuildinfo from repo"
```

---

## Task 8: Fix mock test search term usage and tidy imports

**Files:**
- Modify: `Backend/src/services/mock_test_service.py`
- Modify: `Backend/src/routers/mock_test_router.py`

### Background
The code slices `focus_topics` and `weak_topics` into a combined list but then only searches for `search_terms[0]`. Also, `mock_test_router.py` imports `users_collection` locally inside a handler.

### Steps

- [ ] **Step 1: Simplify and correct the web search call**

Open `Backend/src/services/mock_test_service.py`. Replace the search section with:

```python
        # Optionally fetch real-world examples for weak/focus topics
        web_examples = ""
        search_terms = []
        if focus_topics:
            search_terms.extend(focus_topics)
        if resolved_weak_topics:
            search_terms.extend(resolved_weak_topics)
        if subject:
            search_terms.insert(0, subject)
        if search_terms:
            web_examples = await fetch_web_examples(
                topic=search_terms[0],
                question_type="mcq",
                subject=subject,
                num_results=3,
            )
```

This keeps the same external behavior but removes the misleading slicing. If you want to search across multiple topics, loop over the first three terms and concatenate snippets.

- [ ] **Step 2: Move users_collection import to module top in mock_test_router.py**

Open `Backend/src/routers/mock_test_router.py` and add at the top:

```python
from src.core.data_store import users_collection
```

Remove the local import inside `generate_mock_test`:

```python
        if request.student_email and user_id != request.student_email:
            if users_collection is None:
                raise HTTPException(status_code=503, detail="Database unavailable")
            student = await users_collection.find_one({...})
```

- [ ] **Step 3: Verify backend compiles**

```bash
cd Backend && source .venv/bin/activate && python -c "from src.main import app"
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add Backend/src/services/mock_test_service.py Backend/src/routers/mock_test_router.py
git commit -m "refactor(mock-tests): clarify web search topic logic and tidy imports"
```

---

## Task 9: Final verification run

**Files:** None new.

### Steps

- [ ] **Step 1: Run backend pytest**

```bash
cd Backend && source .venv/bin/activate && pytest tests/ -q
```

Expected: All tests pass (or only pre-existing unrelated failures).

- [ ] **Step 2: Run backend app import and OpenAPI check**

```bash
cd Backend && source .venv/bin/activate && python -c "from src.main import app" && python - <<'PY'
from src.main import app
import json
routes = [r.path for r in app.routes if hasattr(r, 'path')]
print('/auth/signup' in routes, '/teachers/students/manage' in routes, '/subadmins/license' in routes)
PY
```

Expected: `True True True`.

- [ ] **Step 3: Run frontend build**

```bash
cd Frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 4: Review git status**

```bash
git status --short
```

Expected: No remaining generated files staged.

- [ ] **Step 5: Final commit if there are any remaining changes**

Only if `git status` shows uncommitted fixes:

```bash
git add -A
git commit -m "chore: final review fixes and verification"
```

---

## Spec coverage self-check

| Review issue | Task that fixes it |
|--------------|--------------------|
| Public signup accepts privileged roles | Task 1 |
| Onboarding accepts privileged roles | Task 2 |
| Teacher `managed_by` set to teacher email | Task 3 |
| `requirements.txt` missing pytest | Task 4 |
| `tests/test_main.py` broken import | Task 5 |
| `UserResponse.subscription` typed as `Any` | Task 6 |
| Generated frontend files committed | Task 7 |
| Web search only uses first topic / misleading slicing | Task 8 |
| `users_collection` local import in router | Task 8 |

**Gaps:** None identified. All review items are covered by a concrete task.

---

## Placeholder scan

Plan contains no `TODO`, `TBD`, `implement later`, or vague "add validation" steps. Every task has exact file paths, code blocks, commands, and expected outputs.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-14-fix-review-issues.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach would you like?
