# Orbit Coaching-Platform Reshape — Phase 3 (Student Class Flows) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the student-side class experience and round out the teacher class-detail page. Students can join classes via enroll code, view their classes, open a class, and see the flashcard decks and mock tests that teachers generated from class materials. Teachers get Tests/Students/Analytics tabs on `/classes/[id]`.

**Architecture:** Extend the P2 backend. Student class membership is tracked by `student_emails` on the `Class` document (P1/P2 already added `add_student_to_class` and `remove_student`). Add endpoints for students to join by enroll code and list their classes. Add endpoints to list class-scoped flashcard decks and mock tests. The frontend creates a `/classes` page for students and updates `/classes/[id]` to be role-aware (teacher sees management tabs; student sees study tabs).

**Tech Stack:** FastAPI, Motor (MongoDB), Pydantic v2, pytest; Next.js 16 App Router, TypeScript, Tailwind, shadcn/ui, axios.

## Global Constraints

- Backend runs from `Backend/` with the venv active: `source venv/bin/activate`. Tests: `pytest tests/ -v`. App: `python -m uvicorn src.main:app --reload --port 8001`.
- Frontend runs from `Frontend/`: `npm run dev` (port 3000), `npm run lint`, `npm run build`. No frontend test runner — frontend tasks are verified with `npm run lint` + `npm run build` + manual browser checks.
- All new Mongo fields are optional with defaults so existing documents keep working — no destructive schema changes.
- Existing collections extended in place, never renamed.
- Roles stay `student | teacher | subadmin | admin`.
- A class is multi-teacher: `Class.teacher_ids` contains the creator + co-teachers.
- Generated flashcard decks and mock tests already carry `class_id`/`class_subject_id`/`created_by` from Phase 2.

## Scope of this plan

Phase 3 covers:
- Student joining classes (enroll code).
- Student listing and viewing their classes.
- Student `/classes/[id]` page showing available study content (flashcards, mock tests).
- Teacher `/classes/[id]` tabs: Subjects, Materials, Tests, Students, Analytics.
- Backend endpoints to list class-scoped content for authorized students.

## File Structure

**Backend — create/modify:**
- `Backend/src/routers/class_router.py` — add `POST /classes/join` (student, by enroll code), `GET /classes/me` (student list), `GET /classes/{id}/content` (student view of decks/tests per subject).
- `Backend/src/core/data_store.py` — add `get_student_classes`, `list_class_decks`, `list_class_mock_tests` (or reuse existing deck/test listers with class filters).
- `Backend/src/services/class_service.py` — add `join_class_by_enroll_code`, `list_student_classes`, `get_class_content_for_student`.

**Frontend — create/modify:**
- `Frontend/app/(dashboard)/classes/page.tsx` — make it role-aware: teachers see existing management list; students see their joined classes + join-by-code input.
- `Frontend/app/(dashboard)/classes/[id]/page.tsx` — make it role-aware: teachers see tabbed management; students see study content (flashcards + mock tests per subject).
- `Frontend/lib/api.ts` — add `classAPI.joinClass`, `classAPI.listMyClasses`, `classAPI.getClassContent`.
- `Frontend/components/dashboard/app-shell.tsx` — add `/classes` to student nav.

**Tests — create:**
- `Backend/tests/test_class_student_flows.py`

---

### Task 1: Backend — student join + list classes

**Files:**
- Modify: `Backend/src/routers/class_router.py`
- Modify: `Backend/src/services/class_service.py` (or create if not present)
- Modify: `Backend/src/core/data_store.py`
- Test: `Backend/tests/test_class_student_flows.py`

**Interfaces:**
- Consumes: `Class` model, `users_collection`, `classes_collection`.
- Produces:
  - `POST /classes/join` (student) → `{class_id, enrolled: True}`
  - `GET /classes/me` (student) → `{classes: [...]}`
  - `GET /classes/{class_id}` now works for students who are enrolled (authorize via `student_emails` in addition to `teacher_ids`).

- [ ] **Step 1: Add data_store helpers**

In `Backend/src/core/data_store.py`, add after `get_teacher_classes`:

```python
async def get_student_classes(student_email: str) -> List[Dict[str, Any]]:
    if classes_collection is None:
        raise Exception("Database connection not available")
    cursor = classes_collection.find({"student_emails": student_email}).sort("created_at", -1)
    classes = await cursor.to_list(length=None)
    return [object_id_to_str(c) for c in classes]
```

- [ ] **Step 2: Add service functions**

Create or update `Backend/src/services/class_service.py`:

```python
"""Class business logic."""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import HTTPException

from src.core.data_store import (
    classes_collection,
    get_class_by_enroll_code,
    get_class_by_id,
    add_student_to_class,
    get_student_classes,
)


async def join_class_by_enroll_code(student_email: str, enroll_code: str) -> dict:
    if classes_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    cls = await get_class_by_enroll_code(enroll_code)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    class_id = cls["id"]
    if student_email in cls.get("student_emails", []):
        return {"class_id": class_id, "enrolled": True}
    updated = await add_student_to_class(class_id, student_email, cls.get("teacher_id"))
    if not updated:
        raise HTTPException(status_code=500, detail="Could not join class")
    return {"class_id": class_id, "enrolled": True}


async def list_student_classes(student_email: str) -> List[dict]:
    classes = await get_student_classes(student_email)
    for c in classes:
        c["student_count"] = len(c.get("student_emails", []))
    return classes


async def get_class_for_user(class_id: str, user_email: str, user_role: str) -> Optional[dict]:
    cls = await get_class_by_id(class_id)
    if not cls:
        return None
    is_teacher = user_email in cls.get("teacher_ids", [cls.get("teacher_id")])
    is_student = user_email in cls.get("student_emails", [])
    if user_role == "teacher" and not is_teacher:
        return None
    if user_role == "student" and not (is_teacher or is_student):
        return None
    return cls
```

- [ ] **Step 3: Update class_router**

In `Backend/src/routers/class_router.py`:

Add request model near `ClassCreateRequest`:
```python
class JoinClassRequest(BaseModel):
    enroll_code: str
```

Update `get_class_detail` authorization to allow enrolled students and class teachers:
```python
@router.get("/{class_id}", response_model=ClassDetail)
async def get_class_detail(
    class_id: str = Path(...),
    user=Depends(require_role()),  # any authenticated user
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    user_email = user["email"]
    user_role = user.get("role")
    is_teacher = user_email in cls.get("teacher_ids", [cls.get("teacher_id")])
    is_student = user_email in cls.get("student_emails", [])
    if not is_teacher and not is_student:
        raise HTTPException(status_code=403, detail="Not authorized to view this class")
    # ... existing detail construction ...
```

Add endpoints:
```python
@router.post("/join", status_code=status.HTTP_200_OK)
async def join_class(
    request: JoinClassRequest,
    student=Depends(require_role("student")),
):
    result = await class_service.join_class_by_enroll_code(student["email"], request.enroll_code)
    return result


@router.get("/me", response_model=ClassListResponse)
async def list_my_classes(
    student=Depends(require_role("student")),
):
    classes = await class_service.list_student_classes(student["email"])
    return ClassListResponse(classes=[ClassSummary(**c) for c in classes])
```

- [ ] **Step 4: Write failing tests**

Create `Backend/tests/test_class_student_flows.py`:

```python
import pytest
from bson import ObjectId

import src.core.data_store as ds
from src.core.security import get_current_user_with_role
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


def _set_auth(role, email):
    def _auth():
        return {"email": email, "user": {"email": email, "role": role}}
    app.dependency_overrides[get_current_user_with_role] = _auth


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl()
    classes = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "classes_collection", classes)

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
```

- [ ] **Step 5: Run tests to verify they fail, then implement, then verify pass**

Run: `cd Backend && source venv/bin/activate && pytest tests/test_class_student_flows.py -v`
Expected after fix: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add Backend/src/routers/class_router.py Backend/src/services/class_service.py Backend/src/core/data_store.py Backend/tests/test_class_student_flows.py
git commit -m "feat(class): student join class, list classes, view enrolled class"
```

---

### Task 2: Backend — class-scoped study content for students

**Files:**
- Modify: `Backend/src/routers/class_router.py` or create `Backend/src/routers/class_content_router.py`
- Modify: `Backend/src/core/data_store.py`
- Test: extend `Backend/tests/test_class_student_flows.py`

**Interfaces:**
- Consumes: `flashcard_decks_collection`, `mock_tests_collection` (already have class fields from P2).
- Produces: `GET /classes/{class_id}/content` (student or teacher) → `{subjects: [...], decks: [...], tests: [...]}` scoped to the class.

- [ ] **Step 1: Add data_store query helpers**

In `Backend/src/core/data_store.py`:

```python
async def list_decks_by_class(class_id: str) -> List[Dict[str, Any]]:
    if flashcard_decks_collection is None:
        raise Exception("Database connection not available")
    cursor = flashcard_decks_collection.find({"class_id": class_id}).sort("created_at", -1)
    decks = await cursor.to_list(length=None)
    return [object_id_to_str(d) for d in decks]


async def list_mock_tests_by_class(class_id: str) -> List[Dict[str, Any]]:
    if mock_tests_collection is None:
        raise Exception("Database connection not available")
    cursor = mock_tests_collection.find({"class_id": class_id}).sort("created_at", -1)
    tests = await cursor.to_list(length=None)
    return [object_id_to_str(t) for t in tests]
```

- [ ] **Step 2: Add service aggregator**

In `Backend/src/services/class_service.py`:

```python
async def get_class_study_content(class_id: str, user_email: str) -> dict:
    cls = await get_class_for_user(class_id, user_email, "student")
    if not cls:
        raise HTTPException(status_code=403, detail="Not authorized to view this class")
    from src.core.data_store import list_class_subjects, list_decks_by_class, list_mock_tests_by_class
    subjects = await list_class_subjects(class_id)
    decks = await list_decks_by_class(class_id)
    tests = await list_mock_tests_by_class(class_id)
    return {
        "class_id": class_id,
        "subjects": [{"id": s.pop("_id", None), **s} for s in subjects],
        "decks": decks,
        "tests": tests,
    }
```

- [ ] **Step 3: Add router endpoint**

In `Backend/src/routers/class_router.py`:

```python
@router.get("/{class_id}/content")
async def get_class_content(
    class_id: str = Path(...),
    user=Depends(require_role()),
):
    result = await class_service.get_class_study_content(class_id, user["email"])
    return result
```

- [ ] **Step 4: Extend tests**

Add to `Backend/tests/test_class_student_flows.py`:

```python
def test_student_gets_class_study_content(setup, monkeypatch):
    from datetime import datetime, timezone
    c = setup["client"]
    cid = setup["class_id"]
    c.post("/classes/join", json={"enroll_code": "JEE123"})

    # stub class content collections
    decks = _FakeColl()
    decks.docs["1"] = {"_id": str(ObjectId()), "class_id": cid, "title": "F1", "card_count": 5, "created_at": datetime.now(timezone.utc)}
    tests = _FakeColl()
    tests.docs["1"] = {"_id": str(ObjectId()), "class_id": cid, "title": "T1", "total_marks": 30, "created_at": datetime.now(timezone.utc)}
    monkeypatch.setattr(ds, "flashcard_decks_collection", decks)
    monkeypatch.setattr(ds, "mock_tests_collection", tests)

    r = c.get(f"/classes/{cid}/content")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["class_id"] == cid
    assert len(data["decks"]) == 1
    assert len(data["tests"]) == 1
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_class_student_flows.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Backend/src/routers/class_router.py Backend/src/services/class_service.py Backend/src/core/data_store.py Backend/tests/test_class_student_flows.py
git commit -m "feat(class): class-scoped study content endpoint for students"
```

---

### Task 3: Frontend API helpers for student class flows

**Files:**
- Modify: `Frontend/lib/api.ts`

**Interfaces:**
- Consumes: new backend endpoints from Tasks 1–2.
- Produces: `classAPI.joinClass`, `classAPI.listMyClasses`, `classAPI.getClassContent`.

- [ ] **Step 1: Add student methods to classAPI**

In `Frontend/lib/api.ts`, inside `classAPI`, add:

```ts
  async joinClass(enrollCode: string): Promise<{ class_id: string; enrolled: boolean }> {
    const res = await api.post("/classes/join", { enroll_code: enrollCode });
    return res.data;
  },
  async listMyClasses(): Promise<{ classes: any[] }> {
    const res = await api.get("/classes/me");
    return res.data;
  },
  async getClassContent(classId: string): Promise<{ class_id: string; subjects: any[]; decks: any[]; tests: any[] }> {
    const res = await api.get(`/classes/${classId}/content`);
    return res.data;
  },
```

- [ ] **Step 2: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add Frontend/lib/api.ts
git commit -m "feat(frontend): student class join/list/content API helpers"
```

---

### Task 4: Frontend — role-aware `/classes` page for students

**Files:**
- Modify: `Frontend/app/(dashboard)/classes/page.tsx`

**Interfaces:**
- Consumes: `user.role` from `useAuth`, `classAPI.listClasses` (teacher), `classAPI.listMyClasses` and `classAPI.joinClass` (student).
- Produces: Teachers see the existing management page. Students see a list of joined classes + an enroll-code input to join a new class.

- [ ] **Step 1: Make the page role-aware**

Update `Frontend/app/(dashboard)/classes/page.tsx`:

- Import `useAuth`.
- For `teacher`: keep existing behavior (RoleGuard + class list + create dialog).
- For `student`: render a student view with:
  - Input for enroll code + Join button.
  - List of joined classes with Open links to `/classes/{id}`.

```tsx
import { useAuth } from "@/lib/context/auth-context"
```

Replace the single `RoleGuard allowedRoles={["teacher"]}` wrapper with conditional rendering:

```tsx
export default function ClassesPage() {
  const { user } = useAuth()
  if (!user) return <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
  return user.role === "teacher" ? <TeacherClassesView /> : <StudentClassesView />
}
```

Extract the existing page body into `TeacherClassesView` (same implementation), and add `StudentClassesView`:

```tsx
function StudentClassesView() {
  const { toast } = useToast()
  const [classes, setClasses] = useState<ClassItem[]>([])
  const [loading, setLoading] = useState(true)
  const [code, setCode] = useState("")
  const [joining, setJoining] = useState(false)

  const fetchClasses = useCallback(async () => {
    setLoading(true)
    try {
      const res = await classAPI.listMyClasses()
      setClasses((res.classes || []) as ClassItem[])
    } catch (e) {
      toast({ title: "Couldn't load classes", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { fetchClasses() }, [fetchClasses])

  const handleJoin = async () => {
    if (!code.trim()) return
    setJoining(true)
    try {
      await classAPI.joinClass(code.trim())
      setCode("")
      fetchClasses()
      toast({ title: "Joined class" })
    } catch (e) {
      toast({ title: "Couldn't join class", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setJoining(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold">My Classes</h1>
        <p className="text-sm text-muted-foreground">Join a class with an enroll code and view your study content.</p>
      </div>
      <div className="flex gap-2 max-w-md">
        <Input value={code} onChange={(e) => setCode(e.target.value)} placeholder="Enter enroll code" />
        <Button onClick={handleJoin} disabled={!code.trim() || joining}>{joining ? <Loader2 className="h-4 w-4 animate-spin" /> : "Join"}</Button>
      </div>
      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      ) : classes.length === 0 ? (
        <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">You haven't joined any classes yet. Enter an enroll code to get started.</div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {classes.map((c) => (
            <div key={c.id} className="rounded-lg border bg-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">{c.name}</h3>
                <span className="text-xs text-muted-foreground flex items-center gap-1"><Users className="h-3 w-3" />{c.student_count}</span>
              </div>
              {c.description && <p className="text-xs text-muted-foreground">{c.description}</p>}
              <Button asChild variant="outline" className="w-full"><Link href={`/classes/${c.id}`}>Open class</Link></Button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/(dashboard)/classes/page.tsx
git commit -m "feat(frontend): role-aware /classes page with student join flow"
```

---

### Task 5: Frontend — role-aware `/classes/[id]` page

**Files:**
- Modify: `Frontend/app/(dashboard)/classes/[id]/page.tsx`
- Modify: `Frontend/components/dashboard/app-shell.tsx` (add `/classes` to student nav)

**Interfaces:**
- Consumes: `user.role`, `classAPI.getClassContent` (student), existing subject/material APIs (teacher).
- Produces: Teacher sees tabbed management (Subjects, Materials, Tests, Students, Analytics). Student sees study tabs (Subjects, Flashcards, Mock Tests).

- [ ] **Step 1: Add Tabs component imports**

Use shadcn/ui Tabs. If not installed, install it:
```bash
cd Frontend && npx shadcn@latest add tabs
```

Import in the page:
```tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
```

- [ ] **Step 2: Make the page role-aware**

Update `Frontend/app/(dashboard)/classes/[id]/page.tsx`:

```tsx
import { useAuth } from "@/lib/context/auth-context"
```

Replace the single `RoleGuard allowedRoles={["teacher"]}` wrapper with conditional rendering. Keep teacher logic as-is but wrap in Tabs. Add a `StudentClassDetailPage`.

Teacher view structure:
```tsx
function TeacherClassDetailPage({ id }: { id: string }) {
  // existing state + loadSubjects/Materials, plus load tests/students/analytics placeholders
  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold">{cls?.name}</h1>
        {cls?.description && <p className="text-sm text-muted-foreground">{cls.description}</p>}
      </div>
      <Tabs defaultValue="subjects">
        <TabsList>
          <TabsTrigger value="subjects">Subjects & Materials</TabsTrigger>
          <TabsTrigger value="tests">Tests</TabsTrigger>
          <TabsTrigger value="students">Students</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>
        <TabsContent value="subjects">{/* existing subjects + materials grid */}</TabsContent>
        <TabsContent value="tests"><div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Tests tab — list class mock tests here.</div></TabsContent>
        <TabsContent value="students"><div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Students tab — roster + remove here.</div></TabsContent>
        <TabsContent value="analytics"><div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Analytics tab — class performance here.</div></TabsContent>
      </Tabs>
    </div>
  )
}
```

Student view structure:
```tsx
function StudentClassDetailPage({ id }: { id: string }) {
  const { toast } = useToast()
  const [content, setContent] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [activeSubject, setActiveSubject] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    classAPI.getClassContent(id)
      .then((c) => { setContent(c); setActiveSubject(c.subjects?.[0]?.id ?? null) })
      .catch((e) => toast({ title: "Couldn't load class", description: getErrorMessage(e), variant: "destructive" }))
      .finally(() => setLoading(false))
  }, [id, toast])

  const decksForSubject = (subjectId: string) => (content?.decks || []).filter((d: any) => d.class_subject_id === subjectId)
  const testsForSubject = (subjectId: string) => (content?.tests || []).filter((t: any) => t.class_subject_id === subjectId)

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      ) : (
        <>
          <div>
            <h1 className="text-xl font-semibold">{content?.class_name || "Class"}</h1>
          </div>
          <Tabs defaultValue="subjects">
            <TabsList>
              <TabsTrigger value="subjects">Subjects</TabsTrigger>
              <TabsTrigger value="flashcards">Flashcards</TabsTrigger>
              <TabsTrigger value="mock-tests">Mock Tests</TabsTrigger>
            </TabsList>
            <TabsContent value="subjects">
              <div className="grid gap-6 lg:grid-cols-[240px,1fr]">
                <div className="space-y-4">
                  <h2 className="text-sm font-semibold">Subjects</h2>
                  <div className="space-y-1">
                    {content?.subjects?.map((s: any) => (
                      <button
                        key={s.id}
                        onClick={() => setActiveSubject(s.id)}
                        className={`w-full text-left rounded-md px-3 py-2 text-sm ${activeSubject === s.id ? "bg-secondary text-foreground" : "hover:bg-muted/50 text-muted-foreground"}`}
                      >{s.name}</button>
                    ))}
                  </div>
                </div>
                <div className="space-y-4">
                  <h2 className="text-sm font-semibold">Study content</h2>
                  {!activeSubject ? (
                    <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Select a subject.</div>
                  ) : (
                    <div className="space-y-4">
                      <div>
                        <h3 className="text-xs font-semibold uppercase text-muted-foreground">Flashcards</h3>
                        {decksForSubject(activeSubject).length === 0 ? (
                          <p className="text-sm text-muted-foreground">No flashcards yet.</p>
                        ) : (
                          <div className="grid gap-2">
                            {decksForSubject(activeSubject).map((d: any) => (
                              <Link key={d.id} href={`/flashcards/${d.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                                <div className="text-sm font-medium">{d.title}</div>
                                <div className="text-xs text-muted-foreground">{d.card_count} cards</div>
                              </Link>
                            ))}
                          </div>
                        )}
                      </div>
                      <div>
                        <h3 className="text-xs font-semibold uppercase text-muted-foreground">Mock Tests</h3>
                        {testsForSubject(activeSubject).length === 0 ? (
                          <p className="text-sm text-muted-foreground">No mock tests yet.</p>
                        ) : (
                          <div className="grid gap-2">
                            {testsForSubject(activeSubject).map((t: any) => (
                              <Link key={t.test_id || t.id} href={`/mock-tests/${t.test_id || t.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                                <div className="text-sm font-medium">{t.title}</div>
                                <div className="text-xs text-muted-foreground">{t.total_marks} marks</div>
                              </Link>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>
            <TabsContent value="flashcards">
              <div className="grid gap-2">
                {content?.decks?.length === 0 ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No flashcards available yet.</div>
                ) : (
                  content.decks.map((d: any) => (
                    <Link key={d.id} href={`/flashcards/${d.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                      <div className="text-sm font-medium">{d.title}</div>
                      <div className="text-xs text-muted-foreground">{d.card_count} cards</div>
                    </Link>
                  ))
                )}
              </div>
            </TabsContent>
            <TabsContent value="mock-tests">
              <div className="grid gap-2">
                {content?.tests?.length === 0 ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No mock tests available yet.</div>
                ) : (
                  content.tests.map((t: any) => (
                    <Link key={t.test_id || t.id} href={`/mock-tests/${t.test_id || t.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                      <div className="text-sm font-medium">{t.title}</div>
                      <div className="text-xs text-muted-foreground">{t.total_marks} marks</div>
                    </Link>
                  ))
                )}
              </div>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  )
}
```

Default export:
```tsx
export default function ClassDetailPage() {
  const { id } = useParams() as { id: string }
  const { user } = useAuth()
  if (!user) return <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
  return user.role === "teacher" ? <TeacherClassDetailPage id={id} /> : <StudentClassDetailPage id={id} />
}
```

- [ ] **Step 3: Add `/classes` to student sidebar**

In `Frontend/components/dashboard/app-shell.tsx`, add `{ href: "/classes", label: "Classes", icon: Users }` to `studentNav` (after Dashboard).

- [ ] **Step 4: Lint + build**

Run: `cd Frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add Frontend/app/(dashboard)/classes/[id]/page.tsx Frontend/components/dashboard/app-shell.tsx
git commit -m "feat(frontend): role-aware /classes/[id] with teacher tabs and student study view"
```

---

### Task 6: Teacher `/classes/[id]` Tests/Students/Analytics tabs (real data)

**Files:**
- Modify: `Backend/src/routers/class_router.py`
- Modify: `Backend/src/services/class_service.py`
- Modify: `Frontend/app/(dashboard)/classes/[id]/page.tsx`

**Interfaces:**
- Teacher can see class roster (`GET /classes/{id}/students`) and generated tests (`GET /classes/{id}/tests`). Analytics is a placeholder or uses existing teacher-analytics endpoints.

- [ ] **Step 1: Add roster endpoint**

In `Backend/src/routers/class_router.py`:

```python
@router.get("/{class_id}/students")
async def get_class_students(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher["email"] not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized")
    from src.core.data_store import users_collection
    students = []
    if users_collection is not None:
        cursor = users_collection.find({"email": {"$in": cls.get("student_emails", [])}})
        students = await cursor.to_list(length=None)
    return {"students": [object_id_to_str(u) for u in students]}
```

- [ ] **Step 2: Add class tests endpoint**

In `Backend/src/routers/class_router.py`:

```python
@router.get("/{class_id}/tests")
async def get_class_tests(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher["email"] not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized")
    from src.core.data_store import list_mock_tests_by_class
    tests = await list_mock_tests_by_class(class_id)
    return {"tests": tests}
```

- [ ] **Step 3: Wire frontend teacher tabs to real data**

In `Frontend/app/(dashboard)/classes/[id]/page.tsx`, add state/effects for `students` and `tests` in `TeacherClassDetailPage` and render them in the respective tabs.

- [ ] **Step 4: Tests**

Extend `Backend/tests/test_class_student_flows.py` with a test for `GET /classes/{id}/students` and `GET /classes/{id}/tests`.

- [ ] **Step 5: Commit**

```bash
git add Backend/src/routers/class_router.py Backend/src/services/class_service.py Frontend/app/(dashboard)/classes/[id]/page.tsx Backend/tests/test_class_student_flows.py
git commit -m "feat(class): teacher tests and roster tabs"
```

---

### Task 7: Final verification + review + merge

- [ ] **Step 1: Run all Phase 3 + Phase 2 backend tests**

```bash
cd Backend && source venv/bin/activate
pytest tests/test_class_router_p2.py tests/test_class_subjects.py tests/test_class_materials.py tests/test_class_material_generation.py tests/test_class_student_flows.py -v
```
Expected: all pass.

- [ ] **Step 2: Run frontend lint + build**

```bash
cd Frontend && npm run lint && npm run build
```
Expected: 0 errors.

- [ ] **Step 3: Final review**

Generate review package and run final review.

- [ ] **Step 4: Merge and push**

```bash
git checkout master
git merge feat/coaching-p3-student-class-flows --no-ff -m "Merge Phase 3: student class flows"
git push origin master
```

---

## Phase 3 completion checklist

- [ ] Student can join class by enroll code.
- [ ] Student can list their classes (`GET /classes/me`).
- [ ] Student can view enrolled class and see class-scoped flashcards/mock tests.
- [ ] Teacher `/classes/[id]` has Subjects, Materials, Tests, Students, Analytics tabs.
- [ ] Backend tests pass.
- [ ] Frontend lint + build pass.
- [ ] Final review approved and branch merged.

## Non-blocking follow-ups for later phases

- Real-time enroll-code validation / copy-to-join.
- Student progress tracking per class.
- Rich analytics on the Analytics tab.
- Assign specific decks/tests to subsets of students.
