# Teacher–Student Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the student ↔ teacher loop in Orbit fully functional: teacher links students, creates targeted mock tests for them, and views per-student/class analytics; students see assigned tests and their own analytics.

**Architecture:** Reuse existing `User` role/linkage fields and the existing `teacher_router.py`, `mock_test_router.py`, and `analytics_router.py`. Add `created_by`/`assigned_to` fields to generated mock tests and update the frontend teacher page, student dashboard, and test creation page.

**Tech Stack:** FastAPI + Motor/MongoDB (Backend), Next.js 15 + TypeScript + Tailwind + shadcn/ui (Frontend), pytest (tests).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Backend/src/core/models.py` | Pydantic models for user, mock test request/response. Add `created_by`/`assigned_to` to generated test dict handling. |
| `Backend/src/core/data_store.py` | MongoDB helpers: `store_mock_test`, `get_user_mock_tests`, `store_mock_test_submission`. Update to store/return new fields. |
| `Backend/src/routers/teacher_router.py` | Existing manage/unmanage/list endpoints. Add `GET /teachers/analytics`. |
| `Backend/src/routers/mock_test_router.py` | Existing generate/list/get/submit endpoints. Update generate to persist `created_by`/`assigned_to`; update list/get to return assigned tests to students. |
| `Backend/src/routers/auth_router.py` | `/auth/me` already returns role/linkage. Verify it stays in sync with model changes. |
| `Backend/src/services/mock_test_service.py` | `generate_mock_test_service` needs to accept `created_by`/`assigned_to` and pass them into `MockTestResponse` and storage. |
| `Backend/src/main.py` | Register `teacher_router`, `analytics_router` if not already registered. |
| `Frontend/lib/api.ts` | Add `teacherAPI.getAnalytics` helper. Ensure `mockTestAPI.listMockTests` returns assigned tests. |
| `Frontend/app/(dashboard)/teacher/page.tsx` | Extend existing teacher dashboard: add analytics fetch, per-student weak topics, assign-test CTA. |
| `Frontend/app/(dashboard)/dashboard/page.tsx` | Add “Assigned Tests” section for students. |
| `Frontend/app/(dashboard)/test/page.tsx` | Add “Assign to student” dropdown and topic/weakness controls for teachers. |
| `Frontend/components/auth/route-protection/role-guard.tsx` | Existing guard. Use it on teacher/test pages. |
| `Frontend/lib/context/auth-context.tsx` | Existing context. Ensure role-based default redirect after onboarding/login. |
| `Backend/tests/test_teacher_student.py` | New backend tests for manage/unmanage/analytics/generate-for-student. |

---

### Task 1: Update backend mock test model/storage to record creator and assignee

**Files:**
- Modify: `Backend/src/core/models.py:167-181` (`MockTestResponse`)
- Modify: `Backend/src/core/data_store.py` (find `store_mock_test` and `get_user_mock_tests`)

- [ ] **Step 1: Add fields to `MockTestResponse`**

```python
class MockTestResponse(BaseModel):
    test_id: str
    title: str
    questions: List[MockTestQuestion]
    total_marks: int
    time_limit: int
    created_at: datetime
    user_id: str
    difficulty_level: Optional[str] = "medium"
    latest_submission: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
```

- [ ] **Step 2: Update `store_mock_test` to store the new fields**

In `Backend/src/core/data_store.py`, locate `store_mock_test` and ensure it writes `created_by` and `assigned_to` if present in the incoming dict:

```python
async def store_mock_test(mock_test: dict):
    if mock_tests_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    await mock_tests_collection.insert_one(mock_test)
```

If the helper already inserts the whole dict verbatim, no change is required. If it explicitly constructs a document, add:

```python
"created_by": mock_test.get("created_by"),
"assigned_to": mock_test.get("assigned_to"),
```

- [ ] **Step 3: Update `get_user_mock_tests` to return assigned tests for students**

Find the function and change the query from:

```python
{"user_id": user_id}
```

to:

```python
{"$or": [{"user_id": user_id}, {"assigned_to": user_id}, {"created_by": user_id}]}
```

This lets a student see tests assigned to them even if `user_id` was the teacher who created the test.

- [ ] **Step 4: Verify no failing tests**

Run: `pytest Backend/tests/ -q`
Expected: All existing tests pass (or only pre-existing failures).

- [ ] **Step 5: Commit**

```bash
git add Backend/src/core/models.py Backend/src/core/data_store.py
git commit -m "feat(models): track created_by and assigned_to on mock tests"
```

---

### Task 2: Update mock test service to accept and propagate creator/assignee

**Files:**
- Modify: `Backend/src/services/mock_test_service.py:24-148` (`generate_mock_test_service`)

- [ ] **Step 1: Update the function signature**

```python
async def generate_mock_test_service(
    syllabus_pdf_id: str,
    question_paper_pdf_ids: List[str],
    notes_pdf_id: Optional[str],
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    user_id: str,
    focus_topics: Optional[List[str]] = None,
    weak_topics: Optional[List[str]] = None,
    subject: Optional[str] = None,
    student_email: Optional[str] = None,
    created_by: Optional[str] = None,
    assigned_to: Optional[str] = None,
) -> MockTestResponse:
```

- [ ] **Step 2: Pass the fields into `MockTestResponse`**

Where `mock_test = MockTestResponse(...)` is built, add:

```python
mock_test = MockTestResponse(
    test_id=test_id,
    title=" ".join(title_parts),
    questions=mock_test_data["questions"],
    total_marks=total_marks,
    time_limit=_calculate_time_limit(total_marks, num_mcq, num_text),
    created_at=created_at,
    user_id=user_id,
    difficulty_level=difficulty_level,
    created_by=created_by,
    assigned_to=assigned_to,
)
```

- [ ] **Step 3: Update the store call**

Change:

```python
await store_mock_test(mock_test.dict())
```

to:

```python
await store_mock_test(mock_test.model_dump())
```

(if `dict()` is already used, leave it unless deprecation warnings appear).

- [ ] **Step 4: Run backend tests**

Run: `pytest Backend/tests/ -q`
Expected: Existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add Backend/src/services/mock_test_service.py
git commit -m "feat(mock-test): propagate created_by and assigned_to into stored tests"
```

---

### Task 3: Update mock test router to set creator/assignee and enforce ownership

**Files:**
- Modify: `Backend/src/routers/mock_test_router.py:27-101`

- [ ] **Step 1: In `generate_mock_test`, determine created_by and assigned_to**

After validating inputs and before calling the service, add:

```python
# Determine creator and target
# By default a user generates for themselves.
created_by = user_id
assigned_to = None

if request.student_email and request.student_email != user_id:
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    student = await users_collection.find_one({
        "email": request.student_email,
        "teacher_id": user_id,
    })
    if not student:
        raise HTTPException(status_code=403, detail="You can only target students you manage")
    assigned_to = request.student_email
```

- [ ] **Step 2: Pass creator/assignee to the service**

Update the service call:

```python
mock_test = await generate_mock_test_service(
    syllabus_pdf_id=request.syllabus_pdf_id,
    question_paper_pdf_ids=request.question_paper_pdf_ids,
    notes_pdf_id=request.notes_pdf_id,
    num_mcq=request.num_mcq,
    num_text=request.num_text,
    total_marks=request.total_marks,
    difficulty_level=request.difficulty_level,
    user_id=user_id,
    focus_topics=request.focus_topics,
    weak_topics=request.weak_topics,
    subject=request.subject,
    student_email=request.student_email,
    created_by=created_by,
    assigned_to=assigned_to,
)
```

- [ ] **Step 3: Update `list_mock_tests` to use the new query**

No change is required if `get_user_mock_tests_service` was updated in Task 1. Otherwise, change the service call to pass the new OR query.

- [ ] **Step 4: Update `get_mock_test` and `submit_mock_test` to allow assignees**

In `get_mock_test_service` (`Backend/src/services/mock_test_service.py`), the access check currently compares `test["user_id"] == user_id`. Change it to:

```python
allowed = (
    test.get("user_id") == user_id
    or test.get("created_by") == user_id
    or test.get("assigned_to") == user_id
)
if not allowed:
    return None
```

Do the same for `get_mock_test_service` so submit works for assigned students.

- [ ] **Step 5: Store student email on submission**

In `submit_mock_test`, before calling `analyze_mock_test_submission_service`, add:

```python
# If the test was assigned to someone else, record the actual submitter
submitter_email = user_id
if test.get("assigned_to") and test.get("assigned_to") != user_id:
    submitter_email = test["assigned_to"]
```

Then pass `submitter_email` to the analysis service. If `analyze_mock_test_submission_service` does not accept it, add the parameter and store it in the submission as `user_id` (since submissions are keyed by email).

- [ ] **Step 6: Run tests**

Run: `pytest Backend/tests/ -q`
Expected: Pass.

- [ ] **Step 7: Commit**

```bash
git add Backend/src/routers/mock_test_router.py Backend/src/services/mock_test_service.py
git commit -m "feat(mock-test-router): enforce teacher-student ownership and record assignment"
```

---

### Task 4: Add teacher analytics endpoint

**Files:**
- Modify: `Backend/src/routers/teacher_router.py`

- [ ] **Step 1: Import the analytics response models**

At the top of `teacher_router.py`, add:

```python
from src.routers.analytics_router import TeacherDashboardAnalytics, TeacherStudentAnalytics
from src.core.data_store import mock_test_submissions_collection
```

- [ ] **Step 2: Add `GET /teachers/analytics`**

Append to the file:

```python
@router.get("/analytics", response_model=TeacherDashboardAnalytics)
async def get_teacher_dashboard_analytics(
    user_info: dict = Depends(require_role("teacher"))
):
    """Aggregate analytics for students managed by the logged-in teacher."""
    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    teacher_email = user_info["email"]
    cursor = users_collection.find({"teacher_id": teacher_email})
    students = await cursor.to_list(length=None)

    student_analytics: List[TeacherStudentAnalytics] = []
    total_tests = 0
    class_score_sum = 0.0
    active_count = 0

    for student in students:
        student_email = student.get("email")
        submissions_cursor = mock_test_submissions_collection.find({"user_id": student_email}).sort("created_at", -1)
        submissions = await submissions_cursor.to_list(length=None)

        tests_taken = len(submissions)
        total_tests += tests_taken
        if tests_taken > 0:
            active_count += 1

        student_score_sum = 0.0
        strengths_set: set = set()
        weaknesses_set: set = set()
        last_active: Optional[str] = None

        for sub in submissions:
            score = float(sub.get("total_score", 0))
            max_score = float(sub.get("max_score", 1))
            percentage = (score / max_score) * 100 if max_score > 0 else 0
            student_score_sum += percentage
            class_score_sum += percentage
            strengths_set.update(sub.get("strengths", []) or [])
            weaknesses_set.update(sub.get("improvements", []) or [])
            created_at = sub.get("created_at")
            if created_at:
                last_active = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

        avg_score = student_score_sum / tests_taken if tests_taken > 0 else 0
        student_analytics.append(
            TeacherStudentAnalytics(
                email=student_email,
                name=student.get("name"),
                tests_taken=tests_taken,
                average_score=round(avg_score, 2),
                last_active_at=last_active,
                strengths=list(strengths_set)[:5],
                weaknesses=list(weaknesses_set)[:5],
            )
        )

    class_average = class_score_sum / total_tests if total_tests > 0 else 0

    return TeacherDashboardAnalytics(
        total_students=len(students),
        active_students=active_count,
        total_tests_taken=total_tests,
        class_average=round(class_average, 2),
        student_analytics=student_analytics,
    )
```

- [ ] **Step 3: Ensure routers are registered in `main.py`**

Open `Backend/src/main.py` and verify these lines exist:

```python
from src.routers import teacher_router, analytics_router
...
app.include_router(teacher_router.router)
app.include_router(analytics_router.router)
```

If missing, add them.

- [ ] **Step 4: Run tests**

Run: `pytest Backend/tests/ -q`
Expected: Pass.

- [ ] **Step 5: Commit**

```bash
git add Backend/src/routers/teacher_router.py Backend/src/main.py
git commit -m "feat(teacher): add GET /teachers/analytics endpoint"
```

---

### Task 5: Add backend tests for teacher-student integration

**Files:**
- Create: `Backend/tests/test_teacher_student.py`

- [ ] **Step 1: Write the test file**

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def _token_for(email: str, password: str):
    resp = client.post("/auth/signup", json={"email": email, "password": password, "name": email.split("@")[0]})
    if resp.status_code == 400 and "already registered" in resp.text:
        resp = client.post("/auth/login", data={"username": email, "password": password})
    return resp.json()["access_token"]


def test_teacher_can_manage_existing_student():
    teacher_token = _token_for("teacher1@example.com", "password123")
    student_token = _token_for("student1@example.com", "password123")

    resp = client.post(
        "/teachers/students/manage",
        json={"student_email": "student1@example.com"},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {student_token}"})
    assert me.json()["teacher_id"] == "teacher1@example.com"


def test_teacher_cannot_manage_nonexistent_student():
    teacher_token = _token_for("teacher2@example.com", "password123")

    resp = client.post(
        "/teachers/students/manage",
        json={"student_email": "missing@example.com"},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )
    assert resp.status_code == 404


def test_teacher_can_unmanage_student():
    teacher_token = _token_for("teacher3@example.com", "password123")
    _token_for("student3@example.com", "password123")

    client.post(
        "/teachers/students/manage",
        json={"student_email": "student3@example.com"},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )

    resp = client.post(
        "/teachers/students/unmanage",
        json={"student_email": "student3@example.com"},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )
    assert resp.status_code == 200

    me = client.get("/auth/me", headers={"Authorization": _token_for("student3@example.com", "password123")})
    assert me.json()["teacher_id"] is None


def test_teacher_analytics_returns_managed_students():
    teacher_token = _token_for("teacher4@example.com", "password123")
    _token_for("student4@example.com", "password123")

    client.post(
        "/teachers/students/manage",
        json={"student_email": "student4@example.com"},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )

    resp = client.get(
        "/teachers/analytics",
        headers={"Authorization": f"Bearer {teacher_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_students"] >= 1
    assert any(s["email"] == "student4@example.com" for s in data["student_analytics"])
```

Note: Tests that require PDF generation (`generate_mock_test`) need real PDFs/Gemini and are better verified manually or with mocked services; skip them here to avoid flakiness.

- [ ] **Step 2: Run the new tests**

Run: `pytest Backend/tests/test_teacher_student.py -v`
Expected: All tests pass. If MongoDB is unavailable, mark them with `@pytest.mark.skip`.

- [ ] **Step 3: Commit**

```bash
git add Backend/tests/test_teacher_student.py
git commit -m "test(teacher-student): add integration tests for manage/unmanage/analytics"
```

---

### Task 6: Add frontend API helper for teacher analytics

**Files:**
- Modify: `Frontend/lib/api.ts:325-340`

- [ ] **Step 1: Add `getAnalytics` to `teacherAPI`**

```typescript
export const teacherAPI = {
  manageStudent: async (studentEmail: string) => {
    const response = await api.post('/teachers/students/manage', { student_email: studentEmail });
    return response.data;
  },

  unmanageStudent: async (studentEmail: string) => {
    const response = await api.post('/teachers/students/unmanage', { student_email: studentEmail });
    return response.data;
  },

  listManagedStudents: async () => {
    const response = await api.get('/teachers/students');
    return response.data.students;
  },

  getAnalytics: async () => {
    const response = await api.get('/teachers/analytics');
    return response.data;
  },
};
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd Frontend && npm run lint`
Expected: No new lint errors.

- [ ] **Step 3: Commit**

```bash
git add Frontend/lib/api.ts
git commit -m "feat(api): add teacher analytics helper"
```

---

### Task 7: Extend the teacher dashboard page

**Files:**
- Modify: `Frontend/app/(dashboard)/teacher/page.tsx`

- [ ] **Step 1: Use `teacherAPI.getAnalytics` instead of `analyticsAPI.getTeacherAnalytics`**

Change the fetch in `fetchData`:

```typescript
const [studentsData, analyticsData] = await Promise.all([
  teacherAPI.listManagedStudents(),
  teacherAPI.getAnalytics(),
]);
```

- [ ] **Step 2: Add per-student weak topics rendering**

In the student list card, add:

```tsx
{student.weaknesses && student.weaknesses.length > 0 && (
  <div className="flex flex-wrap gap-1 mt-2">
    {student.weaknesses.map((topic) => (
      <span key={topic} className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full">
        {topic}
      </span>
    ))}
  </div>
)}
```

- [ ] **Step 3: Update assign-test CTA to pass selected student**

Change the Create Test button to navigate with the first selected student if one is chosen, or without:

```typescript
const [selectedStudentEmail, setSelectedStudentEmail] = useState<string | null>(null);
```

Add a student selector above the Create Test button:

```tsx
<select
  className="border rounded px-2 py-1 text-sm"
  value={selectedStudentEmail ?? ""}
  onChange={(e) => setSelectedStudentEmail(e.target.value || null)}
>
  <option value="">Assign to me</option>
  {students.map((s) => (
    <option key={s.email} value={s.email}>{s.name || s.email}</option>
  ))}
</select>
<Button onClick={() => router.push(`/test?tab=mock&student=${selectedStudentEmail ?? ""}`)}>
  <BookOpen className="h-4 w-4 mr-2" />
  Create Test
</Button>
```

- [ ] **Step 4: Run lint and build**

Run:
```bash
cd Frontend && npm run lint
npm run build
```
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add Frontend/app/(dashboard)/teacher/page.tsx
git commit -m "feat(teacher-dashboard): use teacher analytics, show weaknesses, assign-test selector"
```

---

### Task 8: Add assigned tests section to student dashboard

**Files:**
- Modify: `Frontend/app/(dashboard)/dashboard/page.tsx`

- [ ] **Step 1: Load mock tests and filter assigned ones**

At the top, import `mockTestAPI`:

```typescript
import { mockTestAPI } from "@/lib/api";
```

Add state and a fetch effect:

```typescript
const [assignedTests, setAssignedTests] = useState<Array<{ test_id: string; title: string; created_at: string; created_by?: string }>>([]);

useEffect(() => {
  mockTestAPI.listMockTests().then((tests) => {
    const mine = tests.filter((t: any) => t.assigned_to === user?.email);
    setAssignedTests(mine);
  });
}, [user?.email]);
```

- [ ] **Step 2: Render assigned tests card**

Add a new card in the dashboard layout:

```tsx
<Card>
  <CardHeader>
    <CardTitle>Assigned Tests</CardTitle>
  </CardHeader>
  <CardContent>
    {assignedTests.length === 0 ? (
      <p className="text-muted-foreground text-sm">No assigned tests yet.</p>
    ) : (
      <div className="space-y-2">
        {assignedTests.map((test) => (
          <div key={test.test_id} className="flex items-center justify-between">
            <div>
              <p className="font-medium">{test.title}</p>
              <p className="text-xs text-muted-foreground">From {test.created_by}</p>
            </div>
            <Button size="sm" onClick={() => router.push(`/test?id=${test.test_id}`)}>
              Start
            </Button>
          </div>
        ))}
      </div>
    )}
  </CardContent>
</Card>
```

- [ ] **Step 3: Lint and build**

Run:
```bash
cd Frontend && npm run lint
npm run build
```
Expected: Pass.

- [ ] **Step 4: Commit**

```bash
git add Frontend/app/(dashboard)/dashboard/page.tsx
git commit -m "feat(student-dashboard): show assigned tests from teacher"
```

---

### Task 9: Add assign-student and topic controls to test creation page

**Files:**
- Modify: `Frontend/app/(dashboard)/test/page.tsx`

- [ ] **Step 1: Read `student` query param for pre-selection**

```typescript
import { useSearchParams } from "next/navigation";
```

Inside the component:

```typescript
const searchParams = useSearchParams();
const preselectedStudent = searchParams.get("student");
```

- [ ] **Step 2: Fetch linked students for teachers only**

If the test page already has `user` from auth context, add:

```typescript
const [linkedStudents, setLinkedStudents] = useState<Array<{ email: string; name?: string }>>([]);
const [selectedStudent, setSelectedStudent] = useState<string>(preselectedStudent || "");
const [targetWeaknesses, setTargetWeaknesses] = useState(false);
const [selectedTopics, setSelectedTopics] = useState<string[]>([]);

useEffect(() => {
  if (user?.role === "teacher") {
    teacherAPI.listManagedStudents().then(setLinkedStudents);
  }
}, [user?.role]);
```

- [ ] **Step 3: Render controls when user is a teacher**

Where the mock test form is rendered, add:

```tsx
{user?.role === "teacher" && (
  <>
    <div className="space-y-2">
      <label className="text-sm font-medium">Assign to student</label>
      <select
        className="w-full border rounded px-3 py-2 text-sm"
        value={selectedStudent}
        onChange={(e) => setSelectedStudent(e.target.value)}
      >
        <option value="">Myself / No assignment</option>
        {linkedStudents.map((s) => (
          <option key={s.email} value={s.email}>{s.name || s.email}</option>
        ))}
      </select>
    </div>
    <div className="flex items-center gap-2">
      <input
        id="target-weaknesses"
        type="checkbox"
        checked={targetWeaknesses}
        onChange={(e) => setTargetWeaknesses(e.target.checked)}
      />
      <label htmlFor="target-weaknesses" className="text-sm">Target weak topics</label>
    </div>
  </>
)}
```

- [ ] **Step 4: Pass values to `generateMockTest`**

Update the call:

```typescript
const test = await mockTestAPI.generateMockTest(
  syllabusId,
  questionPaperIds,
  notesId,
  numMcq,
  numText,
  totalMarks,
  difficultyLevel,
  selectedTopics.length > 0 ? selectedTopics : undefined,
  targetWeaknesses ? [] : undefined,  // empty array triggers backend weak-topic extraction
  subject,
  selectedStudent || undefined,
);
```

Note: The backend treats a present but empty `weak_topics` list as a signal to extract weak topics when `student_email` is set. Verify this in `mock_test_service.py` lines 78-80.

- [ ] **Step 5: Lint, build, and commit**

Run:
```bash
cd Frontend && npm run lint
npm run build
```
Expected: Pass.

```bash
git add Frontend/app/(dashboard)/test/page.tsx
git commit -m "feat(test-creation): add assign-to-student, topic, and weakness controls"
```

---

### Task 10: Verify role-based routing

**Files:**
- Modify: `Frontend/components/auth/route-protection/role-guard.tsx` (no change if already working)
- Modify: `Frontend/lib/context/auth-context.tsx` or login flow

- [ ] **Step 1: Ensure `/teacher` and `/test` use `RoleGuard`**

Verify `Frontend/app/(dashboard)/teacher/page.tsx` wraps content in:

```tsx
<RoleGuard allowedRoles={["teacher"]}>
  <TeacherDashboardContent />
</RoleGuard>
```

Add the same guard to `Frontend/app/(dashboard)/test/page.tsx` if it should be restricted. Since students also take tests, restrict only the generation tab or the entire page based on existing behavior. If the page supports both roles, skip this step.

- [ ] **Step 2: Add role-aware default redirect**

In the auth context or login handler, after fetching `/auth/me`, redirect:

```typescript
if (user.role === "teacher") router.replace("/teacher");
else router.replace("/dashboard");
```

Implement this in `Frontend/lib/context/auth-context.tsx` inside the login/signup success path, or in `Frontend/app/(dashboard)/layout.tsx`.

- [ ] **Step 3: Test manually**

1. Start backend: `cd Backend && source venv/bin/activate && python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001`
2. Start frontend: `cd Frontend && npm run dev`
3. Sign up a teacher and a student.
4. Teacher: link student, create test assigned to student.
5. Student: log in, see assigned test on `/dashboard`, start and submit it.
6. Teacher: refresh `/teacher`, see updated analytics and weak topics.

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/auth/route-protection/role-guard.tsx Frontend/lib/context/auth-context.tsx Frontend/app/(dashboard)/test/page.tsx
# only add files that actually changed
git commit -m "feat(routing): role-aware redirects and guards for teacher/student flows"
```

---

## Self-Review

- **Spec coverage:**
  - Teacher links/unlinks students → Tasks 1, 4, 5.
  - Teacher creates targeted tests → Tasks 2, 3, 9.
  - Student sees assigned tests → Tasks 1, 8.
  - Teacher sees analytics → Tasks 4, 6, 7.
  - Role guards/routing → Task 10.
- **Placeholder scan:** No TBD/TODO. Every step includes code/commands.
- **Type consistency:** `created_by`/`assigned_to` added to `MockTestResponse`, passed through service, stored in data_store, and queried in `get_user_mock_tests`. Frontend uses `t.assigned_to === user?.email`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-15-teacher-student-integration.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach do you want?
