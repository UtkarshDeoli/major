# Teacher–Student Integration Design

Date: 2026-06-15
Goal: Make the student ↔ teacher loop in Orbit fully functional.

## 1. Scope

In scope:
- Teacher links/unlinks students by email.
- Teacher creates targeted mock tests for themselves or a linked student.
- Student sees assigned tests and personal analytics.
- Teacher sees class-level and per-student analytics.
- Role-aware routing and guards.

Out of scope for this design:
- Admin dashboard.
- Sub-admin licensing enforcement.
- Payments/subscriptions.
- AI web-search agent.

## 2. Core Principle

**Students can exist independently or be linked to a teacher.** Linking is optional. An unlinked student keeps full access to their own materials, tests, and analytics. A linked student additionally receives teacher-assigned tests and appears in the teacher’s analytics.

## 3. Architecture

### Backend (FastAPI)

Reuse existing collections and role fields.

- `users` collection already has `role`, `teacher_id`, `managed_by`, `license_id`.
- `teacher_router.py` already exposes `POST /teachers/students/manage`, `POST /teachers/students/unmanage`, `GET /teachers/students`.
- Add `GET /teachers/analytics` — aggregate analytics over the teacher’s linked students.
- Extend `mock_test_router.py`:
  - `MockTestGenerationRequest.student_email` is already present; wire it so teachers can generate for a selected student.
  - Persist `created_by` (teacher email) and `assigned_to` (student email) on each generated test.
- Extend `analytics_router.py`:
  - `GET /analytics/student` for a student’s own analytics.
  - `GET /analytics/teacher` for a teacher’s aggregate analytics.
- Ensure `auth_router.py` `/auth/me` returns `role`, `teacher_id`, `managed_by`, and `license_id`.

### Frontend (Next.js)

- `/teacher` page already exists. Extend it with:
  - Assign-test flow.
  - Per-student weak-topics summary.
- `/dashboard` student page: add an “Assigned Tests” section.
- `/test` page: add “Assign to student” dropdown (teacher only) and topic/weakness controls.
- Add role-aware default redirect after login/onboarding based on `/auth/me`.

## 4. Data Flow

1. Teacher signs up as `teacher` (admin or existing flow assigns role).
2. Teacher goes to `/teacher` and links a student by email (`POST /teachers/students/manage`).
3. Teacher creates a test at `/test?tab=mock`, selects a linked student, and optionally picks topics or targets weak topics.
4. Backend generates the test and sets `created_by=teacher_email`, `assigned_to=student_email`.
5. Student opens `/dashboard`, sees the assigned test under “Assigned Tests”.
6. Student takes the test at `/test?id=<test_id>` and submits it.
7. Teacher refreshes `/teacher` → Class Analytics tab, sees updated per-student scores and class average.

## 5. Data Model Changes

No new collections. Modify in place:

- `users` — `teacher_id` is optional on students. `managed_by`/`license_id` are reserved for sub-admin propagation.
- `mock_tests` collection entries gain:
  - `created_by: str` (email of creator)
  - `assigned_to: Optional[str]` (email of target student, null if self-generated)
- `mock_test_submissions` entries gain:
  - `student_email: str`
  - `topic_scores: Dict[str, float]`
  - `weak_topics: List[str]`

## 6. Permissions

- `require_role("teacher")` on teacher endpoints.
- A teacher may only manage/view students where `teacher_id == current_user.email`.
- A student may only see tests where `assigned_to == current_user.email` OR `created_by == current_user.email`.
- A teacher can generate a test for a student only if that student is linked to them.
- If a teacher tries to link a non-existent student, return 404 and instruct them to ask the student to sign up first.

## 7. UI Components

### Teacher dashboard (`/teacher`)

- My Students tab
  - Link-student form (email input + button).
  - List of linked students with name, email, tests taken, average score, last active, strengths, weaknesses, unlink action.
- Class Analytics tab
  - Stats cards: total students, active students, tests taken, class average.
  - Bar chart of average score per student.
- CTA: “Create Test” navigates to `/test?tab=mock`.

### Test creation (`/test?tab=mock`, teacher only)

- “Assign to student” dropdown populated from linked students; optional.
- Topic chips derived from selected syllabus/question papers.
- “Target weak topics” toggle (uses prior submission analysis if available).

### Student dashboard (`/dashboard`)

- Existing stats cards.
- New “Assigned Tests” card listing pending teacher-assigned tests with due/created date and CTA to start.

## 8. Error Handling

- 403 if a non-teacher calls teacher endpoints.
- 404 if a teacher tries to manage a student that does not exist.
- 400 if a teacher tries to generate a test for a student they do not manage.
- Frontend shows toast errors for failed link/unlink/test creation.

## 9. Testing

- Backend pytest:
  - `test_manage_existing_student` — link succeeds, `teacher_id` set.
  - `test_manage_nonexistent_student` — 404.
  - `test_unmanage_student` — removes linkage.
  - `test_teacher_analytics` — aggregate stats computed correctly.
  - `test_generate_test_for_managed_student` — success.
  - `test_generate_test_for_unmanaged_student` — 400.
- Frontend manual verification:
  - Teacher links student, creates test, sees it in analytics.
  - Student sees assigned test, submits, score updates for teacher.

## 10. Success Criteria

- A teacher can link a student and create a test for them.
- The student sees the assigned test and can complete it.
- The teacher sees the student’s score and class analytics.
- Role guards prevent cross-role access.
- Existing student self-service features still work when unlinked.
