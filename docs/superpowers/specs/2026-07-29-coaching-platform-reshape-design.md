# Orbit Coaching-Platform Reshape — Design Spec

**Date:** 2026-07-29
**Status:** Approved (pending user spec review)
**Scope:** Reshape Orbit from a per-user study tool into a coaching-institute platform: coaching-admin monitoring, teacher class/subject/material/test management with AI generation, independent students who gain class + org branding when enrolled, coaching logo/branding, and an AI-chat fix.

---

## 1. Goals & non-goals

### Goals
1. **Coaching admin (monitoring only):** an admin account per coaching institute that manages its teachers and students and monitors classes — no content creation.
2. **Teacher feature set:** create classes; add multiple subjects per class (multi-teacher per class); upload materials; generate flashcards/mock tests from materials via AI; create mock tests (immediate AI feedback) and **actual tests** (scheduled, strictly timed, no student feedback); see all student attempts/marks. Keep existing student-monitoring features; remove student-only features (focus mode, study planner/timer) from the teacher experience.
3. **Independent + enrolled students:** students sign up and operate freely with no organization. When a teacher adds them to a class, they gain visibility of that class and the coaching (org) they belong to, including branding.
4. **Coaching branding:** an API to capture coaching basic info + logo at org creation; logo visible to teachers in their dashboard at all times and to students in the classes section and home dashboard.
5. **AI chat fix:** make chat functional — restore conversational memory and allow free (no-material) chat.

### Non-goals
- Rewriting the existing billing/Razorpay, auth/JWT, or RAG (ChromaDB + BM25 + RRF) pipelines. These are extended, not replaced.
- Removing the platform-level super-admin (`admin` role) — it stays for the platform operator.
- Removing the personal study hierarchy (Exam → Subject → Collection → Material) — it stays for independent/unenrolled students.
- Real-time proctoring or anti-cheat for actual tests (out of scope; actual tests are timed and feedback-withheld only).

---

## 2. Key decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Coaching admin role | Reuse existing `subadmin` (org owner) | Org/billing/seat machinery already built; keep `admin` as platform super-admin |
| Class subjects & materials | New `ClassSubject` under Class; materials/tests link to `(class_id, class_subject_id)` | Clean separation of class-scoped vs personal study; reuses RAG via `doc_id` |
| Class ownership | Multi-teacher per class (creator + co-teachers, same org) | Matches "if any other teacher adds a subject, the student sees 3 subjects" |
| Actual test behavior | `start_at`/`end_at` window, timed, **no student feedback**; teacher sees all marks/attempts | Matches description; no publish-to-student step |
| Build order | P1 Foundation → P2 Teacher → P3 Tests → P4 Student → P5 Chat | Each phase ships verifiable end-to-end value |

---

## 3. Architecture approach

Extend, don't rebuild. The platform already has: JWT auth with roles, an `organizations` concept with billing/seats/invites, a teacher `Class` with enroll codes, a RAG pipeline (ChromaDB + BM25 + RRF) keyed by `doc_id`, and a Gemini service for mock-test/flashcard/summary generation. The reshape adds a **class → subject → material/test graph** alongside the existing personal hierarchy, extends the test model with an `actual` mode, adds org branding, and fixes chat memory.

### Phase decomposition
- **P1 — Foundation:** roles/org/logo branding API + data-model field additions + `curriculum` field.
- **P2 — Teacher class/subject/material + AI generation:** class pages, `ClassSubject`, class-scoped material upload + RAG, generate flashcards/mock from a class material.
- **P3 — Actual tests + scheduling:** `mode=actual` tests with `start_at`/`end_at`, timed runner, no student feedback, teacher attempts/marks view.
- **P4 — Student enrolled experience + branding display:** coaching banner/logo on home + classes section, class detail (subjects/materials/tests), org auto-enroll on class join.
- **P5 — AI chat fix:** thread conversation history into Gemini + free (no-material) chat + visible session history.

---

## 4. Data model (backend, MongoDB)

Collections are managed via `Backend/src/core/data_store.py`. New collections follow the existing pattern. Field names are concrete.

### 4.1 `organizations` (extend existing)
Existing: `org_id`, `name`, `brand_name`, `owner_user_id`, `tier`, `seats_total`, `seats_used`, `status`, `billing_cycle`, `created_at`, `updated_at`.
**Add:**
- `logo_url: str | None` — public URL/path served from disk.
- `logo_file_path: str | None` — disk path under `uploads/orgs/{org_id}/`.
- `tagline: str | None`
- `primary_color: str | None` (optional, future theming)

### 4.2 `users` (extend existing)
Existing: `email`, `name`, `role`, `org_id`, `member_role`, `class_ids`, `teacher_ids`, `institute`, `preferred_language`, `onboarding_completed`, `active_exam_id`, `license_id`, `subscription`.
**Add:**
- `curriculum: str | None` — exam-preset key (e.g. `jee-mains`) captured at signup/onboarding for independent students.

### 4.3 `classes` (extend existing `Class`)
Existing: `id`, `teacher_id`, `name`, `description`, `exam_preset`, `enroll_code`, `student_emails`, `created_at`, `updated_at`.
**Add:**
- `org_id: str` — the coaching this class belongs to.
- `teacher_ids: List[str]` — creator + co-teachers; all must be `member_role=teacher` in `org_id`.
- `subject_ids: List[str]` — references to `class_subjects`.
- `teacher_id` kept for backward compatibility as `teacher_ids[0]`.

### 4.4 `class_subjects` (new collection)
- `id: str`
- `class_id: str`
- `name: str`
- `icon: str | None`
- `created_by: str` (teacher email)
- `created_at`, `updated_at`

Any teacher in the class's org may add a subject. All students enrolled in the class see every subject.

### 4.5 `class_materials` (new collection)
- `id: str`
- `class_id: str`
- `class_subject_id: str`
- `teacher_id: str` (uploader)
- `name: str`
- `type: "pdf" | "image" | "text"`
- `size: int`
- `page_count: int | None`
- `doc_id: str` — links into existing `pdfs_collection` + ChromaDB RAG pipeline (reuses `pdf_service`, `document_processor`, `vector_store`).
- `rag_indexed: bool`
- `created_at`, `updated_at`

AI generation from a class material works identically to personal materials because retrieval is keyed by `doc_id`.

### 4.6 `tests` (extend existing mock-test model)
Existing mock-test fields: `test_id`, `title`, `questions`, `total_marks`, `time_limit`, `user_id`, `created_by`, `assigned_to`, `difficulty_level`, `subject`, `grading_mode`, `status`, `latest_submission`.
**Add:**
- `mode: "mock" | "actual"` (default `"mock"`)
- `class_id: str | None` — when class-scoped.
- `class_subject_id: str | None`
- `created_by: str` (teacher email)
- For `mode="actual"` only:
  - `start_at: datetime` — window open.
  - `end_at: datetime` — window close.
  - `duration_minutes: int` — timed duration within the window.
  - `results_published: bool` — internal flag (default `false`); not used to reveal results to students (per decision, students never see actual-test feedback), but available for teacher workflow/analytics.

For class tests, `assigned_to` is derived from `class.student_emails` (class-wide). Mock = immediate AI feedback (unchanged). Actual = no feedback to student; teacher sees all.

### 4.7 `test_submissions` (extend existing)
Existing: `submission_id`, `test_id`, `user_id`, `answers`, `time_taken`, `total_score`, `max_score`, `percentage`, `feedback_summary`, `question_feedback`, `strengths`, `improvements`, `study_recommendations`, `grading_mode`, `status`, `subject`.
**Add:**
- `test_mode: "mock" | "actual"`

Student-facing endpoints strip score + feedback when `test_mode="actual"`. Teacher-facing endpoints return everything.

### 4.8 `class_invites` (new collection)
For email-add when the student does not yet have an account.
- `id: str`
- `class_id: str`
- `email: str`
- `token: str`
- `status: "pending" | "used"`
- `created_by: str` (teacher email)
- `created_at`, `used_at`

When that email later signs up, the signup flow auto-enrolls the student into the class (and the class's org). The teacher immediately receives a copyable invite link. Email delivery is best-effort: if SMTP is configured, send the link by email; otherwise the teacher shares the link manually. Link-sharing is the default mechanism.

---

## 5. Roles, auth & coaching admin

### 5.1 Role definitions
- **`subadmin` = coaching admin.** Creates the coaching (org) via checkout, uploads logo/branding, invites teachers into the org (existing org invite with `member_role="teacher"`), buys seats. **Monitoring only — no content creation.**
- **`admin` = platform super-admin (unchanged).** Cross-coaching oversight, billing, manual subscriptions, analytics via `/admin`.
- **`teacher`.** Added to an org by the coaching admin. Creates classes, adds subjects, uploads/generates materials, creates mock + actual tests, monitors students.
- **`student`.** Signs up independently (public signup forced to `student`, unchanged). Operates freely with no org. When added to a class by a teacher, gains `org_id` and sees coaching branding + that class.

### 5.2 Coaching-admin dashboard (`/org`, extended)
Existing `/org` shows org card, members, invites, seats. **Extend with monitoring-only tabs:**
- **Overview:** seats, tier, logo, branding.
- **Teachers:** list (add via invite, remove). Read-only on their content.
- **Students:** all students across all classes in the coaching (read-only).
- **Classes:** every class in the coaching (read-only): teacher(s), subject count, student count, test count. No edit access to materials/tests.

### 5.3 Teacher navigation
Remove student-only items from the teacher sidebar: **Focus**, **Plans** (study planner/timer). Keep: Dashboard, Classes, Materials, Tests, Analytics, Chat, Settings. Gating is enforced both client-side (role-based nav in `app-shell.tsx`) and server-side (`require_role` + class/org ownership checks).

### 5.4 Enrollment flows
- **Invite link (class):** existing `enroll_code` → copyable link; student previews and enrolls. Enrolling in a class whose `org_id` is set enrolls the student into that org (`org_id`, `member_role="student"`, `org_joined_at`).
- **Email-add (account exists):** teacher enters a student email; if an account exists, it is added to `class.student_emails` and the user's `class_ids` (and org enrollment if applicable).
- **Email-add (no account):** creates a `class_invites` record and returns a copyable link; auto-enroll on future signup with that email.

---

## 6. Teacher feature set (core reshape)

### 6.1 New frontend pages
- **`/classes` (teacher):** list classes in the teacher's org; create class (name, description, exam preset); get enroll code/invite link; open a class.
- **`/classes/[id]` (teacher):** tabs:
  - **Subjects:** add/remove subjects. Any co-teacher in the org can add a subject.
  - **Students:** roster; add by email or invite link; remove.
  - **Materials:** per subject — upload PDF; select a material and generate **flashcards** or a **mock test** from it (calls existing `gemini_service` generation flows, now class-scoped via `class_materials.doc_id`).
  - **Tests:** create a **mock** or **actual** test. For actual: pick subject, set `start_at`, `end_at`, `duration_minutes`, marks; publish to class.
  - **Analytics:** per-student attempts, marks, strengths/weaknesses (existing teacher analytics, scoped to the class). Relocates today's `TeacherAlertsPanel`, `ClassChart`, and `StudentDetailPanel`.

### 6.2 Mock test
Existing flow, class-scoped. Immediate AI feedback to the student. All attempts visible to the teacher.

### 6.3 Actual test
- Scheduled window `start_at` → `end_at`; strictly timed `duration_minutes` once started.
- Visible to enrolled students only within the window. A student may start any time within `[start_at, end_at)`; once started they get `duration_minutes`, but the test **auto-submits at `end_at`** if still in progress (effective duration is `min(duration_minutes, end_at − start_time)`). No late starts after `end_at`.
- On submit, the student sees only a **"Submitted"** confirmation — no score, no analysis.
- Teacher sees every student's marks + answers + timing after submit (and within the window for in-progress status).

### 6.4 Preserved teacher features
Student monitoring (at-risk alerts, class performance chart, student detail with strengths/weaknesses, "create targeted test") is preserved and relocated into the class Analytics tab.

---

## 7. Student experience (independent + enrolled) & branding

### 7.1 Independent student (no `org_id`)
Unchanged: personal exams/subjects/collections/materials, mock tests, flashcards, focus mode, study planner, AI chat. `curriculum` is captured at onboarding and now also at **signup** (per "input at the start of the signup process").

### 7.2 Enrolled student (`org_id` set)
Everything above **plus:**
- **Home dashboard banner:** "You're enrolled as a student at {coaching name}" with **logo**.
- **Classes section:** cards for each enrolled class showing coaching **logo + name**, class name, subject count.
- **`/classes/[id]` (student view):** subjects → materials (read-only) → mock tests (attempt, get AI feedback) + actual tests (visible only within `start_at`–`end_at`; timed; no feedback after).

### 7.3 Branding plumbing
- `POST /orgs/logo` (multipart upload) → stored on disk under `uploads/orgs/{org_id}/` → `logo_url` on the org. Reuses the disk-upload pattern from PDFs.
- `GET /orgs/me` and a public-ish `GET /orgs/{org_id}/branding` return `name`, `brand_name`, `logo_url`, `tagline`.
- Frontend caches org branding; the teacher shell and student home/classes render the logo from the relevant org. Logo is visible to teachers in their dashboard chrome at all times; to students in the classes section and home dashboard.

---

## 8. AI chat fix (P5)

### 8.1 Bug 1 — no conversational memory
**Root cause:** `llm_service.ask_question` accepts no `history`; `question_router` session endpoints load prior messages but never pass them to the LLM. Each turn is answered statelessly from RAG context only.
**Fix:** Add `history: List[{role, content}]` param to `ask_question` / `stream_llm_response`. Session endpoints load prior session messages (cap to last ~10 turns) and pass them. Build Gemini `Content` list as `user/model/.../user(current)` turns, with the RAG context as a system-style preamble. Apply to both non-stream and stream paths (`/questions/sessions/{id}/messages` and `.../messages/stream`, plus the one-shot `/questions/ask[/stream]`).

### 8.2 Bug 2 — forced material selection / no free chat / no visible history
**Root cause:** `ChatInterface` refuses to send unless a material + session are selected; `ChatHistoryViewer` exists but is not mounted.
**Fix:** Allow a "General chat" session with no `doc_ids` (RAG returns no sources; Gemini answers from its own knowledge, flagged to the UI as uncited). Mount `ChatHistoryViewer` as a session sidebar so past sessions are reachable. Keep material-scoped chat as the other mode.

---

## 9. API surface (new/changed endpoints)

### Org & branding
- `POST /orgs/logo` (multipart) — subadmin uploads logo. *(new)*
- `GET /orgs/{org_id}/branding` — public branding (name, brand_name, logo_url, tagline). *(new)*
- `PATCH /orgs/` — extend to accept `tagline` (and `brand_name` as today). *(changed)*
- `GET /orgs/me` — include `logo_url`, `tagline`, plus monitoring lists (teachers, students, classes). *(changed)*

### Classes & subjects (teacher)
- `POST /classes/` — set `org_id`, `teacher_ids`. *(changed)*
- `POST /classes/{id}/subjects` — add a subject. *(new)*
- `GET /classes/{id}/subjects` — list subjects. *(new)*
- `DELETE /classes/{id}/subjects/{subject_id}` — remove subject (creator or org admin). *(new)*
- `POST /classes/{id}/teachers` — add a co-teacher (same org). *(new)*

### Class materials (teacher)
- `POST /classes/{id}/subjects/{subject_id}/materials` (multipart) — upload + index. *(new)*
- `GET /classes/{id}/subjects/{subject_id}/materials` — list (students read, teachers manage). *(new)*
- `DELETE /classes/{id}/materials/{material_id}` — remove (teacher). *(new)*
- `POST /classes/{id}/materials/{material_id}/generate-flashcards` — AI generate. *(new)*
- `POST /classes/{id}/materials/{material_id}/generate-mock-test` — AI generate. *(new)*

### Class students / invites (teacher)
- `POST /classes/{id}/students` — add by email (account exists → enroll; no account → create `class_invites`, return link). *(new/changed)*
- `POST /classes/{id}/invite-link` — return copyable invite link. *(new)*
- `DELETE /classes/{id}/students/{email}` — remove. *(existing, unchanged)*

### Tests (teacher)
- `POST /mock-tests/generate` (or a new `POST /classes/{id}/tests`) — class-scoped, `mode` param. *(changed/new)*
- `POST /classes/{id}/tests` — create actual test with schedule. *(new)*
- `GET /classes/{id}/tests` — list class tests. *(new)*
- `GET /classes/{id}/tests/{test_id}/submissions` — teacher sees all attempts/marks. *(new)*

### Tests (student)
- `GET /classes/me/tests` — tests available to the student (mock always; actual only within window). *(new)*
- `POST /mock-tests/{test_id}/submit` — unchanged for mock; for actual, suppress feedback in response. *(changed)*
- `GET /mock-tests/{test_id}` — for actual, strip `correctAnswer` and any feedback. *(changed)*

### Student classes & branding
- `GET /classes/me` — classes the student is enrolled in, with org branding. *(new)*
- `GET /classes/{id}` (student view) — subjects, materials (read), tests. *(changed)*

### Chat fix
- `POST /questions/sessions/{id}/messages[/stream]` — thread history. *(changed)*
- `POST /questions/ask[/stream]` — accept optional (empty) `doc_ids` for free chat. *(changed)*

---

## 10. Frontend changes summary

- **New routes:** `/classes` and `/classes/[id]` (shared teacher/student views, role-branched), teacher materials page consolidated into class view.
- **Sidebar (`app-shell.tsx`):** role-aware — teachers lose Focus/Plans; students gain Classes (when enrolled); coaching admins get monitoring tabs under `/org`.
- **Teacher dashboard (`/teacher`):** keep monitoring widgets; add entry to `/classes`; remove student-only feature links.
- **Student dashboard (`/dashboard`):** add coaching banner + logo (when `org_id`); add Classes section (when enrolled).
- **Onboarding/signup:** capture `curriculum` at signup step.
- **Chat (`/chat`):** relax material requirement; mount `ChatHistoryViewer`; surface "General chat" mode.
- **Org pages (`/org`, `/onboarding/org`):** add logo upload; add monitoring tabs (teachers, students, classes — read-only).

---

## 11. Testing & verification approach

- **Backend:** `pytest` per phase. New services get unit tests (class_service, class_material_service, actual_test scheduling/feedback suppression). API tests for role/ownership enforcement and the feedback-suppression contract for actual tests.
- **Frontend:** manual verification per phase against running dev servers (frontend :3000, backend :8001) using the `run`/browser flow; role-branched checks (coaching admin, teacher, independent student, enrolled student).
- **Per-phase verification gate:** each phase is verifiable end-to-end before the next begins (e.g., P2 = teacher can upload a class material and generate a flashcard from it; P3 = student can take an actual test in-window and sees no feedback while teacher sees marks).
- **Chat fix verification:** a multi-turn conversation where the answer to turn 2 depends on turn 1 (proves memory); a free-chat session with no material selected.

---

## 12. Open assumptions (flagged for user)
- Invite-by-email defaults to a **copyable link** unless SMTP is configured; email send is best-effort.
- Teachers retain access to **Chat** (used for material-scoped AI generation); only Focus/Plans are removed from the teacher experience.
- The personal study hierarchy (Exam → Subject → Collection → Material) stays intact for independent students.
- `admin` (platform super-admin) remains separate from `subadmin` (coaching admin).