# Orbit Implementation Plan

## Goal
Make every existing frontend feature functional, then extend the platform to support a four-role ecosystem:
- **Student**: upload syllabus/notes, chat with materials, attempt AI-generated quizzes/mock tests, view per-subject analytics, generate flashcards.
- **Teacher**: manage multiple students, create topic-aware mock tests, optionally target student weaknesses, monitor student results.
- **Sub-admin**: license holder who can enroll teachers/students under a branded instance and monitor all users beneath them.
- **Admin**: super-user with a separate admin dashboard.

---

## Current State Snapshot

### Frontend (`Frontend/`)
- Next.js 16 + App Router. JWT stored in `localStorage`.
- Pages exist for marketing, auth (`/login`, `/signup`), onboarding, and dashboard (`/dashboard`, `/chat`, `/test`, `/settings`).
- Dashboard context fetches exams via `/api/exams`; onboarding uses `/api/onboarding`.
- Test page wires to PDF/mock/analysis endpoints (`/pdfs`, `/mock-tests`, `/analysis`).
- **Broken/placeholder areas**
  1. Signup/onboarding role is not persisted (backend `UserCreate` has no role field).
  2. Dashboard stats are hardcoded to `—`.
  3. Chat page uses `Material` objects from dashboard context, but materials uploaded through `/api/collections/{id}/materials` are only metadata stubs with a fake `url`; they are **not** parsed, chunked, or indexed, so chat over them is non-functional.
  4. The materials panel in dashboard (`CollectionsPanel`, `SubjectCard`, etc.) is currently a stub that shows empty-state copy.
  5. No role-based routing; a teacher lands on the same dashboard as a student.
  6. No student analytics backend or global report view.
  7. No flashcard generation UI or backend.

### Backend (`Backend/src/`)
- FastAPI, MongoDB via Motor, ChromaDB for vectors, Gemini (`gemini-2.5-flash`) for generation.
- Auth endpoints: `/auth/login`, `/auth/signup`, `/auth/me`. JWT `sub` = email.
- PDF endpoints: `/pdfs/upload`, `/pdfs/`, `/pdfs/{id}`, `/pdfs/{id}/download`.
- Document endpoints: `/documents/upload` (extracts, chunks, embeds, stores to Chroma + Mongo chunks).
- Chat endpoints: `/questions/ask`, `/questions/sessions`, etc.
- Mock test endpoints: `/mock-tests/generate`, `/mock-tests/`, `/mock-tests/{id}`, `/mock-tests/{id}/submit`, `/mock-tests/submissions/{id}/analysis`.
- Workspace endpoints: `/api/exams`, `/api/exams/{id}/subjects`, `/api/subjects/{id}/collections`, `/api/collections/{id}/materials`, `/api/onboarding`.
- Models already have `User.role: "student" | "teacher"` but no `admin`/`subadmin`, no teacher/student linkage, no subscriptions.

### Key Gaps
1. Role system incomplete.
2. No teacher/student relationship model.
3. Materials in workspace are not processed for RAG (unlike `/documents/upload`).
4. No analytics/weakness tracking across submissions.
5. No admin/sub-admin dashboards or user management.
6. No subscription/license data model.
7. No internet-enabled AI agent workflow.
8. No flashcard generation flow.

---

## Proposed Architecture

### 1. User Roles & Ownership
Add roles: `student`, `teacher`, `subadmin`, `admin`.

```
User
├── role
├── managed_by: Optional[teacher_id | subadmin_id]
├── teacher_id: Optional[str]        # direct teacher link for students
├── license_id: Optional[str]        # for sub-admins
├── subscription: { plan, started_at, expires_at, status }
├── onboarding_completed
└── active_exam_id
```

- A **teacher** has a list of student ids (`managed_students`).
- A **sub-admin** has a license record and a list of managed teachers/students under their brand.
- An **admin** can see everything.

### 2. Teacher Manages Students
Implementation: teacher dashboard with a "My Students" section.
- Teacher can add a student by email.
- If the student exists, link via `teacher_id` and add to teacher's `managed_students`.
- If not, create a placeholder/stub record (optional) or send an invite token.
- Teacher can remove students.
- Students can see their assigned teacher.

### 3. Materials Pipeline (Make Chat Work)
The chat page expects materials attached to subjects/collections. The cleanest fix:
- Reuse the existing `/documents/upload` processor when a material is uploaded through `/api/collections/{id}/materials`.
- Store the resulting `doc_id` on the `Material` record.
- Update `Document` so it can optionally reference `collection_id`/`subject_id`.
- Add tags (`syllabus`, `notes`, `question-paper`, `general`) based on upload context.
- Chat component uses `doc_id` for RAG context instead of the fake `url`.

### 4. Mock Tests & Analytics
- Extend `MockTestGenerationRequest` with:
  - `targeted_student_id` (for teachers)
  - `selected_topics`: list of topics/units
  - `target_weaknesses: bool` — when true, the prompt instructs Gemini to emphasize topics the student got wrong in prior submissions.
- Store analytics per submission:
  - per-topic score breakdown
  - time per question
  - weak topics list
- Add `/analytics/me` endpoint returning global and per-subject dashboards.
- Add `/students/{id}/analytics` for teachers.

### 5. Flashcards
- New endpoint `/flashcards/generate` that takes a `doc_id` or `subject_id` and returns AI-generated flashcards.
- Store them in a new `flashcards` collection.
- Simple UI to generate, flip through, and mark known/unknown.

### 6. AI Agent Workflow + Internet
Two practical approaches:
- **Option A**: Use Gemini's native `google_search` tool if the API key supports it. Simplest; no extra dependency.
- **Option B**: Add a web-search agent (SerpAPI / DuckDuckGo) that fetches real examples, then passes them to Gemini for question generation.

Proposed: implement an `AgentQuestionGenerator` service that:
  1. Parses the syllabus into topics.
  2. For each topic, optionally searches the web for recent exam questions/examples.
  3. Sends a structured prompt to Gemini with syllabus, previous papers, notes, and web examples.
  4. Parses JSON output and validates it.
  5. Falls back to local-only generation when search is unavailable.

### 7. Subscriptions & Licensing
Simple model first:
- `License` collection for sub-admins: `brand_name`, `seats`, `expires_at`, `status`.
- `Subscription` collection for students: `plan` (`weekly`, `monthly`), `started_at`, `expires_at`, `status`.
- Add middleware/guard endpoints to enforce active subscription before test generation.
- Payment gateway is left as a follow-up integration (Stripe/Razorpay). For now, add a manual activation endpoint for admin/sub-admin.

### 8. Admin Dashboard (Separate)
- New route group `(admin)/` in frontend.
- New backend router `/admin/...` with role guard.
- Admin can: list all users by role, assign licenses, activate subscriptions, view platform analytics.

---

## Implementation Phases

Because the request is large, I recommend implementing in phases. Each phase leaves the app in a buildable, testable state.

### Phase 0 — Foundation & Fix Existing Features
1. Update `User` model and `UserCreate`/`UserResponse` to support all four roles.
2. Persist onboarding role to the user record.
3. Add role-aware `/auth/me` response.
4. Create a global frontend `AuthContext` that loads `/auth/me`, exposes `user` + `role`, and is used by route guards.
5. Update dashboard layout to redirect by role (teacher → `/teacher/dashboard`, student → `/dashboard`, etc.).
6. Make workspace material upload use the document processor so chat works.
7. Update chat components to query by `doc_id`.
8. Add backend enforcement: every RAG endpoint verifies the document belongs to the user (or their teacher/student chain).

### Phase 1 — Student Core Features
1. Student dashboard global analytics (`/analytics/me`).
2. Per-subject progress and weakness cards.
3. Flashcard generation endpoint + UI page.
4. Fix quiz/test results page integration (already mostly done; verify analysis persistence).

### Phase 2 — Teacher Features
1. Teacher dashboard layout and "My Students" page.
2. Add/remove students API + UI.
3. Create mock test for self or selected student.
4. Topic selection UI and `target_weaknesses` toggle.
5. Student analytics viewer (`/teacher/students/{id}`).

### Phase 3 — Admin & Sub-admin
1. Role guards (`require_role`).
2. Admin dashboard: user list, role assignment, license management.
3. Sub-admin dashboard: enroll teachers/students under their brand, view their users.
4. License CRUD backend.

### Phase 4 — Subscriptions
1. `Subscription` and `License` models/collections.
2. Student subscription status check before test generation.
3. Sub-admin license seat/expiration check before enrolling users.
4. Manual activation endpoints (payment integration as future work).

### Phase 5 — AI Agent + Internet
1. Add `AgentQuestionGenerator` service.
2. Optional web-search provider (native Gemini search tool or external API).
3. Wire mock-test generation to the agent.
4. Add flashcard agent workflow.

---

## Immediate Next Steps (what I will do first)

1. Refactor backend auth/user models and add role guards.
2. Create frontend `AuthContext` and role-based route wrappers.
3. Make workspace materials actually processable for RAG so chat works.
4. Verify and fix the existing test/mock-test flow end-to-end.

---

## Decisions

1. **Web search for AI**: External search API (SerpAPI or DuckDuckGo) for more control.
2. **Payments**: Manual activation by admin/sub-admin for this cycle (no payment gateway yet).
3. **Priority**: Student + teacher features first, then admin/sub-admin/licensing.
