# Orbit — End-to-End Testing Checklist by Role

Use this checklist before any release, demo, or major refactor. Each section maps to a real user journey and includes the expected success criteria. Mark items as ✅, ❌, or ⏳ and note the tested commit/branch.

**Test environment assumptions**
- Backend running at `http://localhost:8001` (or staging URL)
- Frontend running at `http://localhost:3000` (or staging URL)
- MongoDB, ChromaDB, and Gemini API are healthy (`GET /healthcheck` returns `mongodb.ok: true`)
- Razorpay keys configured for billing tests (`/healthcheck` reports `razorpay.ok` status)

---

## 0. Platform Health & Onboarding

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 0.1 | Healthcheck | `GET /healthcheck` | Returns `status`, `mongodb.ok`, `razorpay.ok`. 503 if MongoDB is down. | |
| 0.2 | Landing page | Visit `/` | Marketing page loads without console errors. | |
| 0.3 | Sign up as student | `/signup` → create account | Logged in and redirected to onboarding or dashboard. | |
| 0.4 | Sign up duplicate email | Try same email again | Clear error: user already exists. | |
| 0.5 | Login | `/login` with valid credentials | Token stored, redirected to role-appropriate dashboard. | |
| 0.6 | Login invalid password | Wrong password | 401 toast, not redirected. | |
| 0.7 | Token expiry handling | Use app after token expires | 401 redirects to `/login` (no infinite loop). | |
| 0.8 | Onboarding flow | `/onboarding` → pick exam/subjects | Preferences persisted; calls `/api/onboarding`. | |
| 0.9 | Google OAuth | Click "Sign in with Google" | Callback succeeds, account created/logged in. | |
| 0.10 | Password reset | Request reset link | Email/link flow works (or placeholder UI handled). | |
| 0.11 | Global error boundary | Trigger an error on a marketing page | `app/error.tsx` fallback shown, no leaked stack traces. | |
| 0.12 | Dashboard error boundary | Trigger an error inside `(dashboard)` | `app/(dashboard)/error.tsx` fallback shown with retry/dashboard buttons. | |
| 0.13 | 404 page | Visit non-existent dashboard route | `not-found.tsx` renders. | |
| 0.14 | Security headers | `curl -I http://localhost:3000` | HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy present. | |
| 0.15 | Rate-limit headers | Hit auth endpoints rapidly | 429 returned by slowapi; plan limits return 402 on AI endpoints. | |

---

## 1. Student Role — Core Study Flow

Use a fresh student account (no subscription, on Starter plan).

### 1.1 Documents & RAG Chat

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 1.1.1 | Upload PDF | Dashboard → upload a syllabus/notes PDF | PDF appears in document list with metadata. | |
| 1.1.2 | PDF storage limit | Upload until 50 MB Starter cap | 402 upgrade prompt with resource `doc_storage`. | |
| 1.1.3 | Workspace material upload | Subject → Collection → Add material | Material created and linked; document is chunked and indexed for RAG. | |
| 1.1.4 | Chat over document | `/chat` → select document → ask a question | Answer is grounded in document; `sources` shown. | |
| 1.1.5 | Multimodal chat | Ask question with image upload | Image reaches backend; Gemini responds (fallback text if unsupported). | |
| 1.1.6 | Chat sessions | Create new session, rename, switch, delete | Session list persists; messages scoped per session. | |
| 1.1.7 | Chat streaming | Ask a question with streaming enabled | Tokens/paragraphs stream in without errors. | |
| 1.1.8 | Chat plan limit | Send 100+ messages on Starter | 402 `chat_message` upgrade prompt. | |
| 1.1.9 | Socratic tutor | `/socratic` or chat → "Explain step by step" | Response guides without giving direct answer. | |
| 1.1.10 | Multi-document RAG | Chat session with 2+ documents | Answer synthesizes across selected documents. | |

### 1.2 Mock Tests

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 1.2.1 | Generate mock test | `/mock-tests` → select syllabus + previous papers → generate | Test created with MCQ + descriptive questions, marks, time limit. | |
| 1.2.2 | Starter plan limit | Generate 4 tests on Starter | 4th generation returns 402 `mock_test`. | |
| 1.2.3 | View test | Click generated test | Questions render; timer starts if enabled. | |
| 1.2.4 | Submit test | Answer MCQs and text questions → submit | Auto-graded for MCQ; descriptive pending or AI-graded depending on mode. | |
| 1.2.5 | View analysis | Go to results after submission | Score, percentage, feedback per question, strengths/improvements shown. | |
| 1.2.6 | Retake / list submissions | Return to test → see attempts | Submissions list loads for the student. | |
| 1.2.7 | Teacher-marked mode | Teacher creates test with `grading_mode: teacher-marked` → student submits | Status shows "pending review" until teacher grades. | |
| 1.2.8 | Adaptive difficulty | Generate adaptive test after weak topics exist | Difficulty distribution shifts based on mastery. | |

### 1.3 Flashcards & AI Materials

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 1.3.1 | Generate flashcards | `/flashcards` → select document → generate | Deck with front/back cards created; flip interaction works. | |
| 1.3.2 | Starter flashcard limit | Generate 51 cards on Starter | 402 `flashcard` upgrade prompt. | |
| 1.3.3 | Review flashcards | Flip cards, mark known/unknown | Progress/state persisted per card. | |
| 1.3.4 | Delete deck | Delete a flashcard deck | Removed from list and DB. | |
| 1.3.5 | AI summary | `/analysis` or document panel → summarize | Summary generated; counts against `ai_material` quota. | |
| 1.3.6 | AI material limit | Generate 6 summaries on Starter | 402 `ai_material` upgrade prompt. | |

### 1.4 Study Planner & Focus Mode

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 1.4.1 | Create study plan | `/plans` → fill exam date, subjects, weak topics → generate | Weekly plan with 7 days/week returned and persisted. | |
| 1.4.2 | Study plan limit | Create 6 plans on Starter | 6th plan returns 402 `study_plan`. | |
| 1.4.3 | Update task progress | Check a task in week/day view | Backend PATCH succeeds; visual strike-through persists on reload. | |
| 1.4.4 | Delete plan | Delete a study plan | Plan removed; list updates. | |
| 1.4.5 | Focus session | `/focus` → enter task → start timer | Session created; timer counts down. | |
| 1.4.6 | Pause / complete focus | Pause, resume, complete | Session PATCH updates `completed` and `ended_at`. | |
| 1.4.7 | Focus stats | Complete a session → view stats | Total minutes, weekly minutes, completed count increase. | |

### 1.5 Analytics & Billing

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 1.5.1 | Student analytics | `/analytics` | Global stats, per-subject progress, weak topics load. | |
| 1.5.2 | Usage meters | `/billing` | Current plan limits and usage shown (mock tests, flashcards, chat, etc.). | |
| 1.5.3 | Upgrade flow | Click upgrade → Razorpay checkout | Test mode checkout succeeds; subscription becomes active. | |
| 1.5.4 | Cancel subscription | `/billing` → cancel | Subscription status becomes `cancelled`; still active until period end. | |
| 1.5.5 | 402 banner | Hit any quota limit | Global `orbit:upgrade-required` event shows upgrade banner. | |

---

## 2. Teacher Role — Classroom Management

Create or promote a user to `teacher`.

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 2.1 | Teacher dashboard | Log in as teacher | Redirected to `/teacher` with "My Students" section. | |
| 2.2 | Add existing student | `/teacher` → add student by email | Student linked (`teacher_id` / `managed_students` updated). | |
| 2.3 | Add non-existent student | Add email with no account | Appropriate error or invite placeholder. | |
| 2.4 | Remove student | Remove a managed student | Link removed; student no longer appears in list. | |
| 2.5 | View student analytics | Click student in teacher dashboard | `/teacher/students/{id}` shows analytics, weak topics, submissions. | |
| 2.6 | Create test for self | Teacher generates mock test | Test appears in teacher's list; student field optional. | |
| 2.7 | Create test for student | Generate test with `student_email` | Test `assigned_to` set to student; student sees it. | |
| 2.8 | Assign existing test | `POST /mock-tests/{id}/assign?student_email=...` | Assignment succeeds only if student is managed by teacher. | |
| 2.9 | Unauthorized assignment | Assign test to a student managed by another teacher | 403 returned. | |
| 2.10 | Teacher-marked grading | Grade pending descriptive answers | `POST /mock-tests/submissions/{id}/grade` updates score and status. | |
| 2.11 | View class submissions | List submissions for a test | Teacher sees assigned student's attempts. | |
| 2.12 | Topic-targeted test | Generate test with `focus_topics` / `target_weaknesses` | Prompt emphasizes selected topics / weak areas. | |
| 2.13 | Teacher plan limits | Teacher on Starter tries to manage >1 class | `class_count` plan enforcement works. | |

---

## 3. Sub-admin Role — Organization / Seat Licensing

Create or promote a user to `subadmin`.

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 3.1 | Create organization | `/org` or admin → create org | Org created; user becomes owner with seat count. | |
| 3.2 | Invite teacher | Generate invite code/link | Code can be shared; link contains org context. | |
| 3.3 | Invite student | Generate invite code/link | Same as above, role scoped to student. | |
| 3.4 | Enroll via invite | New user signs up with invite code | Seat consumed; user linked to org with correct role. | |
| 3.5 | Seat limit | Enroll users up to seat cap | Next enrollment blocked or prompts to add seats. | |
| 3.6 | Add seats | Purchase/upgrade seats | Razorpay order created; seats increased on success. | |
| 3.7 | Remove member | Remove teacher/student from org | Seat freed; member loses org-scoped access. | |
| 3.8 | Org usage view | `/org` dashboard | Seat usage, member list, invitations visible. | |
| 3.9 | Org plan enforcement | Org on Pro tier vs individual on Starter | Effective plan resolves to org tier for members. | |
| 3.10 | Branding/instance isolation | Members see org context | Org name/tier reflected in dashboard (if UI supports). | |

---

## 4. Admin Role — Platform Management

Create or promote a user to `admin`.

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 4.1 | Admin login | Log in as admin | Redirected to `/admin` dashboard. | |
| 4.2 | List all users | `/admin` users tab | All users with roles, statuses, subscriptions visible. | |
| 4.3 | Filter users | Filter by role/status | Results update correctly. | |
| 4.4 | Change user role | PATCH `/admin/users/{email}/role` | Role updated; user sees new dashboard on next login. | |
| 4.5 | Suspend/activate user | PATCH `/admin/users/{email}/status` | Status toggled; suspended user cannot access protected routes. | |
| 4.6 | Manual subscription activation | Admin activates subscription for user | User's effective plan changes; usage meter updates. | |
| 4.7 | List organizations | `/admin/orgs` | All orgs with tiers, seat usage, status. | |
| 4.8 | Suspend org | PATCH `/admin/orgs/{id}` → suspended | Org members fall back to individual/free plan. | |
| 4.9 | Platform analytics | `/admin` analytics tab | Revenue, user growth, active subscriptions render. | |
| 4.10 | Role-based route guard | Non-admin visits `/admin` | Redirected away or 403 page shown. | |
| 4.11 | Admin impersonation (if exists) | Act as another user | Session scoped correctly; audit trail logged. | |

---

## 5. Billing & Payments

Test across all roles where billing applies.

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 5.1 | Plans endpoint | `GET /subscriptions/plans` | Starter/Pro/Premium prices and limits returned. | |
| 5.2 | Checkout create | `POST /subscriptions/checkout` | Razorpay order_id returned. | |
| 5.3 | Checkout verify | Submit Razorpay payment signature | Subscription created/updated. | |
| 5.4 | Webhook signature | Send test Razorpay webhook | Signature verified; subscription/order updated. | |
| 5.5 | Invoice list | `GET /subscriptions/invoices` | Invoices returned for the user. | |
| 5.6 | Cancel subscription | `POST /subscriptions/cancel` | Status changes; renewal stopped. | |
| 5.7 | Plan enforcement matrix | Verify each resource limit per plan | See README table; 402 returned at each threshold. | |
| 5.8 | Free fallback | Cancel subscription, remove org | User reverts to Starter limits. | |

---

## 6. Security & Hardening

| # | Test | Steps | Expected Result | Status |
|---|---|---|---|---|
| 6.1 | CORS | Request backend from wrong origin | Blocked or no credentials accepted. | |
| 6.2 | Auth-protected routes | Call `/questions/ask` without token | 401 returned. | |
| 6.3 | Ownership checks | Student A tries to fetch Student B's test/chat | 403 or 404 (no data leak). | |
| 6.4 | SQL/NoSQL injection | Send `$ne` or other operators in form fields | Inputs treated as strings; no injection. | |
| 6.5 | File upload limits | Upload very large PDF | Rejected or limited by plan/storage. | |
| 6.6 | Slowapi throttling | Burst AI endpoints from same user/IP | 429 after limit; legitimate users not blocked globally. | |
| 6.7 | Error message leakage | Trigger backend exception on chat/test endpoints | Generic 500 message; no stack trace in response body. | |
| 6.8 | CSP violation | Check browser console on dashboard | No blocked script/style errors from self-hosted assets. | |

---

## 7. API-Specific Regression Tests

Run these with `pytest` or via Postman/Bruno for confidence before release.

| # | Endpoint Suite | Test File / Route | Status |
|---|---|---|---|
| 7.1 | Auth | `tests/test_auth.py` | |
| 7.2 | Admin | `tests/test_admin.py` | |
| 7.3 | Organizations | `tests/test_orgs.py` | |
| 7.4 | Subscriptions/Billing | `tests/test_subscriptions.py`, `test_billing.py`, `test_webhooks.py` | |
| 7.5 | Plan enforcement | `tests/test_plans_enforcement.py` | |
| 7.6 | Rate limiting | `tests/test_rate_limit.py` | |
| 7.7 | Teacher-student | `tests/test_teacher_student.py` | |
| 7.8 | Study tools | `tests/test_study.py` | |
| 7.9 | Phase 2a AI tutoring | `tests/test_phase2a.py` | |
| 7.10 | RAG pipeline | `tests/test_multidoc_rag.py`, `test_query_engine.py`, `test_vector_store.py`, `test_bm25_index.py`, `test_document_processor.py` | |

---

## 8. Frontend Build & Quality

| # | Command | Expected Result | Status |
|---|---|---|---|
| 8.1 | `npm run lint` | 0 errors | |
| 8.2 | `npm run build` | Build succeeds, all routes prerender | |
| 8.3 | `npm test` | All Vitest tests pass | |
| 8.4 | Manual smoke (Chrome) | Core flows work, no console errors | |
| 8.5 | Manual smoke (Safari/Firefox) | Cross-browser compatibility acceptable | |
| 8.6 | Mobile responsive | Dashboard pages usable at 375 px width | |

---

## How to use this checklist

1. Create test accounts for each role (`student-a@example.com`, `teacher-a@example.com`, `subadmin-a@example.com`, `admin-a@example.com`).
2. Run the backend test suite: `cd Backend && pytest tests/ -v`.
3. Run the frontend quality gates: `cd Frontend && npm run lint && npm run build && npm test`.
4. Walk through Sections 0–6 manually in the browser or with an API client.
5. For each failed item, open a bug ticket with: role, route, reproduction steps, expected vs actual, and browser/API response.

---

## Appendix: Quick role-seed commands

```bash
# Promote an existing user to teacher/subadmin/admin via MongoDB
cd Backend
source venv/bin/activate
python - <<'PY'
import asyncio
from src.core.data_store import users_collection
from src.services.auth_service import get_password_hash

async def seed():
    password = get_password_hash("TestPass123!")
    for email, role in [
        ("admin-test@example.com", "admin"),
        ("subadmin-test@example.com", "subadmin"),
        ("teacher-test@example.com", "teacher"),
        ("student-test@example.com", "student"),
    ]:
        await users_collection.update_one(
            {"email": email},
            {"$set": {"email": email, "password_hash": password, "role": role}},
            upsert=True,
        )
        print(f"Seeded {email} as {role}")

asyncio.run(seed())
PY
```
