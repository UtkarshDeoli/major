# Phase 2b/3 — Study Experience + Production Hardening

**Goal:** Finish the sellable platform: add focus-mode study tools, multimodal chat, lesson planning, and the production safeguards needed before launching to coaching centers and students.

**Branch:** `phase1/sellable-backbone`

## Scope

### Phase 2b — Study Experience

1. **Focus mode / Pomodoro timer**
   - Backend: `POST /study/focus-sessions` to log focus sessions; `GET /study/focus-stats`.
   - Frontend: full-screen focus overlay with Pomodoro timer, task selector, and session summary.

2. **Image input for AI chat**
   - Backend: extend `POST /questions/ask` to accept an optional `image_url` / `image_base64`; include image in Gemini prompt via `gemini-2.5-flash` multimodal support.
   - Frontend: image upload button in chat input.

3. **AI lesson planner / study calendar**
   - Backend: `POST /study/plans` — given exam date, subjects, weak topics, generate a weekly study plan using Gemini. Store plans in new `study_plans` collection.
   - Frontend: `/plans` page with calendar/list view and progress tracking.

4. **NotebookLM-style persistent document chat**
   - Already partially built via `/questions/sessions`. Harden it: list sessions in sidebar, rename, delete.

### Phase 3 — Production Hardening

1. **Rate limit AI-generation endpoints**
   - Apply slowapi limits to `/mock-tests/generate`, `/flashcards/generate`, `/ai-materials/summarize`, `/analysis/question-papers` (per-user, e.g. 10/hour for generation endpoints).

2. **Security headers**
   - Add CORS, CSP, HSTS, X-Content-Type-Options via middleware/turbopack headers in `next.config.js`.

3. **Backend health / readiness**
   - Extend `/healthcheck` to report MongoDB + Razorpay readiness.

4. **Frontend error boundaries**
   - Add `error.tsx` boundaries for dashboard routes and global fallback.

5. **Documentation / README**
   - Update root README with architecture, env setup, deployment notes.

## Technical Decisions

- **Focus sessions:** Stored per user with `started_at`, `ended_at`, `duration_minutes`, `task`, `completed`.
- **Image input:** Use base64 data URI for inline images. Gemini 2.5 Flash supports image parts.
- **Lesson plans:** Generated as a list of `weeks -> days -> tasks`. Each task links to a subject/topic and has `completed` boolean.
- **Rate limits:** slowapi already wired; add per-user limits using `get_remote_address` + user email via custom key function.

## Files to Touch

- `Backend/src/routers/question_router.py`
- `Backend/src/services/gemini_service.py`
- `Backend/src/services/llm_service.py`
- `Backend/src/routers/study_router.py` (new)
- `Backend/src/core/data_store.py`
- `Backend/src/core/limiter.py`
- `Backend/src/routers/mock_test_router.py`, `flashcard_router.py`, `ai_material_router.py`, `analysis_router.py`
- `Backend/src/main.py`
- `Frontend/app/(dashboard)/focus/page.tsx`
- `Frontend/app/(dashboard)/plans/page.tsx`
- `Frontend/components/dashboard/chat/chat-input.tsx`
- `Frontend/components/dashboard/chat/chat-interface.tsx`
- `Frontend/components/dashboard/chat/chat-history-viewer.tsx`
- `Frontend/components/dashboard/sidebar.tsx`
- `Frontend/next.config.js`
- `Frontend/app/error.tsx`
- `README.md`

## Acceptance Criteria

- [ ] `pytest tests/` passes.
- [ ] `npm run build` and `npm run lint` pass.
- [ ] Focus mode page works with timer and session logging.
- [ ] Chat accepts image uploads and receives multimodal responses.
- [ ] Lesson planner generates and persists weekly plans.
- [ ] AI generation endpoints are rate-limited per user.
- [ ] Frontend has error boundaries and security headers.

## Deferred

- Voice/audio chat (too niche for v1).
- Mobile app.
- Advanced analytics dashboards beyond what exists.
