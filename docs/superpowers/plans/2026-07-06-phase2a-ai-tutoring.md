# Phase 2a — AI Tutoring & Study Experience (Sellable Backbone)

**Goal:** Make Orbit pedagogically effective, not just feature-complete. Build the AI tutoring layer that directly improves learning outcomes: Socratic explanations, rubric grading, adaptive difficulty, at-risk alerts, and multilingual responses.

**Branch:** `phase1/sellable-backbone`

## Scope

1. **Socratic step-by-step explanations**
   - Backend: `POST /questions/socratic` — given a question/concept, return guided hints instead of direct answers.
   - Frontend: "Explain like a tutor" button in chat and mock-test review.

2. **AI rubric-based grading for text answers**
   - Backend: extend mock-test analysis to use a rubric (accuracy, completeness, clarity, examples) for text answers; store rubric scores.
   - Frontend: show rubric breakdown in test results.

3. **Adaptive difficulty mock tests**
   - Backend: `POST /mock-tests/generate` accepts optional `adaptive=true`; analyze last 3 submissions to pick difficulty mix.
   - Store per-student mastery scores by topic.

4. **Teacher early-warning analytics**
   - Backend: `GET /analytics/teacher/alerts` returns at-risk students (low score, dropping trend, inactive >7 days).
   - Frontend: alerts panel on teacher dashboard.

5. **Multilingual AI response support**
   - Backend: read `preferred_language` from user profile; append "respond in X" to LLM prompts in `llm_service`.
   - Frontend: no changes needed; responses arrive in selected language.

## Technical Decisions

- **Socratic mode:** Uses Gemini with a strict prompt that never gives the final answer in the first turn. Returns an array of `steps` (hint → probe → partial answer → next probe). Frontend renders step cards one by one.
- **Rubric grading:** Extend `AnswerFeedback` model with `rubric_scores` and `rubric_total`. Gemini returns JSON with per-criterion scores (0-max) and justification. No DB schema change needed; rubric lives inside `question_feedback`.
- **Adaptive difficulty:** New lightweight collection `student_mastery` keyed by `(user_id, topic)`. After each submission, update topic mastery using `+delta` (correct hard = +2, easy = +0.5, wrong = -1). Generation reads this to bias question difficulty.
- **Early-warning:** Pure analytics computation over existing submission data. Flags:
  - `score_drop`: avg last 2 tests vs previous 3 drops >15 pts.
  - `inactive`: no submission in 7 days.
  - `low_mastery`: >2 weak sections with accuracy <40%.
- **Multilingual:** Read user profile's `preferred_language` in `question_router` and pass to `ask_question`. Use ISO-639-1 codes mapped to human language names for the prompt.

## Data Model Changes

### `student_mastery_collection` (new)
- `user_id: str`
- `topic: str`
- `score: float` (0-100)
- `updated_at: datetime`

### `AnswerFeedback` (extend)
- Add `rubric_scores: dict[str, int] | None`
- Add `rubric_max: dict[str, int] | None`

### `MockTestQuestion` (extend)
- Add `difficulty: str | None` (already present in generation prompt but not persisted)

## API Additions

- `POST /questions/socratic` (Body: `{question, doc_ids?, subject?, concept?}`)
- `GET /analytics/teacher/alerts`
- `GET /analytics/teacher/insights` (per-student weak topics + recommended actions)
- `POST /mock-tests/{test_id}/socratic-feedback` (Body: `{question_id, answer}`) — get Socratic feedback on a wrong answer

## Frontend Changes

- Chat: "Socratic mode" toggle/button in `chat-interface.tsx`
- Mock test results: rubric breakdown in results page
- Teacher dashboard: alerts panel and insights table
- Settings: confirm `preferred_language` is saved

## Tests

- Backend pytest: add `test_socratic.py`, `test_rubric_grading.py`, `test_adaptive_difficulty.py`, `test_teacher_alerts.py`
- Frontend: rely on `next build` and `npm run lint` (0 errors).

## Acceptance Criteria

- [ ] `pytest tests/` passes.
- [ ] `npm run build` and `npm run lint` pass.
- [ ] Socratic endpoint returns step-by-step hints, not direct answers.
- [ ] Text answers include rubric scores in submission analysis.
- [ ] Adaptive mock tests generate harder/easier questions based on past performance.
- [ ] Teacher alerts surface at-risk students with reasons.
- [ ] AI chat respects user's `preferred_language`.

## Deferred

- Voice/image multimodal input (Phase 2b).
- Focus mode / Pomodoro timer (Phase 2b).
- Lesson calendar / study planner (Phase 3).
