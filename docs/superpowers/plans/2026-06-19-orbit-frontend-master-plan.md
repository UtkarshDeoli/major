# Orbit Frontend Improvements — Master Implementation Plan

> **For agentic workers:** This is a master/index plan. Each phase is a self-contained subsystem
> that should be executed from its own detailed plan document (linked under each phase). When you
> start a phase, open its dedicated plan file and follow it task-by-task with
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

**Goal:** Bring the Orbit frontend from its current state (broken stubs, `any`-everywhere,
client-only fetching, a11y gaps, missing features) to a correct, accessible, typed, and
feature-complete study platform.

**Architecture:** Incremental, phase-gated refactors. Phase 0 fixes correctness bugs that harm
users today. Phase 1 establishes the typed API layer + data-fetching cache + centralized error
handling that every later feature depends on. Phases 2–6 layer UX/a11y polish, new features, code
refactor, perf, and testing on top. Each phase produces working, testable software on its own and
can ship independently.

**Tech Stack:** Next.js 16 (App Router) · React 19 · TypeScript · Tailwind · Radix/shadcn · axios ·
framer-motion · recharts · react-pdf. (Testing: Vitest + Testing Library + Playwright — to be
added in Phase 1 Task 1.1 / Phase 6.)

**Source spec:** [`Frontend/IMPROVEMENT_PLAN.md`](../../Frontend/IMPROVEMENT_PLAN.md) (full audit
findings with file:line evidence).

---

## Scope check (why this is split)

The spec spans 7 independent subsystems. A single TDD plan with complete per-step code for ~65
items across all of them would be unwieldy and error-prone. Per `writing-plans` Scope Check, this
master plan declares the split into per-phase plans. Each phase plan is self-contained and
producible as working software on its own.

**Phase plans (detailed task-by-task files):**
- Phase 0 — [2026-06-19-orbit-frontend-phase0-critical-fixes.md](./2026-06-19-orbit-frontend-phase0-critical-fixes.md) ✅ written
- Phase 1 — `2026-06-19-orbit-frontend-phase1-foundation.md` (generate before executing)
- Phase 2 — `2026-06-19-orbit-frontend-phase2-ux-a11y.md` (generate before executing)
- Phase 3 — `2026-06-19-orbit-frontend-phase3-features.md` (generate before executing)
- Phase 4 — `2026-06-19-orbit-frontend-phase4-code-quality.md` (generate before executing)
- Phase 5 — `2026-06-19-orbit-frontend-phase5-perf-infra.md` (generate before executing)
- Phase 6 — `2026-06-19-orbit-frontend-phase6-testing-ci.md` (generate before executing)

---

## Execution order & dependencies

```
Phase 0 (critical fixes)  ──┐
                             ├─► Phase 1 (foundation: types, cache, errors)
                             │        │
                             │        ├─► Phase 2 (UX/a11y)        ─┐
                             │        ├─► Phase 3 (features)        ├─► Phase 4 (code quality)
                             │        │                              │        │
                             │        └─► Phase 5 (perf/infra)      ─┘        ├─► Phase 6 (testing/CI)
                             └────────────────────────────────────────────────┘
```

- **Phase 0** has no prerequisites — start here.
- **Phase 1** unblocks safe feature work (typed APIs + cache + error helper). Do it second.
- **Phases 2/3/5** can proceed in parallel after Phase 1 (different files, minimal overlap).
- **Phase 4** (code refactor) is largely a cleanup pass best done after features land.
- **Phase 6** (tests + CI) should be added incrementally alongside Phases 1–3 rather than after.

---

## Branching

Each phase gets its own branch off `master`:
- `phase0/critical-fixes`
- `phase1/foundation`
- `phase2/ux-a11y`
- `phase3/features`
- `phase4/code-quality`
- `phase5/perf-infra`
- `phase6/testing-ci`

Create a branch per phase, commit frequently (the Phase 0 plan commits after each task), and open
a PR per phase for review.

---

## File structure overview

New files created across the whole effort (grouped by responsibility):

**Types & API (`lib/`):**
- `lib/types/api.ts` — shared request/response interfaces + `User` (single source of truth)
- `lib/errors.ts` — `getErrorMessage(error: unknown)`, `isApiValidationError`
- `lib/session.ts` — single token-storage module (replaces scattered `localStorage` access)
- `lib/api.ts` — add `examAPI`, `subjectAPI`, `collectionAPI`, `materialAPI`, `documentAPI`,
  `flashcardAPI`, `accountAPI`, `adminAPI`, `notificationsAPI`; delete `authAPI`; type every method

**Data fetching (`hooks/`):**
- `hooks/use-documents.ts`, `hooks/use-mock-tests.ts`, `hooks/use-chat-sessions.ts`,
  `hooks/use-analytics.ts` — SWR/React Query wrappers

**Shared UI (`components/`):**
- `components/auth/auth-split-layout.tsx` — shared login/signup shell (de-duplicates + fixes hydration)
- `components/ui/chart-tooltip.tsx` — shared recharts tooltip
- `components/dashboard/test/analysis-tab.tsx`, `mock-test-tab.tsx`, `mock-test-list.tsx`,
  `analysis-result-view.tsx` — split of `test/page.tsx`
- `components/dashboard/teacher/roster.tsx`, `student-detail.tsx`, `assign-test-section.tsx` —
  split of `teacher/page.tsx`
- `components/dashboard/quiz/submit-confirmation-dialog.tsx`
- `components/dashboard/empty-state.tsx` — shared empty-state component
- `components/dashboard/loading-skeleton.tsx` — shared skeleton

**Route infrastructure (`app/`):**
- `app/(dashboard)/{dashboard,test,analytics,teacher,chat}/loading.tsx` + `error.tsx` per segment
- `app/(dashboard)/test/history/page.tsx` — test history
- `app/(dashboard)/profile/[userId]/page.tsx` — public profile
- `app/(auth)/forgot-password/page.tsx`, `app/(auth)/terms/page.tsx`, `app/(auth)/privacy/page.tsx`
- `app/api/{onboarding,exams,...}/route.ts` (only if we keep the proxy approach — otherwise remove
  the `fetch` calls; see Phase 0 Task 0.12)

**Backend (new routers/services for features):**
- `Backend/src/routers/flashcard_router.py` + `services/flashcard_service.py`
- `Backend/src/routers/account_router.py` + `services/account_service.py`
- `Backend/src/routers/admin_router.py`
- `Backend/src/routers/notifications_router.py`

**Testing:**
- `vitest.config.ts`, `playwright.config.ts`, `tests/unit/`, `tests/e2e/`

---

## Phase summaries (task-level; full detail in each phase plan)

### Phase 0 — Critical correctness & broken-feature fixes
Detailed in [phase0 plan](./2026-06-19-orbit-frontend-phase0-critical-fixes.md). Tasks:
- 0.1 Fix auth-page `Math.random()` hydration mismatch (extract `<AuthSplitLayout>`)
- 0.2 Remove/implement dead auth links (`/forgot-password`, `/terms`, `/privacy`)
- 0.3 Wire Settings "Save Changes" (account update) or mark Coming soon
- 0.4 Fix `/test` double-active sidebar (include `?tab=` in active detection)
- 0.5 Sign-out via `useAuth().logout()`; delete legacy `header.tsx`
- 0.6 Quiz submit confirmation dialog + timer warnings + `beforeunload` guard
- 0.7 Results: API-first, keep sessionStorage as cache (don't delete on read)
- 0.8 Flashcards: mark as beta / hide generate (real impl in Phase 3)
- 0.9 Admin: hide for non-superusers (real impl in Phase 3)
- 0.10 Landing CTAs → `/signup`
- 0.11 Surface silent errors via toast (`getErrorMessage` helper introduced here)
- 0.12 Resolve missing `app/api/*` routes (switch `dashboard-context`/onboarding/exam-setup to the
  typed axios `api` instance; add `examAPI`/`onboardingAPI`)

### Phase 1 — Foundation (types, data fetching, errors)
- 1.1 Set up Vitest + Testing Library + `lib/errors.ts` + `lib/session.ts`
- 1.2 `lib/types/api.ts` + single `User`; type every `lib/api.ts` method return; delete duplicate `User`
- 1.3 Type SSE chunk payload; type `convertApi*` in `lib/data.ts`
- 1.4 Add SWR (or React Query) + `useDocuments`/`useMockTests`/`useChatSessions`/`useAnalytics` hooks
- 1.5 Add per-segment `loading.tsx` + `error.tsx`; convert list pages toward Server Components
- 1.6 Unify auth: delete `authAPI`, route all auth through `auth-context.tsx` + `lib/session.ts`
- 1.7 Add `<RouteErrorBoundary>` per segment; replace `catch (error: any)` with `getErrorMessage`

### Phase 2 — UX & UI polish, accessibility, theme
- 2.1 `prefers-reduced-motion` across `animated-counter`, `orbiting-circles`, `particles`,
  `testimonials-carousel`, `hero`, framer `Container`
- 2.2 Keyboard-accessible exam presets + document dropzone (real buttons + validation)
- 2.3 Focus rings + ARIA: accordion, mobile menu, AppShell buttons, password toggle, nav labels
- 2.4 Real empty states for analytics (replace fabricated data)
- 2.5 Fix "Tests Taken" stat semantics
- 2.6 Theme: `.dark` strategy, replace hardcoded colors with tokens, accent picker sets
  `--ring`/`--accent`/`--chart-1`, remove dead media query
- 2.7 Onboarding overlay + step indicators (real spinner, `aria-current`, labels)
- 2.8 Chat interface: clear session on doc change, remove debug logs, stop-generation button, drop
  local-only reactions (or persist)
- 2.9 Quiz palette responsive
- 2.10 Landing: vary/remove 5-star ratings, remove misleading `hover:pause`, gate hero motion,
  fix dead Contact link

### Phase 3 — Missing features
- 3.1 Real flashcards (backend router + Gemini + frontend deck/review)
- 3.2 Persistent settings & account management (`accountAPI` + profile/password/delete/export)
- 3.3 Test history list + detail
- 3.4 Revision / weakness-driven study flow
- 3.5 Progress-over-time (replace `fallbackWeekly` with real activity log)
- 3.6 Streaks / gamification / achievements
- 3.7 Notifications (backend + bell + wire Settings toggles)
- 3.8 Study planner / schedule / calendar
- 3.9 Reminders (on top of notifications)
- 3.10 Export (results/analytics/chat → PDF/CSV)
- 3.11 Global search (cmdk command palette)
- 3.12 Public profile page `/profile/[userId]`
- 3.13 Onboarding expansion (preset selection, preferences, first-test prompt)
- 3.14 Teacher nav link in sidebar
- 3.15 Material upload placeholder wired
- 3.16 Typed clients + management UI for exam/subject/collection/material/document

### Phase 4 — Code quality & best practices
- 4.1 Remove `any`; add `@typescript-eslint/no-explicit-any` + `noUncheckedIndexedAccess`; shared
  `<ChartTooltip />`
- 4.2 Split `test/page.tsx` (1075 lines) + `teacher/page.tsx` (706 lines)
- 4.4 Delete dead code: unused `header.tsx`, `DEFAULT_*` mock data, dead CSS media query, misleading
  `hover:pause`, `// ...existing code...` comment
- 4.5 Re-enable `next/image`; replace raw `<img>`
- 4.7 Move tag-parsing into `convertApiDocumentToDocument`
- 4.8 Stop leaking setters from `DashboardContext` (intention-based actions)
- 4.9 Sidebar close-button semantics (`onClose`)
- 4.10 Adopt or remove unused deps (RHF+zod in auth/onboarding; align eslint-config-next,
  typescript; pick one toast system)
- 4.11 Guard URL-derived union types (validate `tab` param)

### Phase 5 — Performance & infrastructure
- 5.1 Decide `output: 'export'` vs Node server; align `deploy` script
- 5.2 Code-split heavy bundles (`next/dynamic` for PDF viewer + recharts; split test/teacher per tab)
- 5.3 Scope custom scrollbar styles to `body`
- 5.4 Gitignore `tsconfig.tsbuildinfo`, `.next/`, `.DS_Store`

### Phase 6 — Testing & CI
- 6.1 Vitest + Testing Library (contexts, hooks, `getErrorMessage`, conversion utils)
- 6.2 Playwright E2E (auth flow + one happy path per dashboard route)
- 6.3 CI config (lint + build + tests on push/PR)

---

## Self-review

**Spec coverage:** Every item in `Frontend/IMPROVEMENT_PLAN.md` (Phases 0–6) maps to a phase task
above. Phase 0 is fully detailed in its own file; Phases 1–6 are listed at task granularity and will
be expanded into full TDD plans before their execution.

**Placeholder discipline:** This master plan deliberately uses task *lists* (not code steps) for
Phases 1–6 because the per-step code lives in each phase's dedicated plan, generated on demand.
Phase 0 contains no placeholders — every step has exact paths, commands, and complete code.

**Execution handoff:** Start with the [Phase 0 plan](./2026-06-19-orbit-frontend-phase0-critical-fixes.md).
When a phase completes and merges, generate the next phase's detailed plan (ask the assistant to
"write the Phase N plan using writing-plans"), then execute it.