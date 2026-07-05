# Orbit Frontend — Improvement Plan

A consolidated, prioritized plan covering UX, UI, missing features, and code-quality best
practices. Derived from a four-axis audit (architecture, UX/UI, feature gaps, code quality).

> Stack: Next.js 16 (App Router) · React 19 · TypeScript · Tailwind · Radix/shadcn · axios ·
> framer-motion · recharts · react-pdf. Everything is currently `"use client"` with
> `useEffect`-based fetching. Zero tests.

---

## How to read this plan

- **Priority**: P0 (correctness bugs / broken features) → P3 (hygiene).
- **Effort**: S (< 1 day) · M (1–3 days) · L (3–5 days+).
- Each item lists **the problem**, **the fix**, and **target files** (relative to `Frontend/`).
- Items are grouped into phases that should be executed roughly in order — Phase 1 unblocks the
  rest by establishing a typed API layer and data-fetching cache.

---

## Phase 0 — Critical correctness & broken-feature fixes (do first)

These are bugs that actively harm users or misrepresent the product today.

### 0.1 Fix `Math.random()` hydration mismatch on auth pages — P0 · S
- **Problem**: `app/(auth)/login/page.tsx` and `app/(auth)/signup/page.tsx` generate star positions
  with `Math.random()` inside render (server components). Server/client differ → hydration
  warnings + flicker.
- **Fix**: Extract a `<AuthSplitLayout>` client component (also de-duplicates the two pages — see
  4.6). Generate the starfield once with `useMemo` after mount, or seed deterministically by index.
  Mark decorative layers `aria-hidden`.
- **Files**: `app/(auth)/login/page.tsx`, `app/(auth)/signup/page.tsx` (new
  `components/auth/auth-split-layout.tsx`).

### 0.2 Remove dead auth links that 404 — P0 · S
- **Problem**: "Forgot password?" → `/forgot-password`, "Terms" → `/terms`, "Privacy" → `/privacy`
  have no pages; users hit the 404.
- **Fix**: Either implement the pages (recommended for Terms/Privacy — cheap static pages) or
  remove the links. For forgot-password, ship a disabled "Coming soon" affordance.
- **Files**: `components/auth/auth-form.tsx` (lines 184, 229, 233).

### 0.3 Settings "Save Changes" saves nothing — P0 · M
- **Problem**: `handleSave` only toasts; profile/password/notification toggles silently revert on
  reload. Users are misled. The `Lock` icon on a Save button is also semantically wrong.
- **Fix**: Wire to a profile-update endpoint (see feature #3 / `accountAPI`). Until the backend
  exists, disable the button and label it clearly. Swap icon to `Save`.
- **Files**: `app/(dashboard)/settings/page.tsx` (lines 60–65).

### 0.4 Fix `/test` double-active sidebar item — P0 · S
- **Problem**: Analysis (`/test?tab=analysis`) and Mock Tests (`/test?tab=mock`) both strip to
  basePath `/test`; `isActive = pathname === basePath` lights both up.
- **Fix**: Include the `?tab=` query param in active detection, or split into distinct routes.
- **Files**: `components/dashboard/app-shell.tsx` (lines 38–42, 60–61).

### 0.5 Sign-out bypasses auth context — P0 · S
- **Problem**: `handleSignOut` only clears `localStorage["token"]` + `router.push("/")`. It never
  calls `useAuth().logout()`, so `user` state stays stale, and it lands on the marketing page
  instead of `/login`.
- **Fix**: Call `logout()` from `useAuth()`. Delete the duplicate logic in the legacy
  `components/dashboard/header.tsx` (which is itself unused — see 4.4).
- **Files**: `components/dashboard/app-shell.tsx` (lines 113–116), `lib/context/auth-context.tsx`.

### 0.6 Quiz: no submit confirmation, no warnings, no unload guard — P0 · M
- **Problem**: Submit fires immediately (no "are you sure?", no unanswered-count). Timer auto-submit
  has no 5-min/1-min warning. No `beforeunload` guard → progress lost on accidental nav.
- **Fix**: Add a confirmation `Dialog` (use existing component) showing answered/unanswered counts;
  add warning toasts at 5 min and 1 min; add a `beforeunload` listener while a test is in progress.
- **Files**: `app/(dashboard)/test/quiz/page.tsx` (lines 71–74, 103–131, 172–184).

### 0.7 Results depend on volatile `sessionStorage` — P0 · S
- **Problem**: `testAnalysis` is stashed in `sessionStorage` and removed on read; refresh / new tab /
  back button loses it.
- **Fix**: Make the API call (`getAnalysisBySubmissionId`) the primary source; keep
  `sessionStorage` as a cache and don't delete it on read.
- **Files**: `app/(dashboard)/test/quiz/page.tsx` (line 117), `app/(dashboard)/test/results/page.tsx`
  (lines 48–56).

### 0.8 Flashcards is a non-functional stub — P1 · M
- **Problem**: `flashcards/page.tsx` ships only `DEMO_DECKS`; "Generate Deck" toasts "coming soon".
  No backend flashcard router exists.
- **Fix**: Add a backend flashcard router + AI generation, then wire the page. Interim: clearly mark
  as beta and hide the generate action.
- **Files**: `app/(dashboard)/flashcards/page.tsx`, new `Backend/src/routers/flashcard_router.py` +
  service, `lib/api.ts` (`flashcardAPI`).

### 0.9 Admin dashboard is an empty scaffold — P1 · M
- **Problem**: `admin/page.tsx` hardcodes `users: []`; stat cards always 0; "requires backend admin
  endpoints".
- **Fix**: Add admin backend endpoints (user list/role management/audit) + wire the table, or hide
  the page for non-superusers.
- **Files**: `app/(dashboard)/admin/page.tsx`, new `Backend/src/routers/admin_router.py`.

### 0.10 Landing CTAs send logged-out users to `/dashboard` — P1 · S
- **Problem**: "Get Started" / "Start Learning Free" → `/dashboard` → `AuthProtection` → `/login`
  (a confusing double hop).
- **Fix**: Point primary CTAs at `/signup`; keep `/dashboard` for the in-app "Go to dashboard"
  case.
- **Files**: `components/marketing/navbar.tsx` (line 39), `components/marketing/hero.tsx` (line 78).

### 0.11 Silent error handling across the app — P1 · S
- **Problem**: Exam creation, onboarding, dashboard stats, and analytics failures are `console.error`
  only — user clicks, nothing visibly happens.
- **Fix**: Surface failures with `useToast({ variant: "destructive" })`; reset loading state
  visibly. Provide the shared `getErrorMessage(error: unknown)` helper (see 1.5).
- **Files**: `components/dashboard/exam-setup-dialog.tsx`, `components/onboarding/onboarding-container.tsx`,
  `app/(dashboard)/dashboard/page.tsx`, `app/(dashboard)/analytics/page.tsx`.

### 0.12 Missing `app/api/*` routes break onboarding & exam creation — P0 · M
- **Problem**: `dashboard-context.tsx`, `onboarding-container.tsx`, `exam-setup-dialog.tsx`, and
  `(dashboard)/layout.tsx` call `fetch("/api/onboarding")`, `fetch("/api/exams")`, etc. There is no
  `app/api/` directory and no `rewrites` proxy in `next.config.js` — these 404 against the Next
  server, and `fetch` does **not** attach the Bearer token the way the axios instance does.
- **Fix**: Decide on one mechanism: (a) add Next.js route handlers that proxy to the FastAPI backend
  with the token, or (b) replace these `fetch` calls with typed methods on the shared `api`
  instance (preferred — keeps auth + types centralized). Align base-URL conventions.
- **Files**: `lib/context/dashboard-context.tsx` (line 82 et al.), `components/onboarding/onboarding-container.tsx`,
  `components/dashboard/exam-setup-dialog.tsx`, `app/(dashboard)/layout.tsx`, `next.config.js`.

---

## Phase 1 — Foundation: types, data fetching, error handling

This phase unblocks safer feature work by removing the `any`-everywhere / fetch-everywhere
pattern.

### 1.1 Add a typed API layer — P0 · M
- **Problem**: `lib/api.ts` has zero return types; every method returns implicit `any`. Callers cast
  with `as any[]` / `as User`. Domain types in `lib/data.ts` aren't used by the API layer. `zod` is
  installed but unused.
- **Fix**: Create `lib/types/api.ts` with request/response interfaces (or zod schemas) for every
  endpoint. Type each `api.ts` method's return (use `api.post<LoginResponse>(...)`). Type the SSE
  chunk payload (`onChunk?: (chunk: StreamChunk) => void`).
- **Files**: new `lib/types/api.ts`, `lib/api.ts`, `lib/data.ts` (`convertApi*` accept typed input).

### 1.2 Consolidate the `User` type — P0 · S
- **Problem**: `User` is defined in both `lib/data.ts:11` and `lib/context/auth-context.tsx:24` and
  they've already drifted (`role`/`preferred_language` optional in one, required in the other).
- **Fix**: Single `User` type in `lib/types/api.ts`; import everywhere.

### 1.3 Introduce a data-fetching cache (React Query or SWR) — P1 · L
- **Problem**: No cache/dedup. `pdfAPI.listPDFs()` is called from 4+ components; `mockTestAPI.listMockTests()`
  from 4; `chatAPI.listChatSessions()` from 2. Each causes redundant requests and inconsistent error
  handling.
- **Fix**: Add SWR/React Query with hooks `useDocuments()`, `useMockTests()`, `useChatSessions()`,
  `useAnalytics()`. Centralized error handling + stale-while-revalidate. Removes duplication and
  the hand-rolled `useEffect` fetch pattern.
- **Files**: new `hooks/use-*.ts`, refactor list pages.

### 1.4 Convert list pages toward Server Components + `loading.tsx`/`error.tsx` — P1 · L
- **Problem**: 84 files are `"use client"`; every dashboard route fetches in `useEffect`, throwing
  away App Router streaming/RSC/metadata and creating data-fetch waterfalls. No `loading.tsx`
  anywhere; only top-level `app/error.tsx`.
- **Fix**: Convert list-fetching pages to Server Components (or Server Actions) with Suspense +
  per-segment `loading.tsx`; keep interactive bits as small client islands. Add `error.tsx` per
  route segment. (If full RSC is too big a jump, at minimum add `loading.tsx`/`error.tsx` per
  segment as a quick win.)
- **Files**: `app/(dashboard)/{dashboard,test,analytics,teacher,chat}/...`, new `loading.tsx` +
  `error.tsx` per segment.

### 1.5 Centralize error handling + remove debug logs — P1 · S
- **Problem**: `catch (error: any) { if (error.response?.data?.detail) ... }` copy-pasted across
  pages. 37 `console.log/warn/error` calls including debug logs left in production paths
  (`test/page.tsx:142-146`, `test/results/page.tsx:61`, `chat-interface.tsx:67`,
  `chat-history-viewer.tsx:53`, `material-list.tsx:25`).
- **Fix**: `getErrorMessage(error: unknown)` helper using `axios.isAxiosError`. Replace `catch (error: any)`
  with `catch (error)`. Strip debug logs or gate behind `process.env.NODE_ENV !== 'production'`.
  Add a `<RouteErrorBoundary>` per segment.
- **Files**: new `lib/errors.ts`, all pages with `catch`.

### 1.6 Unify the auth code path — P1 · S
- **Problem**: `authAPI` in `lib/api.ts` (lines 40–76) re-implements login/signup/logout/getMe with
  direct `localStorage`, parallel to `auth-context.tsx`'s `login/signup/logout` (the one actually
  wired to the UI). They've drifted (e.g. `authAPI.login` doesn't redirect; context does).
- **Fix**: Delete `authAPI` from `lib/api.ts` (or have the context call it). Centralize token
  storage behind a single `session` module.

---

## Phase 2 — UX & UI polish, accessibility, theme

### 2.1 Add `prefers-reduced-motion` support — P1 · M
- **Problem**: Zero `prefers-reduced-motion`/`motion-reduce` usage anywhere. Always-animating
  components ignore the OS setting: `animated-counter`, `orbiting-circles`, `particles`,
  `testimonials-carousel`, `hero` (orbiting + flip/rotate badge), framer `Container` fade-ins.
- **Fix**: Gate all motion behind `useReducedMotion` (framer-motion) or a media query. Counters
  snap to final value; carousels stop auto-scroll; orbit/particles render a static fallback.

### 2.2 Keyboard accessibility on custom interactive elements — P1 · S
- **Problem**: Exam presets in `exam-setup-dialog.tsx` are `<div onClick>` with no role/tabIndex/
  keyboard handler. Document dropzone in `document-uploader.tsx` same issue. MagicCard preset
  selector unusable by keyboard/SR.
- **Fix**: Use real `<button>`s, or add `role="button"` + `tabIndex={0}` + `onKeyDown` for
  Enter/Space. Add file size/type validation on the dropzone ("max 10MB" helper is unenforced).
- **Files**: `components/dashboard/exam-setup-dialog.tsx`, `components/dashboard/documents/document-uploader.tsx`.

### 2.3 Fix focus rings & ARIA gaps — P1 · S
- **Problem**: `theme-toggle.tsx` removes focus ring with no replacement; `faq-accordion.tsx` uses
  `focus:outline-none` with nothing back; accordion trigger lacks `aria-expanded`/`aria-controls`,
  panel has no `id`/`role="region"`, and `max-h-48` clips long answers. Mobile menu trigger lacks
  `aria-label`; mobile menu doesn't close on navigation. AppShell mobile buttons lack labels; nav
  regions unlabeled. Password show/hide button has no `aria-label`.
- **Fix**: Restore visible `focus-visible:ring-2 focus-visible:ring-ring`. Add ARIA linkage to
  accordion; switch to `grid-rows-[0fr/1fr]` height transition; remove the auto-open effect that
  fights user interaction. Add `aria-label`s; wrap mobile nav links in `SheetClose`.

### 2.4 Replace fake analytics fallbacks with real empty states — P1 · S
- **Problem**: `analytics/page.tsx` shows fabricated Math/Physics/Chemistry/Biology scores, weakness
  topics, `ProgressRing value={72}` "Completion", `value={85}` "Consistency", "5 days" streak,
  hardcoded `"3"`/`"12%"` trends. New users see a populated dashboard that misrepresents reality.
- **Fix**: Render explicit empty states ("No analytics yet — take your first test to see insights").
  Only show trends when computed from real data.
- **Files**: `app/(dashboard)/analytics/page.tsx` (lines 56–68, 95–132).

### 2.5 Fix misleading "Tests Taken" stat — P2 · S
- **Problem**: `dashboard/page.tsx` sets `testsTaken: testList.length` (counts all mock test records,
  not submissions).
- **Fix**: Rename to "Mock Tests" or count only tests with `latest_submission`.

### 2.6 Theme system: pick one strategy; make light mode actually work — P1 · M
- **Problem**: `:root` defines dark tokens; `.light` overrides; a dead `prefers-color-scheme` block
  sets identical dark values. Tailwind `dark:` variants look for `.dark`, but overrides use
  `.light` — inconsistent. Hardcoded colors (`bg-[#0D1520]`, `text-green-600`, etc.) break light
  mode. Accent picker only sets `--primary` (leaves `--ring`/`--accent`/`--chart-1`).
- **Fix**: Move dark tokens to `.dark`, keep `:root` light defaults (matches `darkMode: ['class']` +
  next-themes). Replace hardcoded colors with theme tokens (`text-destructive`, `bg-secondary`,
  etc.). Accent picker should also set `--ring`, `--accent`, `--chart-1`. Remove the dead media
  query block.
- **Files**: `app/globals.css`, auth pages, `test/results/page.tsx`, `settings/page.tsx`.

### 2.7 Onboarding overlay + step indicators broken — P2 · S
- **Problem**: Loading overlay is `absolute inset-0` with no `relative` ancestor; "spinner" is a
  static `bg-primary` circle (no animation). Step indicators are unlabeled dots with no
  `aria-current`.
- **Fix**: Add `relative` to the card; use a real spinner (`Loader2` + `animate-spin`); label steps.

### 2.8 Chat interface fixes — P2 · S
- **Problem**: Leftover `console.log`; unreachable empty-state (system message always inserted);
  stale `chatSession` reused when document changes (effect at line 48 doesn't clear it); no "Stop
  generation" during streaming; reactions are local-only and noisy.
- **Fix**: Clear `chatSession` on document change; remove debug logs; persist reactions (or drop
  the feature); add a stop button during streaming.
- **Files**: `components/dashboard/chat/chat-interface.tsx`.

### 2.9 Quiz palette responsive + quiz UX — P2 · S
- **Problem**: `grid grid-cols-10` cramps 40px buttons on small screens / long tests.
- **Fix**: `grid-cols-5 sm:grid-cols-10` or `flex flex-wrap`.

### 2.10 Landing polish — P2 · S
- **Problem**: Testimonials all 5-star (inauthentic); `hover:pause` is not a real Tailwind class
  (pause works only via CSS); hero has three simultaneous infinite animations; "Contact" link is
  `#` (dead).
- **Fix**: Vary/remove ratings; remove the misleading class; gate hero animations behind
  reduced-motion; implement or remove Contact.

---

## Phase 3 — Missing features

Grouped by theme; sizes are rough.

### 3.1 Real flashcards (generate, review, spaced repetition) — M
- Backend flashcard router + Gemini generation from a material/collection. Frontend deck list,
  card flip UI, "mark known/unknown", basic SM-2-style scheduling. (Un-stubs 0.8.)

### 3.2 Persistent settings & account management — M
- Profile update (name, bio, avatar, language), password change, notification preferences,
  account deletion, data export (GDPR). Adds `accountAPI` consumed by Settings. (Un-stubs 0.3.)

### 3.3 Test history / past attempts list — S–M
- Browsable history page of all submissions with score, date, re-review. Replaces the single-
  submission `?testId=&submissionId=` view with a list + detail.

### 3.4 Revision / weakness-driven study flow — M
- Analytics already identifies weak topics (radar). Add "turn weak topics into a targeted study
  plan / revision deck / re-test" actions.

### 3.5 Progress tracking over time — M
- Replace `fallbackWeekly` mock in analytics with a real activity log endpoint + persistence. Streaks
  require this (3.6).

### 3.6 Streaks / gamification / achievements — M
- `analytics/page.tsx` imports `Flame` but no streak is tracked. Add a streak counter (days studied),
  XP, badges. Depends on 3.5.

### 3.7 Notifications — L
- Backend notifications + in-app bell (currently commented out in `header.tsx`). Wire the three
  notification toggles in Settings. Covers reminders (3.9) too.

### 3.8 Study planner / schedule / calendar — L
- Planner UI with due dates for assigned tests, study sessions. Uses `react-day-picker` +
  `date-fns` (already installed, unused).

### 3.9 Reminders — L
- Tied to notifications (3.7): assignment due, test pending, streak at risk.

### 3.10 Export (results, analytics, chat transcripts) — S
- Client-side PDF/CSV export; the analysis report already has a "Download Report" button — extend
  to results/analytics/chat.

### 3.11 Global search — M
- Currently cosmetic. Add search across chats, tests, flashcards, results. `cmdk` is installed
  and unused — wire a command palette (`Cmd+K`).

### 3.12 Public profile page — S–M
- Settings has a "Public Profile" switch but no `/profile/[userId]` route. Add it.

### 3.13 Onboarding expansion — S–M
- Currently 2 steps. Add exam/subject preset selection, preferences (notifications/dark-mode
  opt-in), sample content upload, first-test prompt.

### 3.14 Teacher nav link — S (quick win)
- `app-shell.tsx` adds Admin for admins but never `/teacher` for teachers — Teacher dashboard is
  URL-only today.

### 3.15 Material upload placeholder — S (quick win)
- `components/dashboard/material-list.tsx:24` has `// Placeholder for upload logic`. Wire it.

### 3.16 Typed API clients for exam/subject/collection/material/document — M
- Backend routers exist (`exam_router`, `subject_router`, `collection_router`,
  `material_router`, `document_router`) but are consumed only via ad-hoc `fetch` in
  `dashboard-context.tsx`. Add `examAPI`/`subjectAPI`/`collectionAPI`/`materialAPI`/`documentAPI`
  exports + management UI (Manage Exams, Edit Subject, material tagging). Pairs with 0.12.

---

## Phase 4 — Code quality & best practices

### 4.1 Remove `any` everywhere; enable lint rules — P1 · M
- ~30 `any` usages; `test/page.tsx` uses `any` 8×. Four near-identical recharts `CustomTooltip`
  copies.
- **Fix**: Type everything (uses 1.1). Add `@typescript-eslint/no-explicit-any` +
  `no-unsafe-assignment`. Enable `noUncheckedIndexedAccess` (keep `strict: true`). Extract one
  shared `<ChartTooltip />`.

### 4.2 Split monolith pages — P1 · M
- `test/page.tsx` is 1075 lines (analysis + mock tabs, duplicated upload UI ~lines 406 & 950,
  debug logs at 142–146). `teacher/page.tsx` is 706 lines.
- **Fix**: `test/page.tsx` → `<AnalysisTab/>`, `<MockTestTab/>`, `<TeacherControls/>`,
  `<MockTestList/>`, `<AnalysisResultView/>`, plus a `useTestPageState()` hook/reducer.
  `teacher/page.tsx` → roster, detail slide-over, assign-test section.

### 4.3 Cache/dedup data fetching — (covered by 1.3)

### 4.4 Delete dead/duplicate code — P2 · S
- Unused `components/dashboard/header.tsx` (duplicates sign-out, has commented notifications).
- `lib/data.ts` ships `DEFAULT_DOCUMENTS`/`DEFAULT_MESSAGES`/`DEFAULT_CHAT_HISTORY` used nowhere in
  production.
- Dead `prefers-color-scheme` block in `globals.css` (lines 11–17). Misleading `hover:pause` class.
- `chat-interface.tsx:194` has a literal `// ...existing code...` comment (AI-generated, never
  cleaned).

### 4.5 Re-enable `next/image` optimization; replace raw `<img>` — P2 · S
- `next.config.js` sets `images.unoptimized: true`. Logos use raw `<img>` in auth pages, navbar,
  footer. Only `app-shell.tsx` and `hero.tsx` use `next/image`.
- **Fix**: Remove `unoptimized: true` (unless static export — see 5.1) and replace `<img>` with
  `<Image>` (with explicit width/height).

### 4.6 Extract shared `AuthSplitLayout` — P2 · S
- login/signup are near-identical copy-paste (also fixes 0.1's hydration issue). Memoize the
  starfield.

### 4.7 Move tag-parsing into the conversion helper — P2 · S
- `sidebar.tsx` reimplements JSON-string-in-array tag parsing at every call site; belongs in
  `convertApiDocumentToDocument` once.

### 4.8 Context API: stop leaking setters — P3 · S
- `DashboardContext` exposes `setExams`/`setActiveExam` directly. Expose intention-based actions
  (`createExam`, `selectExam`) instead.

### 4.9 Sidebar close-button semantics — P3 · S
- `sidebar.tsx:158-165` close `X` calls `onSelectDocument(...)` — a close button that selects a
  document. Should be `onClose`.

### 4.10 Adopt or remove unused deps — P3 · S
- `react-hook-form`, `@hookform/resolvers`, `zod`, `input-otp`, `react-day-picker`, `cmdk`,
  `sonner` installed but unused (sonner vs shadcn toaster duplicated). `eslint-config-next@13.5.1`
  and `@next/swc-wasm-nodejs@13.5.1` mismatch `next@^16`. `typescript@5.2.2` old.
- **Fix**: Adopt RHF + zod in `auth-form.tsx` (which rolls its own) and onboarding; align
  eslint-config-next → 16, typescript → 5.4+. Pick one toast system. `cmdk` feeds 3.11.

### 4.11 Guard URL-derived union types — P3 · S
- `test/page.tsx:34` casts `activeTabFromUrl as 'analysis' | 'mock'` with no validation; bad URL →
  bad state silently. Validate with zod or a guard.

---

## Phase 5 — Performance & infrastructure

### 5.1 Decide export vs Node server — P1 · S
- `output: 'export'` is commented out, but `deploy` runs `npx serve out` (expects static export).
  If Node server, `serve out` is wrong; if static export, `output: 'export'` must be on and RSC
  data fetching / image optimization constraints apply. Pick one and align.
- **Files**: `next.config.js`, `package.json` (`deploy` script).

### 5.2 Code-split heavy client bundles — P2 · M
- `react-pdf`/`pdfjs-dist`, `recharts`, `framer-motion`, `vaul`, `embla` all client-bundled.
  `test/page.tsx` (1075 lines) and `teacher/page.tsx` (706) ship as single chunks.
- **Fix**: `next/dynamic` for the PDF viewer and recharts wrappers; code-split test/teacher per
  tab (pairs with 4.2).

### 5.3 Scope custom scrollbar styles — P3 · S
- `globals.css` `::-webkit-scrollbar` overrides apply to every scroll region including small
  `ScrollArea` viewports. Scope to `body`.

### 5.4 Gitignore build artifacts — P3 · S
- `tsconfig.tsbuildinfo` (197 kB) is committed. Add `tsconfig.tsbuildinfo`, `.next/`, `.DS_Store`
  to `.gitignore`.

---

## Phase 6 — Testing & CI

### 6.1 Add Vitest + Testing Library — P2 · M
- Zero tests, no runner. Cover contexts (`auth-context`, `dashboard-context`) and hooks
  (`use-toast`), the `getErrorMessage` helper, and conversion utilities.

### 6.2 Add Playwright E2E — P2 · M
- Cover the auth flow (login/signup/onboarding) and one happy path per dashboard route
  (dashboard, test create → quiz → results, analytics, teacher).

### 6.3 Add CI config — P2 · S
- Run `npm run lint`, `npm run build`, and tests on push/PR.

---

## Suggested execution order

1. **Phase 0** — critical bugs (0.1–0.12). Big trust/credibility wins, mostly S/M.
2. **Phase 1** — foundation (types 1.1–1.2, error helper 1.5, auth unify 1.6). Unblocks safe feature
   work. Cache/RSC (1.3–1.4) is the longer L item; can land incrementally.
3. **Phase 2** — UX/UI/a11y polish (2.1–2.10), interleaved with quick feature wins from Phase 3
   (3.10 export, 3.14 teacher nav, 3.15 material upload).
4. **Phase 3** — feature build-out, prioritized by user value: 3.1 flashcards, 3.2 settings/account,
   3.3 history, 3.4 revision, 3.5 progress, then gamification/planner/notifications.
5. **Phase 4** — code quality (4.1–4.11) largely falls out of Phases 1–3; tackle remaining items
   (4.2 split, 4.4 delete dead code, 4.5 images) as cleanup passes.
6. **Phase 5 & 6** — perf infra + testing; add tests alongside each Phase 3 feature rather than
   afterwards.

## Quick-win checklist (highest impact, lowest effort — do these first)

- [ ] 0.1 auth hydration mismatch
- [ ] 0.2 dead auth links
- [ ] 0.4 `/test` double-active sidebar
- [ ] 0.5 sign-out via `useAuth().logout()`
- [ ] 0.7 results: API-first, keep sessionStorage cache
- [ ] 0.10 landing CTAs → `/signup`
- [ ] 2.4 real analytics empty states
- [ ] 3.14 teacher nav link
- [ ] 3.15 material upload placeholder wired
- [ ] 4.4 delete unused `header.tsx` + dead `globals.css` media query
- [ ] 4.5 / 5.4 images + gitignore