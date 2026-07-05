# Phase 1 — Sellable Backbone (Subscriptions, Multi-tenant, Admin)

**Date:** 2026-07-06
**Branch:** `phase1/sellable-backbone` (created from `master`, normal branch — not a worktree)
**Status:** Design approved; pending spec review

## Goal

Make Orbit billable and multi-tenant so it can be sold via subscriptions to individual students worldwide and to Indian coaching centers, schools, and tuition classes (B2B seat licenses). This is the first of five phases toward the full 2026 ed-tech feature set.

**What this phase is NOT:** pedagogy features (Socratic tutoring, step-by-step, rubric grading, adaptive difficulty), educator ops (lesson planning, early-warning, AI dashboards), AI multilingual, focus mode, and multimodal input are explicitly deferred to later phases (see Roadmap). Existing features are enhanced only where needed to support billing/enforcement — none are removed.

## Non-goals (deferred)

- Phase 2: Socratic/guided tutoring, structured step-by-step explanations, rubric-aligned descriptive grading, adaptive-difficulty heuristics + per-skill logging.
- Phase 3: Lesson/curriculum auto-planning (standards-grounded), AI-summarized teacher dashboards, early-warning dropout flags, parent-facing summaries.
- Phase 4: AI multilingual (thread `preferred_language` through every Gemini prompt; English UI).
- Phase 5: Focus/scope-locked study mode, photo/handwriting OCR input, optional text roleplay.

Trendy/low-evidence items from the market report (metaverse avatars, leaderboards-as-primary mechanic, generic no-pedagogy chatbot) are not built.

## Decisions (locked)

1. **First slice = sellable backbone** (subscriptions + multi-tenant + admin/sub-admin + enforcement).
2. **Payments = Razorpay India-first** (UPI, cards, netbanking, recurring mandates). Global students handled via Razorpay international-card support where available; a Stripe add-on is a later-phase option if global demand requires it.
3. **Languages = AI multilingual + English UI**, designed in now but built in Phase 4. UI stays English in Phase 1.

## Phase roadmap (context)

1. **Phase 1 — Sellable backbone** (this spec).
2. Phase 2 — Pedagogy core.
3. Phase 3 — Educator ops.
4. Phase 4 — AI multilingual.
5. Phase 5 — Focus + input.

---

## 1. Plans and limits

Wire the existing marketing pricing (Starter/Pro/Premium in `lib/constants/plans.ts` and the landing Pricing section) to real backend-enforced limits.

| Plan | Price | Mock tests/mo | Flashcards/mo | AI materials/mo | Chat msgs/mo | Doc storage | Classes (teacher) |
|---|---|---|---|---|---|---|---|
| Starter | Free | 3 | 50 | 5 | 100 | 50 MB | 1 |
| Pro | ₹299/mo (₹2990/yr) | 50 | 500 | 50 | 1000 | 1 GB | 10 |
| Premium | ₹599/mo (₹5990/yr) | ∞ | ∞ | ∞ | ∞ | 10 GB | ∞ |
| Org seat | per-seat (Pro or Premium tier) | inherits tier | inherits tier | inherits tier | inherits tier | inherits tier | inherits tier |

Org members inherit the org's tier; they do not need individual subscriptions. A sub-admin/coaching owner pays for seats.

**Single source of truth:** plan limits live in backend constants (`core/plans.py`) and are mirrored in `Frontend/lib/constants/plans.ts`. The frontend reads limits from a `/subscriptions/plans` endpoint to avoid drift.

## 2. Data model (additive — nothing existing removed)

### New collections

**`subscriptions`**
- `user_id`, `plan` (`starter|pro|premium`), `status` (`active|past_due|cancelled|expired|trialing`), `source` (`self|org`), `billing_cycle` (`monthly|yearly`), `razorpay_subscription_id`, `razorpay_plan_id`, `current_period_start`, `current_period_end`, `cancel_at_period_end`, `started_at`, `updated_at`.

**`payments`**
- `user_id`, `razorpay_payment_id` (unique index, idempotency), `razorpay_subscription_id`, `razorpay_order_id`, `amount` (paise), `currency` (`INR`), `status` (`captured|failed|refunded`), `plan`, `billing_cycle`, `created_at`.

**`organizations`**
- `org_id`, `name`, `brand_name`, `owner_user_id` (the sub-admin), `tier` (`pro|premium`), `seats_total`, `seats_used` (denormalized count), `status` (`active|suspended|expired`), `expires_at`, `billing_cycle`, `razorpay_subscription_id`, `created_at`, `updated_at`.
- Unique index on `owner_user_id` (one org per sub-admin in v1).

**`org_invites`**
- `org_id`, `code` (unique), `member_role` (`teacher|student`), `email` (optional, if targeted), `created_at`, `expires_at`, `used_by_user_id`.

**`usage_events`** (only for resources not derivable from existing records — i.e. chat messages)
- `user_id`, `resource` (`chat_message`), `period_key` (`YYYY-MM`), `count`, `updated_at`. Upserted on each chat message. All other resources (mock tests, flashcards, ai_materials, doc storage, classes) are computed from existing collections on demand.

### User additions (existing `users` collection)

Add fields (all optional / backward-compatible):
- `org_id` (when the user belongs to an org).
- `member_role` (`teacher|student`) — role within the org (distinct from platform `role`).
- `org_joined_at`.

The existing `role`, `license_id`, and `subscription` fields are reused:
- `role` gains `subadmin` and `admin` as already-modeled values (onboarding currently restricts to `student|teacher`; we add a sub-admin onboarding path).
- `subscription` (embedded `SubscriptionInfo`) is kept as a denormalized cache of the active `subscriptions` record for fast `/auth/me` reads; the `subscriptions` collection is the source of truth.

## 3. Backend — new routers and services

### 3.1 `subscription_router` (`/subscriptions`)
- `GET /plans` — returns plan list + limits (public).
- `GET /me` — current user's effective plan, status, usage vs limits, invoices.
- `POST /checkout` — body `{plan, billing_cycle}` → creates Razorpay subscription (monthly) or one-time payment (yearly) → returns `{razorpay_subscription_id, razorpay_order_id?, key_id, amount, currency}`.
- `POST /verify` — body `{razorpay_payment_id, razorpay_subscription_id, razorpay_signature}` → verifies HMAC signature → records `payment` → activates `subscription` (idempotent on `razorpay_payment_id`).
- `POST /cancel` — Razorpay "cancel at period end"; local status updated optimistically and confirmed by webhook.
- `GET /invoices` — list of the user's `payments`.

Wraps a new **`billing_service`** (Razorpay SDK: `razorpay` python package) and **`subscription_service`** (db + plan logic). All endpoints require auth; reads are owner-scoped.

### 3.2 `webhook_router` (`/webhooks/razorpay`)
- `POST /` — raw-body HMAC verification with `RAZORPAY_WEBHOOK_SECRET`. Idempotent on Razorpay event id + `razorpay_payment_id`/`razorpay_subscription_id`.
- Handled events: `subscription.activated`, `subscription.charged`, `subscription.cancelled`, `subscription.expired`, `payment.failed`. Each updates the `subscriptions`/`payments` record and the denormalized `user.subscription` cache.
- Returns `200` on success, `400` on bad signature. No auth dependency (webhook is signature-authenticated).

### 3.3 `org_router` (`/orgs`)
- `POST /` — sub-admin creates org (name, brand_name, tier, seats_total, billing_cycle). Triggers Razorpay checkout for the seat subscription. Sets `role=subadmin`, `org_id`.
- `GET /me` — sub-admin's org + seat usage + members.
- `POST /invite` — body `{member_role, email?}` → creates `org_invites` code. If `email` provided and a user exists with that email, can auto-link on first login/accept.
- `POST /enroll/{code}` — student/teacher joins org (consumes a seat; checks `seats_used < seats_total` and `status=active` and not expired). Sets `org_id`, `member_role`.
- `GET /members` — paginated roster.
- `DELETE /members/{user_id}` — removes member, frees seat.
- `POST /seats` — add seats (Razorpay checkout for the delta; or upgrade tier).
- Role guard: `require_role("subadmin")` for owner endpoints; `enforce_org_seat` on enroll.

### 3.4 `admin_router` (`/admin`)
Role guard `require_role("admin")` on every endpoint.
- `GET /users` — list/filter by role/org/status; pagination.
- `PATCH /users/{id}/role` — change role.
- `PATCH /users/{id}/status` — activate/deactivate.
- `GET /orgs` — list orgs with seat usage, status, MRR contribution.
- `PATCH /orgs/{id}` — suspend/unsuspend, adjust `seats_total` (manual), extend `expires_at`.
- `GET /subscriptions` — list all subscriptions/payments.
- `POST /subscriptions/{id}/activate` — manual activation/extension (for offline/coaching cash deals).
- `GET /analytics` — platform KPIs: total users by role, active subscriptions, MRR (sum of active `payments` for current period), org count, conversion rate (paid / signed-up), recent signups.

### 3.5 `core/plan_enforcement.py` (new)
- `get_effective_plan(user_id) -> (plan, source, scope)` — resolves: active own subscription → `source=self`; else `org_id`'s org tier → `source=org`; else `starter`/`source=free`.
- `get_usage(user_id, resource) -> (used, limit)` — computes current-period usage from existing collections (mock tests, flashcards, ai_materials, doc storage, classes) and `usage_events` for chat messages.
- `enforce_limit(resource)` — FastAPI dependency: on the request's user, if `used >= limit` (and limit != ∞) → raise `HTTPException(402, detail={resource, used, limit, plan, upgrade_url})`. The 402 body is a clean upgrade payload, not a generic error.
- Pluggable per resource so Phase 2–5 features reuse it by adding a resource key + a usage counter.
- **Plan-gated feature flags** (Phase 2+): `require_plan("premium")` dependency for future high-tier features (early-warning, lesson planning, parent dashboards). Defined now, used later.

### 3.6 Enforcement points (where limits bite)
Add `enforce_limit(resource)` to:
- `/mock-tests/generate` → `mock_test`
- `/flashcards/generate` → `flashcard`
- `/ai-materials/summarize` → `ai_material`
- `/questions/ask` and `/questions/.../messages` (stream + non-stream) → `chat_message` (also writes a `usage_events` upsert)
- `/documents/upload` and `/api/collections/{id}/materials` → `doc_storage` (sum of uploaded sizes vs limit; reject if over)
- `/classes/` POST → `class_count` (teacher)

Org-seat enrollment gets `enforce_org_seat` (seats + expiry + status) on `POST /orgs/enroll/{code}`.

## 4. Frontend — new/changed pages

- **`/pricing`** (landing Pricing section becomes real): plan cards → `POST /subscriptions/checkout` → **Razorpay Checkout JS** (`<script src="https://checkout.razorpay.com/v1/checkout.js">`) → redirect to `/billing?status=success|cancel`. Yearly toggle for Pro/Premium.
- **`(dashboard)/billing`**: current plan + status, usage-vs-limit bars (mock tests / flashcards / AI materials / chat / storage / classes), invoices list, upgrade/cancel buttons, "manage org seats" panel for sub-admins.
- **`(dashboard)/admin`**: wire the existing scaffold to real `/admin/*` — users table (filter by role/org, change role, deactivate), orgs table (suspend, adjust seats, extend expiry), subscriptions table (manual activate), platform analytics cards + charts (reuse `recharts`).
- **`(dashboard)/org`** (sub-admin): org profile, seat usage meter, invite-link/code generator, member roster with remove, add-seats/upgrade-tier checkout.
- **Onboarding**: add a third path "Sign up my coaching / school / tuition" → creates `organization` + sets `role=subadmin` (currently restricted to `student|teacher`; the new path bypasses that restriction and creates the org record post-checkout). Student/teacher paths unchanged.
- **Upgrade prompts**: a shared `<UpgradeBanner>` component. When any API returns `402`, the calling page shows an inline upgrade card with the limit that was hit and a CTA to `/pricing`. A small axios interceptor maps `402` to a typed error so pages can render the banner without per-call handling.
- **`lib/api` additions**: `subscriptionAPI`, `orgAPI`, `adminAPI` clients mirroring the routers.
- **Razorpay Checkout script SRI**: the official `https://checkout.razorpay.com/v1/checkout.js` is an unversioned URL that Razorpay updates silently, so a pinned `integrity="sha384-..."` hash would break checkout on their next push. We load it without SRI (matching Razorpay's documented integration) and mitigate CDN-compromise risk by (a) loading it only on the `/pricing` and `/billing` checkout paths, not globally, and (b) post-message verification: the Checkout handler only trusts data it sends to our own `verify` endpoint, which independently verifies the Razorpay HMAC signature server-side. The server-side signature check is the real trust boundary — a compromised CDN cannot forge a valid HMAC without `RAZORPAY_KEY_SECRET`.

## 5. Razorpay checkout flow

1. User picks a plan (and billing cycle) on `/pricing` → `POST /subscriptions/checkout` creates a Razorpay subscription (monthly, recurring mandate) or a one-time order (yearly) → returns `razorpay_subscription_id` / `razorpay_order_id` + `key_id` + `amount`.
2. Frontend opens Razorpay Checkout with the id(s), user info, and a handler. UPI/cards/netbanking/recurring mandate shown.
3. On payment success, the Checkout handler calls `POST /subscriptions/verify` with `razorpay_payment_id`, `razorpay_subscription_id`, `razorpay_signature` → backend verifies the HMAC signature using `RAZORPAY_KEY_SECRET`, records a `payment` (idempotent on `razorpay_payment_id`), and activates the `subscription`.
4. The **webhook is the source of truth**: `subscription.activated`/`.charged`/`.cancelled`/`.expired` and `payment.failed` update `subscriptions` + `payments` + the denormalized `user.subscription` cache. Webhooks survive client drop-off and are idempotent on event id + payment/subscription id.
5. Cancel: `POST /subscriptions/cancel` → Razorpay "cancel at period end"; local status flips to `cancel_at_period_end=true` optimistically and to `cancelled` when the webhook confirms at period end.

Razorpay plan IDs (`RAZORPAY_PLAN_PRO_MONTHLY`, etc.) are created in the Razorpay dashboard and stored as env vars; the checkout endpoint looks them up by `(plan, billing_cycle)`.

## 6. Production hardening (minimal but real)

- **Rate limiting** via `slowapi`: per-IP limits on `/auth/*` (login/signup brute-force) and per-user limits on all AI-generation endpoints (`/mock-tests/generate`, `/flashcards/generate`, `/ai-materials/summarize`, `/questions/ask*`). Protects Razorpay cost + Gemini cost.
- **Razorpay webhook security**: raw-body HMAC-SHA256 verification with `RAZORPAY_WEBHOOK_SECRET`; idempotency on event id + `razorpay_payment_id`/`razorpay_subscription_id`; reject replays.
- **Config** (`core/config.py`): `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`, `RAZORPAY_PLAN_PRO_MONTHLY`, `RAZORPAY_PLAN_PREMIUM_MONTHLY`, `RAZORPAY_PLAN_PRO_YEARLY`, `RAZORPAY_PLAN_PREMIUM_YEARLY`, `Razorpay` test vs live mode flag. `.env.example` updated; no live keys in repo.
- **Role guards**: `require_role("admin"|"subadmin")` on every new admin/org endpoint; ownership checks on all billing reads (a user can only read their own subscription/invoices; a sub-admin only their own org).
- **Logging**: structured logs on every billing/webhook call (event id, user_id, plan, status, amount) for support + reconciliation.
- **Existing restricted CORS** (already non-wildcard) is kept; the webhook route is exempted from the JSON-body parser to allow raw-body signature verification.

## 7. Testing

### Backend (pytest)
- `tests/test_plan_enforcement.py`: limit math for starter/pro/premium/org; effective-plan resolution (own active sub vs org tier vs free); 402 payload shape; usage computation from existing collections.
- `tests/test_billing.py`: Razorpay signature verification with sample vectors (known HMAC); `verify` idempotency on repeated `razorpay_payment_id`; checkout → verify → webhook activation happy path (mocked Razorpay client).
- `tests/test_webhooks.py`: signature rejection (bad secret), event idempotency (same event twice → one update), each event type mapping to the right status.
- `tests/test_orgs.py`: seat boundary (enroll succeeds until `seats_total`, then 402/409), expired/suspended org rejects enroll, sub-admin-only endpoints 403 for students.
- `tests/test_admin.py`: admin-only endpoints 403 for non-admins; manual activation extends `current_period_end`; platform analytics aggregation correctness.
- Mock the Razorpay SDK with a fake client in tests; no network.

### Frontend
- `npm run build` + `npm run lint` green.
- Manual checkout in Razorpay **test mode** end-to-end: pricing → checkout → verify → webhook → billing page shows active plan + usage.
- 402 → upgrade banner renders on the triggering page.

### No live keys in the repo; test-mode keys via `.env`.

## 8. Build sequence (high level — detailed plan comes next)

1. Backend models + config + plans constants + `.env.example`.
2. `plan_enforcement` + `billing_service` + `subscription_service`.
3. `subscription_router`, `webhook_router`, `org_router`, `admin_router`; wire into `main.py`.
4. Add `enforce_limit` to existing endpoints.
5. Frontend `subscriptionAPI`/`orgAPI`/`adminAPI`; `/billing`, `/pricing` real, onboarding sub-admin path, `/admin` wired, `/org` page.
6. 402 interceptor + `<UpgradeBanner>`.
7. Rate limiting + hardening.
8. Tests (backend) + build/lint (frontend) + manual test-mode checkout.

## 9. Risks and mitigations

- **Razorpay key/secret leakage** → secrets only in env; webhook secret separate from key secret; test keys default in dev.
- **Webhook/web client race** → webhook is source of truth; `verify` writes `payment` but `subscription` status is reconciled by webhook; reads tolerate a brief `pending` state.
- **Org seat drift** → `seats_used` is denormalized; reconcile on every enroll/leave and via a nightly admin job (out of scope for v1 — a read-time recompute fallback is included).
- **Existing features broken by enforcement** → `enforce_limit` is additive; free-tier limits are generous enough that existing flows keep working; the 402 path is a clean upgrade banner, never a hard crash.
- **Plan drift between backend and frontend** → frontend reads limits from `/subscriptions/plans`; constants are mirrors only.

## 10. Open items (none blocking)

- Per-seat pricing for orgs: exact INR amounts TBD in Razorpay dashboard (design assumes `seats_total` × tier price, billed as one Razorpay subscription with quantity). Confirm during implementation.
- Yearly billing via Razorpay: implemented as a one-time payment + manual renewal reminder, or as a 12-month subscription — confirm Razorpay capability during implementation.