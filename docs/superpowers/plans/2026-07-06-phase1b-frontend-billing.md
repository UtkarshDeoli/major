# Phase 1b — Frontend Billing, Orgs & Admin (Sellable Backbone)

**Goal:** Wire the sellable backbone into the Next.js frontend so authenticated users can subscribe, manage org seats, and admins can manage the platform. Also add a 402 upgrade prompt so plan enforcement feels like a funnel, not an error wall.

**Branch:** `phase1/sellable-backbone`

## Scope

1. Live pricing & Razorpay checkout (marketing page + authenticated checkout)
2. `/billing` dashboard (current plan, usage, invoices, cancel)
3. Sub-admin onboarding path (create org after signup)
4. `/org` management page (seats, invites, members)
5. Wire `/admin` page to backend endpoints
6. Global 402 upgrade prompt
7. Sidebar/nav updates for new pages

## Technical Decisions

- **Plan data source of truth:** Backend `GET /subscriptions/plans`. The existing `lib/constants/plans.ts` becomes a fallback static skeleton only; the live marketing page fetches plans to show INR prices and limits.
- **Razorpay Checkout:** Loaded via dynamic script from `https://checkout.razorpay.com/v1/checkout.js` with SRI fallback (defer load, show spinner until ready). The backend `POST /subscriptions/checkout` returns `order_id`, `amount`, `currency`, `key_id`, and `prefill`. After payment, `POST /subscriptions/verify` activates the subscription.
- **Auth-aware checkout:** Public pricing page shows plans but redirects anonymous users to `/signup` with `?redirect=/pricing` (or a query param) when they click a paid plan. Authenticated users open Razorpay directly.
- **Sub-admin onboarding:** After signup, if `user.role === "subadmin"`, the existing `getPostAuthRedirect` sends them to `/admin`. We intercept this with a new route `/onboarding/org` that is the required first step for sub-admins without an org. Once the org is created, they land on `/admin`.
- **402 handling:** The axios response interceptor in `lib/api.ts` already handles 401. We add handling for 402: emit a custom event `orbit:upgrade-required` with the backend detail payload. A new `UpgradeBanner` component listens to this event and shows a dismissible sticky banner with a CTA to `/billing`.
- **Role guard updates:** The `/admin` page will allow `admin` and `subadmin`; `/org` will allow `subadmin`.

## API Wrappers to Add

### Subscriptions (`lib/api.ts`)
- `subscriptionAPI.getPlans()` → `GET /subscriptions/plans`
- `subscriptionAPI.getMe()` → `GET /subscriptions/me`
- `subscriptionAPI.checkout(plan, billing_cycle)` → `POST /subscriptions/checkout`
- `subscriptionAPI.verify(payload)` → `POST /subscriptions/verify`
- `subscriptionAPI.cancel()` → `POST /subscriptions/cancel`
- `subscriptionAPI.getInvoices()` → `GET /subscriptions/invoices`

### Organizations (`lib/api.ts`)
- `orgAPI.create(payload)` → `POST /orgs/`
- `orgAPI.getMe()` → `GET /orgs/me`
- `orgAPI.invite(payload)` → `POST /orgs/invite`
- `orgAPI.listMembers()` → `GET /orgs/members`
- `orgAPI.removeMember(email)` → `DELETE /orgs/members/{email}`
- `orgAPI.addSeats(add_seats)` → `POST /orgs/seats`
- `orgAPI.enroll(code)` → `POST /orgs/enroll/{code}`

### Admin (`lib/api.ts`)
- `adminAPI.listUsers(params)` → `GET /admin/users`
- `adminAPI.updateRole(email, role)` → `PATCH /admin/users/{email}/role`
- `adminAPI.updateStatus(email, active)` → `PATCH /admin/users/{email}/status`
- `adminAPI.listOrgs()` → `GET /admin/orgs`
- `adminAPI.updateOrg(orgId, payload)` → `PATCH /admin/orgs/{orgId}`
- `adminAPI.listSubscriptions()` → `GET /admin/subscriptions`
- `adminAPI.activateSubscription(userId, plan, days)` → `POST /admin/subscriptions/{userId}/activate`
- `adminAPI.getAnalytics()` → `GET /admin/analytics`

## New Routes / Pages

- `app/pricing/page.tsx` — standalone public pricing page, reuses `Pricing` marketing component.
- `app/(dashboard)/billing/page.tsx` — subscription dashboard.
- `app/(dashboard)/org/page.tsx` — org management (sub-admin only).
- `app/onboarding/org/page.tsx` — sub-admin org creation step.
- Update `app/(dashboard)/admin/page.tsx` to fetch real data.

## Files to Touch

- `Frontend/lib/api.ts`
- `Frontend/lib/constants/plans.ts`
- `Frontend/lib/context/auth-context.tsx` (add subscription/org refresh helpers if needed)
- `Frontend/components/marketing/pricing.tsx`
- `Frontend/components/marketing/navbar.tsx`
- `Frontend/components/dashboard/app-shell.tsx`
- `Frontend/components/auth/auth-form.tsx` (pass redirect after login)
- `Frontend/components/auth/route-protection/role-guard.tsx`
- `Frontend/lib/errors.ts` (402 message formatting)
- New: `Frontend/components/billing/razorpay-loader.tsx`
- New: `Frontend/components/billing/upgrade-banner.tsx`
- New: `Frontend/components/org/org-form.tsx`
- New: `Frontend/components/org/invite-manager.tsx`

## Tests

Frontend vitest suite exists but is minimal. We will rely on `next build` and `next lint` as the primary acceptance gates. No new unit tests are required for Phase 1b, but the pages must type-check and lint clean.

## Acceptance Criteria

- [ ] `npm run build` in `Frontend/` completes with 0 errors.
- [ ] `npm run lint` in `Frontend/` passes.
- [ ] Pricing page fetches live plans and triggers Razorpay checkout for logged-in users.
- [ ] `/billing` displays current plan, usage meters, and invoice list.
- [ ] Sub-admin without org is redirected to `/onboarding/org` and can create an org.
- [ ] `/org` shows members, allows invite generation, seat addition, and member removal.
- [ ] `/admin` lists real users, orgs, subscriptions, and supports role/status/subscription updates.
- [ ] Any 402 response from the backend shows the upgrade banner with a link to `/billing`.

## Deferred

- Frontend tests for billing/org/admin components (out of scope for Phase 1b).
- Webhook-driven UI refresh (payment success auto-refreshes via `/billing` mount today).
