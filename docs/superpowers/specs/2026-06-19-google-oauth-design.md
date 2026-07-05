# Google OAuth Integration Design

**Date:** 2026-06-19
**Status:** Approved

## Overview

Add full Google OAuth sign-in/sign-up to Orbit using a backend-mediated flow. The FastAPI backend handles the OAuth redirect and token exchange with Google, then redirects the frontend with an existing-style JWT. This reuses the existing auth infrastructure (JWT, localStorage, auth-context) with minimal new code.

## Decisions (Clarified & Approved)

| Decision | Choice |
|---|---|
| Flow type | Backend-mediated (redirect to FastAPI `/auth/google/login`) |
| User model | `password_hash` becomes `Optional[str]`, add `auth_provider` + `provider_uid` fields |
| Account merging | Merge by email — Google login with an existing email logs into that account |
| Redirect logic | Same as email login — teachers → `/teacher`, others → `/dashboard` |

## 1. Configuration & Dependencies

### New env vars (`Backend/.env.example`, `Backend/src/core/config.py`)

```
GOOGLE_CLIENT_ID=<from Google Cloud Console>
GOOGLE_CLIENT_SECRET=<from Google Cloud Console>
GOOGLE_REDIRECT_URI=http://localhost:8001/auth/google/callback
FRONTEND_URL=http://localhost:3000
```

`FRONTEND_URL` is the frontend's base URL — the backend redirects the user's browser here after successful OAuth, with the JWT as a query param.

### New Python dependency

- `google-auth-oauthlib` — provides the `Flow` class for server-side OAuth 2.0 Authorization Code + PKCE flow.

(Note: `google-auth` is already present via Gemini's dependencies, so we only need the oauthlib companion.)

### CORS fix

`Backend/src/main.py`: change `allow_origins=["*"]` to `allow_origins=[settings.FRONTEND_URL]`. Wildcard origin with credentialed redirects violates the CORS spec and breaks OAuth callbacks.

---

## 2. User Model Changes

**File:** `Backend/src/core/models.py`

| Field | Current | New |
|---|---|---|
| `password_hash` | `str` | `Optional[str] = None` |
| `auth_provider` | — | `Optional[str] = None` (stores `"email"` or `"google"`) |
| `provider_uid` | — | `Optional[str] = None` (stores Google's `sub` claim) |

### Migration logic (no migration script — just new behavior forward)

- **New email signups:** `auth_provider="email"`, `password_hash` set as before, `provider_uid=None`
- **New Google signups:** `auth_provider="google"`, `password_hash=None`, `provider_uid=<Google sub>`
- **Existing users who later Google-login:** `auth_provider="google"`, `provider_uid=<Google sub>` set on the existing document (merging the account)
- **Old documents missing these fields:** existing users with `password_hash` set and no `auth_provider` field — the code treats missing auth_provider as email-based auth (backward compatible)

### Auth service updates (`Backend/src/services/auth_service.py`)

- `create_user()` updated to handle the new optional fields
- `authenticate_user()` — when `auth_provider != "email"` (or is `None` and `password_hash` is `None`), password-based auth fails gracefully
- New helper `find_or_create_google_user(email, name, google_sub)` — core merge logic

---

## 3. Backend Endpoints

**File:** `Backend/src/routers/auth_router.py`

### `GET /auth/google/login`

1. Create a `google-auth-oauthlib` `Flow` from client config:
   - `client_id`, `client_secret` from settings
   - `redirect_uri` from settings (`GOOGLE_REDIRECT_URI`)
   - `scopes = ["openid", "email", "profile"]`
2. Generate authorization URL and state token
3. Store state in a short-lived signed cookie (or server-side session)
4. Redirect user's browser to Google's consent screen

### `GET /auth/google/callback`

1. Receive `code` and `state` from Google
2. Verify state matches the stored value (CSRF protection)
3. Exchange `code` for tokens using the Flow object
4. Decode the ID token to extract: `email`, `name`, `sub` (Google user ID)
5. **Account resolution:**
   - Query MongoDB for user by `email`
   - If found: update `auth_provider="google"`, `provider_uid=sub`, `name` (if provided)
   - If not found: create new user with `auth_provider="google"`, `provider_uid=sub`, `name`, role=`"student"`
6. Issue JWT via existing `create_access_token(data={"sub": user.email})`
7. Redirect to `{FRONTEND_URL}/auth/callback?token={jwt}`

### Error handling

- If the state doesn't match → redirect to `{FRONTEND_URL}/auth/callback?error=invalid_state`
- If Google returns an error → redirect to `{FRONTEND_URL}/auth/callback?error={error}`
- If the email is missing from the Google profile → redirect to `{FRONTEND_URL}/auth/callback?error=no_email`

### New helper module: `Backend/src/services/google_oauth_service.py`

Encapsulates the Flow creation, state management, and token exchange. Keeps `auth_router.py` thin.

---

## 4. Frontend — Callback Page

**New file:** `Frontend/app/(auth)/auth/callback/page.tsx`

A thin client component (no SSR needed — it's a redirect handler):

1. On mount, extract `token` or `error` from URL search params (`useSearchParams`)
2. If error: display error message with "Try again" link back to login
3. If token:
   - Store token in `localStorage["token"]`
   - Call `auth-context`'s `GET /auth/me` to hydrate user (reuses existing flow)
   - Redirect to `/teacher` if role is teacher, else `/dashboard` (reuses existing redirect logic)
4. Show a loading spinner while processing

### Auth form changes

**File:** `Frontend/components/auth/auth-form.tsx`

Replace the placeholder `handleGoogleSignIn` function (currently shows a toast) with:

```ts
const handleGoogleSignIn = () => {
  window.location.href = `${process.env.NEXT_PUBLIC_API_URL}/auth/google/login`
}
```

### API client

**File:** `Frontend/lib/api.ts`

No new methods needed — the token returned by the callback flows through the existing `localStorage` + axios interceptor pattern. The callback page stores the token, then the existing `getMe()` call in the auth context handles hydration.

---

## 5. Security Considerations

1. **State parameter:** Prevents CSRF on the callback. We'll store it in a signed cookie (`itsdangerous` or just use `google-auth-oauthlib`'s built-in state handling which stores it in a session cookie by default).
2. **ID token verification:** `google-auth-oauthlib` verifies the token's signature, issuer (`accounts.google.com`), audience (our `GOOGLE_CLIENT_ID`), and expiry automatically.
3. **HTTPS in production:** The callback must be served over HTTPS in production. Locally, `http://localhost` is acceptable.
4. **CORS hardening:** `allow_origins=["*"]` → `allow_origins=[settings.FRONTEND_URL]` prevents credential misuse.
5. **No password for Google users:** Since `password_hash` is Optional[None] for Google users, the password-based login path naturally rejects them (no hash to compare against).

---

## 6. Files Changed / Created

### Backend
| File | Action |
|---|---|
| `Backend/requirements.txt` | Add `google-auth-oauthlib` |
| `Backend/.env.example` | Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `FRONTEND_URL` |
| `Backend/src/core/config.py` | Add new Settings fields |
| `Backend/src/main.py` | Fix CORS `allow_origins` |
| `Backend/src/core/models.py` | Update `User` model (optional password_hash, new fields) |
| `Backend/src/services/auth_service.py` | Update `create_user`, `authenticate_user` for new fields; add `find_or_create_google_user` |
| `Backend/src/services/google_oauth_service.py` | **New** — Flow creation, state mgmt, token exchange |
| `Backend/src/routers/auth_router.py` | Add `/auth/google/login`, `/auth/google/callback` endpoints |

### Frontend
| File | Action |
|---|---|
| `Frontend/app/(auth)/auth/callback/page.tsx` | **New** — token handler page |
| `Frontend/components/auth/auth-form.tsx` | Wire real Google redirect to the button |

### Docs
| File | Action |
|---|---|
| `docs/superpowers/specs/2026-06-19-google-oauth-design.md` | **New** — this document |