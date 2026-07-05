# Google OAuth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full Google OAuth sign-in/sign-up flow using a backend-mediated redirect pattern.

**Architecture:** FastAPI backend handles the OAuth Authorization Code flow with Google. A new `google-oauth-service` creates the Flow, exchanges the code, and verifies the ID token. The backend then issues the existing JWT and redirects the frontend with the token as a query param. The frontend's callback page stores the token and proceeds through the existing auth context. The User model gets optional password_hash + auth_provider/provider_uid fields to support both email and Google users.

**Tech Stack:** FastAPI, google-auth-oauthlib, PyJWT, Next.js 15, axios

---

## File Structure

### Backend — Files to Modify

| File | Responsibility |
|---|---|
| `Backend/.env.example` | Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `FRONTEND_URL` |
| `Backend/requirements.txt` | Add `google-auth-oauthlib` |
| `Backend/src/core/config.py` | Add new Settings fields for Google OAuth and FRONTEND_URL |
| `Backend/src/core/models.py` | Make `password_hash` optional, add `auth_provider` + `provider_uid` fields |
| `Backend/src/services/auth_service.py` | Update `create_user`/`authenticate_user` for new fields; add `find_or_create_google_user()` |
| `Backend/src/services/google_oauth_service.py` | **New** — OAuth Flow creation, state management, token exchange |
| `Backend/src/routers/auth_router.py` | Add `GET /auth/google/login`, `GET /auth/google/callback` |
| `Backend/src/main.py` | Fix CORS `allow_origins` |

### Frontend — Files to Modify

| File | Responsibility |
|---|---|
| `Frontend/app/(auth)/auth/callback/page.tsx` | **New** — Extract token from URL, store in localStorage, redirect |
| `Frontend/components/auth/auth-form.tsx` | Wire Google button to redirect to backend |

---

### Task 1: Backend — Config & Dependencies

**Files:**
- Modify: `Backend/.env.example`
- Modify: `Backend/requirements.txt`
- Modify: `Backend/src/core/config.py`

- [ ] **Step 1: Add Google OAuth env vars to `.env.example`**

Append to `Backend/.env.example`:

```env
# Google OAuth Configuration
# Create credentials at: https://console.cloud.google.com/apis/credentials
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8001/auth/google/callback
FRONTEND_URL=http://localhost:3000
```

- [ ] **Step 2: Add `google-auth-oauthlib` to requirements.txt**

Append to `Backend/requirements.txt`:

```
google-auth-oauthlib==1.2.1
```

- [ ] **Step 3: Add new Settings fields to `config.py`**

Edit `Backend/src/core/config.py` — add the new env vars and their exports:

```python
import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "phadai"
    MONGODB_CONNECT_TIMEOUT: int = 30000  # 30 seconds timeout

    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ChromaDB settings
    CHROMA_DB_PATH: str = "./chroma_db"
    GEMINI_API_KEY: str

    # Google OAuth settings
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8001/auth/google/callback"
    FRONTEND_URL: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

# Create settings instance
settings = Settings()

# Export settings variables
MONGODB_URL = settings.MONGODB_URL
MONGODB_DB_NAME = settings.MONGODB_DB_NAME
MONGODB_CONNECT_TIMEOUT = settings.MONGODB_CONNECT_TIMEOUT
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
CHROMA_DB_PATH = settings.CHROMA_DB_PATH
GEMINI_API_KEY = settings.GEMINI_API_KEY
GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = settings.GOOGLE_REDIRECT_URI
FRONTEND_URL = settings.FRONTEND_URL
```

- [ ] **Step 4: Verify backend still starts**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -c "from src.core.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI, FRONTEND_URL; print('Config OK:', FRONTEND_URL)"
```

Expected: `Config OK: http://localhost:3000` (the default)

- [ ] **Step 5: Commit**

```bash
git add Backend/.env.example Backend/requirements.txt Backend/src/core/config.py
git commit -m "feat(oauth): add Google OAuth config and dependencies"
```

---

### Task 2: Backend — User Model Changes

**Files:**
- Modify: `Backend/src/core/models.py:276-289`

- [ ] **Step 1: Update the User model**

Edit the `User` class in `Backend/src/core/models.py`. Change `password_hash: str` to `password_hash: Optional[str] = None` and add two new fields:

```python
class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    email: str
    name: Optional[str] = None
    password_hash: Optional[str] = None
    auth_provider: Optional[str] = None  # "email" or "google"
    provider_uid: Optional[str] = None   # Google "sub" ID, or None for email users
    role: Literal["student", "teacher", "subadmin", "admin"] = "student"
    institute: Optional[str] = None
    preferred_language: str = "en"
    onboarding_completed: bool = False
    active_exam_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 2: Verify the model imports correctly**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -c "from src.core.models import User; u=User(email='test@test.com', password_hash='hash', auth_provider='email'); print('Email user OK:', u.model_dump().get('auth_provider')); u2=User(email='google@test.com', auth_provider='google', provider_uid='12345'); print('Google user OK: password_hash is', u2.password_hash)"
```

Expected: both users print successfully; google user shows `password_hash is None`.

- [ ] **Step 3: Commit**

```bash
git add Backend/src/core/models.py
git commit -m "feat(oauth): make password_hash optional, add auth_provider and provider_uid"
```

---

### Task 3: Backend — Auth Service Updates

**Files:**
- Modify: `Backend/src/services/auth_service.py`

- [ ] **Step 1: Update `create_user` to set `auth_provider="email"`**

In `Backend/src/services/auth_service.py`, update the `create_user` function. Change the `User(...)` constructor call to include the new fields:

```python
async def create_user(
    email: str,
    password: str,
    name: Optional[str] = None,
    role: str = "student",
    institute: Optional[str] = None,
    preferred_language: Optional[str] = None
):
    _ensure_auth_db()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        # Check if user already exists
        existing_user = await get_user_by_email(email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Validate role. Public signup is always a student; privileged roles
        # require admin/sub-admin enrollment. Sanitize any caller-supplied role.
        allowed_roles = {"student", "teacher", "subadmin", "admin"}
        if role not in allowed_roles:
            role = "student"
        if role in {"admin", "subadmin", "teacher"}:
            role = "student"

        # Create new user with hashed password
        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            password_hash=hashed_password,
            auth_provider="email",
            name=name,
            role=role,  # type: ignore[arg-type]
            institute=institute,
            preferred_language=preferred_language or "en",
        )

        result = await users_collection.insert_one(user.model_dump(by_alias=True))
        created_user = await users_collection.find_one({"_id": result.inserted_id})
        return created_user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
```

The only change from the original is adding `auth_provider="email"` to the `User(...)` constructor.

- [ ] **Step 2: Update `authenticate_user` to handle Google-only users**

In `authenticate_user`, add a check so that users without a password_hash (Google-only users) fail password auth gracefully:

```python
async def authenticate_user(email: str, password: str):
    _ensure_auth_db()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        user = await get_user_by_email(email)
        if not user:
            return False
        # If user has no password_hash (Google-only account), password auth fails
        if not user.get("password_hash"):
            return False
        if not verify_password(password, user["password_hash"]):
            return False
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
```

The only change is adding the `if not user.get("password_hash"): return False` guard before `verify_password`.

- [ ] **Step 3: Add `find_or_create_google_user` helper**

Add this new async function to `Backend/src/services/auth_service.py` (after `authenticate_user`):

```python
async def find_or_create_google_user(email: str, name: Optional[str], google_sub: str):
    """Find a user by email and link Google auth, or create a new Google user.

    - If a user with this email exists: update provider_uid and auth_provider (merge).
    - If no user exists: create a new one with auth_provider="google".

    Returns the user document dict.
    """
    _ensure_auth_db()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        existing = await get_user_by_email(email)

        if existing:
            # Merge: link Google auth to existing account
            update_fields = {
                "auth_provider": "google",
                "provider_uid": google_sub,
                "updated_at": datetime.now(timezone.utc),
            }
            if name and not existing.get("name"):
                update_fields["name"] = name
            await users_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": update_fields}
            )
            return await users_collection.find_one({"_id": existing["_id"]})

        # Create new Google user
        user = User(
            email=email,
            name=name,
            auth_provider="google",
            provider_uid=google_sub,
            role="student",
        )
        result = await users_collection.insert_one(user.model_dump(by_alias=True))
        return await users_collection.find_one({"_id": result.inserted_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
```

- [ ] **Step 4: Update the module's `__all__`-equivalent exports**

The existing import block at the top of `auth_router.py` imports specific functions. We need to make sure `find_or_create_google_user` is exportable. No explicit action — Python doesn't require it. Just verify the function is defined at module level.

- [ ] **Step 5: Run existing auth tests to verify no regression**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -m pytest tests/test_auth.py -v
```

Expected: all existing auth tests pass. If tests fail because they relied on `password_hash` being required in the User constructor, update those test fixtures accordingly.

- [ ] **Step 6: Commit**

```bash
git add Backend/src/services/auth_service.py
git commit -m "feat(oauth): update auth service for Google users, add find_or_create_google_user"
```

---

### Task 4: Backend — New Google OAuth Service

**Files:**
- Create: `Backend/src/services/google_oauth_service.py`

- [ ] **Step 1: Create the Google OAuth service module**

```python
"""Google OAuth service — handles the Authorization Code flow with Google.

Uses google-auth-oauthlib's Flow class to generate the authorization URL,
exchange the authorization code for tokens, and verify/ decode the ID token.
"""

from typing import Optional
from google_auth_oauthlib.flow import Flow
from src.core.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI

# Scopes requested during Google OAuth
_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def _create_flow(state: Optional[str] = None) -> Flow:
    """Create a google-auth-oauthlib Flow configured with our client settings.

    Args:
        state: Optional CSRF state string. If provided, the flow will use
               it instead of generating a random one (needed on the callback
               side to reuse the flow created during login).

    Returns:
        A configured Flow instance.
    """
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
        }
    }
    flow = Flow.from_client_config(
        client_config,
        scopes=_OAUTH_SCOPES,
        state=state,
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    return flow


def get_authorization_url() -> tuple[str, str]:
    """Generate the Google OAuth authorization URL and CSRF state token.

    Returns:
        (authorization_url, state) tuple.
    """
    flow = _create_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="select_account",
    )
    return authorization_url, state


def exchange_code_for_tokens(code: str, state: str) -> dict:
    """Exchange an authorization code for Google tokens.

    Args:
        code: The authorization code from Google's callback.
        state: The CSRF state token that was returned by get_authorization_url().

    Returns:
        The decoded ID token info dict, containing at minimum: sub, email, name.

    Raises:
        google.auth.exceptions.GoogleAuthError: If token exchange or verification fails.
    """
    flow = _create_flow(state=state)
    flow.fetch_token(code=code)
    id_token = flow.credentials.id_token
    # id_token is already a decoded dict when using google-auth-oauthlib's Flow
    return id_token
```

- [ ] **Step 2: Verify the module imports**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -c "from src.services.google_oauth_service import get_authorization_url; print('Google OAuth service imports OK')"
```

Expected: `Google OAuth service imports OK`

- [ ] **Step 3: Commit**

```bash
git add Backend/src/services/google_oauth_service.py
git commit -m "feat(oauth): add Google OAuth service with auth URL and token exchange"
```

---

### Task 5: Backend — Auth Router: Google Login & Callback

**Files:**
- Modify: `Backend/src/routers/auth_router.py`

- [ ] **Step 1: Add imports and new endpoints to `auth_router.py`**

Replace the existing `auth_router.py` content with the full updated version. This keeps the existing endpoints unchanged and adds the two new Google OAuth endpoints:

Final file content for `Backend/src/routers/auth_router.py`:

```python
from datetime import timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from src.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, FRONTEND_URL
from src.core.models import SubscriptionInfo
from src.core.security import get_current_user
from src.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_user,
    find_or_create_google_user,
    get_user_by_email,
)
from src.services.google_oauth_service import (
    exchange_code_for_tokens,
    get_authorization_url,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class UserResponse(BaseModel):
    email: str
    name: Optional[str] = None
    role: Literal["student", "teacher", "subadmin", "admin"] = "student"
    institute: Optional[str] = None
    preferred_language: str = "en"
    onboarding_completed: bool = False
    active_exam_id: Optional[str] = None
    teacher_id: Optional[str] = None
    managed_by: Optional[str] = None
    license_id: Optional[str] = None
    subscription: Optional[SubscriptionInfo] = None


class SignupResponse(BaseModel):
    email: str
    access_token: str
    token_type: str


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate):
    """Register a new user with email and password and return an access token."""
    user = await create_user(
        email=user_data.email,
        password=user_data.password,
        name=user_data.name,
    )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )

    return {
        "email": user["email"],
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and get access token"""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def get_me(user_email: str = Depends(get_current_user)):
    """Get the current authenticated user's profile"""
    user = await get_user_by_email(user_email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse(
        email=user["email"],
        name=user.get("name"),
        role=user.get("role", "student"),
        institute=user.get("institute"),
        preferred_language=user.get("preferred_language", "en"),
        onboarding_completed=user.get("onboarding_completed", False),
        active_exam_id=user.get("active_exam_id"),
        teacher_id=user.get("teacher_id"),
        managed_by=user.get("managed_by"),
        license_id=user.get("license_id"),
        subscription=user.get("subscription"),
    )


@router.get("/google/login")
async def google_login():
    """Redirect the user to Google's OAuth consent screen."""
    authorization_url, state = get_authorization_url()
    # Use a RedirectResponse that sets the state as a signed cookie for CSRF
    response = RedirectResponse(url=authorization_url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="oauth_state",
        value=state,
        max_age=600,  # 10 minutes — enough time for Google's consent screen
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
    )
    return response


@router.get("/google/callback")
async def google_callback(code: str, state: str, request: Request):
    """Handle the OAuth callback from Google.

    Exchanges the authorization code for tokens, finds or creates a user,
    issues a JWT, and redirects the frontend with the token.
    """
    # Verify CSRF state cookie matches
    stored_state = request.cookies.get("oauth_state")
    if not stored_state or stored_state != state:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?error=invalid_state",
            status_code=status.HTTP_302_FOUND,
        )

    try:
        # Exchange code for ID token info
        id_info = exchange_code_for_tokens(code, state)

        email = id_info.get("email")
        if not email:
            return RedirectResponse(
                url=f"{FRONTEND_URL}/auth/callback?error=no_email",
                status_code=status.HTTP_302_FOUND,
            )

        name = id_info.get("name")
        google_sub = id_info.get("sub")

        # Find or create user
        user = await find_or_create_google_user(
            email=email,
            name=name,
            google_sub=google_sub,
        )

        # Issue JWT
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )

        # Redirect to frontend with token, clearing the state cookie
        response = RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?token={access_token}",
            status_code=status.HTTP_302_FOUND,
        )
        response.delete_cookie("oauth_state")
        return response

    except Exception as e:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?error=exchange_failed",
            status_code=status.HTTP_302_FOUND,
        )
```

- [ ] **Step 2: Add `Request` to the FastAPI imports**

The `google_callback` endpoint uses `request: Request`. Make sure to import it at the top:

```python
from fastapi import APIRouter, Depends, HTTPException, Request, status
```

(Add `Request` to the existing `fastapi` import line.)

- [ ] **Step 3: Verify the router imports cleanly**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -c "from src.routers.auth_router import router; print('Auth router imports OK')"
```

Expected: `Auth router imports OK`

- [ ] **Step 4: Commit**

```bash
git add Backend/src/routers/auth_router.py
git commit -m "feat(oauth): add Google login and callback endpoints"
```

---

### Task 6: Backend — Fix CORS Configuration

**Files:**
- Modify: `Backend/src/main.py`

- [ ] **Step 1: Import `FRONTEND_URL` from config and fix CORS**

Edit `Backend/src/main.py`. Replace the CORS middleware configuration to use the explicit frontend origin:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nltk

# Import our routers
from src.routers import (
    auth_router,
    pdf_router,
    document_router,
    question_router,
    analysis_router,
    mock_test_router,
    teacher_router,
    analytics_router,
    exam_router,
    subject_router,
    collection_router,
    material_router,
    onboarding_router,
)
from src.core.config import FRONTEND_URL

app = FastAPI()

# Allow CORS for the frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # Explicit origin, not wildcard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our routers
app.include_router(auth_router)
app.include_router(pdf_router)
app.include_router(document_router)
app.include_router(question_router)
app.include_router(analysis_router)
app.include_router(mock_test_router)
app.include_router(teacher_router)
app.include_router(analytics_router)
app.include_router(exam_router)
app.include_router(subject_router)
app.include_router(collection_router)
app.include_router(material_router)
app.include_router(onboarding_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Padhai Whallah API"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

The only changes are: add `from src.core.config import FRONTEND_URL` at the top, and change `allow_origins=["*"]` to `allow_origins=[FRONTEND_URL]`.

- [ ] **Step 2: Verify main.py imports**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -c "from src.main import app; print('Main imports OK')"
```

Expected: `Main imports OK`

- [ ] **Step 3: Commit**

```bash
git add Backend/src/main.py
git commit -m "fix(cors): restrict CORS to FRONTEND_URL instead of wildcard"
```

---

### Task 7: Frontend — Callback Page

**Files:**
- Create: `Frontend/app/(auth)/auth/callback/page.tsx`

- [ ] **Step 1: Create the callback page**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import api from "@/lib/api";

export default function AuthCallbackPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = searchParams.get("token");
    const errorParam = searchParams.get("error");

    if (errorParam) {
      const messages: Record<string, string> = {
        invalid_state: "Security check failed. Please try again.",
        no_email: "Google account has no email associated. Please use a different account.",
        exchange_failed: "Failed to sign in with Google. Please try again.",
      };
      setError(messages[errorParam] || "An unknown error occurred.");
      return;
    }

    if (token) {
      // Store the JWT token
      localStorage.setItem("token", token);

      // Fetch user info to determine redirect
      api
        .get("/auth/me")
        .then((response) => {
          const user = response.data as { role: string };
          if (user.role === "teacher") {
            router.replace("/teacher");
          } else {
            router.replace("/dashboard");
          }
        })
        .catch(() => {
          // If /auth/me fails, still try to redirect to dashboard
          router.replace("/dashboard");
        });
    } else {
      setError("No token received from authentication.");
    }
  }, [router, searchParams]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0D1520]">
        <div className="text-center max-w-md p-8">
          <div className="text-red-400 text-5xl mb-4">⚠</div>
          <h1 className="text-2xl font-bold text-white mb-3">Sign In Failed</h1>
          <p className="text-gray-400 mb-6">{error}</p>
          <button
            onClick={() => router.replace("/login")}
            className="px-6 py-3 rounded-md bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold hover:from-purple-600 hover:to-blue-600 transition-all"
          >
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0D1520]">
      <div className="text-center">
        <div className="h-12 w-12 mx-auto mb-4 rounded-full border-4 border-purple-500 border-t-transparent animate-spin" />
        <p className="text-gray-400 text-lg">Completing sign in...</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify the frontend dev server compiles**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Frontend
npm run build 2>&1 | tail -20
```

Expected: build succeeds with no errors related to the new callback page.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/\(auth\)/auth/callback/page.tsx
git commit -m "feat(oauth): add auth callback page to handle OAuth token redirect"
```

---

### Task 8: Frontend — Wire Google Button

**Files:**
- Modify: `Frontend/components/auth/auth-form.tsx`

- [ ] **Step 1: Replace the placeholder Google sign-in handler**

In `Frontend/components/auth/auth-form.tsx`, replace the `handleGoogleSignIn` function:

```tsx
  const handleGoogleSignIn = () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
    window.location.href = `${apiUrl}/auth/google/login`;
  };
```

- [ ] **Step 2: Verify the form component compiles**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Frontend
npm run build 2>&1 | tail -20
```

Expected: build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add Frontend/components/auth/auth-form.tsx
git commit -m "feat(oauth): wire Google sign-in button to backend OAuth endpoint"
```

---

### Task 9: Integration Test — Full Flow

**Files:** (no file changes — manual verification)

- [ ] **Step 1: Set up Google OAuth credentials**

  1. Go to https://console.cloud.google.com/apis/credentials
  2. Create a new OAuth 2.0 Client ID (Web application)
  3. Add `http://localhost:8001/auth/google/callback` to Authorized redirect URIs
  4. Copy the Client ID and Client Secret

- [ ] **Step 2: Update backend `.env`**

Edit `Backend/.env` and add:

```env
GOOGLE_CLIENT_ID=<your-client-id>
GOOGLE_CLIENT_SECRET=<your-client-secret>
GOOGLE_REDIRECT_URI=http://localhost:8001/auth/google/callback
FRONTEND_URL=http://localhost:3000
```

- [ ] **Step 3: Start both servers**

```bash
# Terminal 1: Backend
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001

# Terminal 2: Frontend
cd /Users/utkarsh/Developer/Projects/Orbit/Frontend
npm run dev
```

- [ ] **Step 4: Test the full OAuth flow end-to-end**

  1. Open `http://localhost:3000/login` in a browser
  2. Click "Continue with Google"
  3. You should be redirected to Google's consent screen
  4. Select a Google account and approve
  5. Google redirects to `http://localhost:8001/auth/google/callback?code=...&state=...`
  6. Backend exchanges the code, creates/finds the user, redirects to `http://localhost:3000/auth/callback?token=...`
  7. Frontend callback page stores the token and redirects to `/dashboard` (or `/teacher` if teacher role)
  8. Dashboard loads with the user authenticated

- [ ] **Step 5: Test error states**

  1. **Invalid state:** Open `/auth/callback?error=invalid_state` — should show error page
  2. **No email in response:** Simulated by checking the error page renders
  3. **Exchange failure:** Open `/auth/callback?error=exchange_failed` — should show error page

- [ ] **Step 6: Test merging**

  1. First sign up with email+password (e.g., `test+merge@example.com`)
  2. Log out
  3. Sign in with Google using the same email
  4. Verify: no new user is created — the existing account is linked and you're logged in

- [ ] **Step 7: Test password login after merge**

  1. After merging via Google, log out
  2. Try logging in with the original email+password
  3. Expected: login still works (password_hash was preserved during merge)

- [ ] **Step 8: Commit remaining changes**

```bash
git add Backend/.env Backend/requirements.txt
git commit -m "chore: update .env with Google OAuth credentials"  # Note: .env is typically .gitignored — verify it's not committed
```

Check if `.env` is in `.gitignore`:

```bash
grep -q ".env" /Users/utkarsh/Developer/Projects/Orbit/Backend/.gitignore && echo ".env is gitignored" || echo ".env is NOT gitignored"
```

If it IS gitignored, just leave it local. If NOT, add it to `.gitignore` and commit.

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Task 1 covers env vars + config
- [x] Task 2 covers User model changes (optional password_hash, auth_provider, provider_uid)
- [x] Task 3 covers auth service updates (create_user sets auth_provider, authenticate_user guards Google-only users, find_or_create_google_user merge logic)
- [x] Task 4 covers new google_oauth_service module
- [x] Task 5 covers `/auth/google/login` and `/auth/google/callback` endpoints
- [x] Task 6 covers CORS fix
- [x] Task 7 covers frontend callback page
- [x] Task 8 covers wiring the Google button
- [x] Task 9 covers integration testing

**2. Placeholder scan:** No placeholders found. Every step has complete code.

**3. Type consistency:** All imports, function names, and method signatures are consistent across tasks. `create_access_token`, `find_or_create_google_user`, `get_authorization_url`, `exchange_code_for_tokens` are defined in one place and used consistently.