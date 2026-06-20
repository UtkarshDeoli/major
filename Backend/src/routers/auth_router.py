from datetime import timedelta
import logging
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, status
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

logger = logging.getLogger(__name__)
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
    response = RedirectResponse(url=authorization_url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="oauth_state",
        value=state,
        max_age=600,
        httponly=True,
        secure=False,
        samesite="lax",
    )
    return response


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    """Handle the OAuth callback from Google.

    Exchanges the authorization code for tokens, finds or creates a user,
    issues a JWT, and redirects the frontend with the token.
    """
    # If Google returned an error (e.g., user denied consent)
    if error:
        logger.warning("Google OAuth error: %s", error)
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?error={quote(error)}",
            status_code=status.HTTP_302_FOUND,
        )

    if not code or not state:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?error=invalid_request",
            status_code=status.HTTP_302_FOUND,
        )

    # Verify CSRF state cookie matches
    stored_state = request.cookies.get("oauth_state")
    if not stored_state or stored_state != state:
        logger.warning("OAuth state mismatch: stored=%s received=%s", stored_state, state)
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?error=invalid_state",
            status_code=status.HTTP_302_FOUND,
        )

    try:
        id_info = exchange_code_for_tokens(code, state)
        email = id_info.get("email")
        if not email:
            return RedirectResponse(
                url=f"{FRONTEND_URL}/auth/callback?error=no_email",
                status_code=status.HTTP_302_FOUND,
            )

        # Verify Google has confirmed the email
        if not id_info.get("email_verified"):
            logger.warning("Google email not verified: %s", email)
            return RedirectResponse(
                url=f"{FRONTEND_URL}/auth/callback?error=email_not_verified",
                status_code=status.HTTP_302_FOUND,
            )

        name = id_info.get("name")
        google_sub = id_info.get("sub")
        if not google_sub:
            logger.warning("Google sub (user ID) missing for email: %s", email)
            return RedirectResponse(
                url=f"{FRONTEND_URL}/auth/callback?error=exchange_failed",
                status_code=status.HTTP_302_FOUND,
            )

        user = await find_or_create_google_user(
            email=email,
            name=name,
            google_sub=google_sub,
        )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )
        response = RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?token={access_token}",
            status_code=status.HTTP_302_FOUND,
        )
        response.delete_cookie("oauth_state")
        logger.info("Google OAuth success: email=%s sub=%s", email, google_sub)
        return response
    except Exception as e:
        logger.exception("Google OAuth exchange failed: %s", e)
        return RedirectResponse(
            url=f"{FRONTEND_URL}/auth/callback?error=exchange_failed",
            status_code=status.HTTP_302_FOUND,
        )