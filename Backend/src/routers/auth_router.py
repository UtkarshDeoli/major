from datetime import timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from src.core.models import SubscriptionInfo
from src.core.security import get_current_user
from src.services.auth_service import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    create_user,
    get_user_by_email,
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
    """Register a new user with email and password and return an access token.

    Public signup always creates a student. Privileged roles must be assigned
    through admin/sub-admin enrollment flows.
    """
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