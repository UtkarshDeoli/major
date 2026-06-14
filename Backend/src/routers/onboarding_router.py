from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.security import get_current_user
from src.core.data_store import users_collection

router = APIRouter(tags=["Onboarding"])


class OnboardingData(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    institute: Optional[str] = None
    preferred_language: Optional[str] = None


class OnboardingStatus(BaseModel):
    onboarding_completed: bool
    name: Optional[str] = None
    role: Optional[str] = None
    institute: Optional[str] = None
    preferred_language: Optional[str] = None


class OnboardingCompleteResponse(BaseModel):
    onboarding_completed: bool


@router.post("/api/onboarding/", response_model=OnboardingStatus)
async def save_onboarding(
    data: OnboardingData,
    user_email: str = Depends(get_current_user)
):
    """Save onboarding data for the current user."""
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        update_fields = {}
        if data.name is not None:
            update_fields["name"] = data.name.strip()
        # Ignore any role sent from onboarding; roles are assigned through
        # admin/sub-admin enrollment or during the signup flow.
        if data.institute is not None:
            update_fields["institute"] = data.institute.strip()
        if data.preferred_language is not None:
            update_fields["preferred_language"] = data.preferred_language.strip()
        update_fields["updated_at"] = datetime.now(timezone.utc)

        await users_collection.update_one(
            {"email": user_email},
            {"$set": update_fields}
        )

        user = await users_collection.find_one({"email": user_email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return OnboardingStatus(
            onboarding_completed=user.get("onboarding_completed", False),
            name=user.get("name"),
            role=user.get("role"),
            institute=user.get("institute"),
            preferred_language=user.get("preferred_language", "en"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving onboarding data: {str(e)}"
        )


@router.get("/api/onboarding/", response_model=OnboardingStatus)
async def get_onboarding_status(user_email: str = Depends(get_current_user)):
    """Get the current onboarding status for the user."""
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        user = await users_collection.find_one({"email": user_email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return OnboardingStatus(
            onboarding_completed=user.get("onboarding_completed", False),
            role=user.get("role"),
            institute=user.get("institute"),
            preferred_language=user.get("preferred_language", "en"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching onboarding status: {str(e)}"
        )


@router.post("/api/onboarding/complete", response_model=OnboardingCompleteResponse)
async def complete_onboarding(user_email: str = Depends(get_current_user)):
    """Mark onboarding as complete for the current user."""
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        await users_collection.update_one(
            {"email": user_email},
            {
                "$set": {
                    "onboarding_completed": True,
                    "updated_at": datetime.now(timezone.utc),
                }
            }
        )

        return OnboardingCompleteResponse(onboarding_completed=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error completing onboarding: {str(e)}"
        )
