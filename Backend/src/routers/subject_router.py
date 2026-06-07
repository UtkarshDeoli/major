from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.models import Subject
from src.core.security import get_current_user
from src.core.data_store import subjects_collection, exams_collection
from bson import ObjectId

router = APIRouter(prefix="/api/exams", tags=["Subjects"])


class SubjectCreate(BaseModel):
    name: str
    icon: Optional[str] = None


class SubjectResponse(BaseModel):
    id: str
    exam_id: str
    name: str
    icon: Optional[str] = None
    progress: int = 0
    last_studied_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class SubjectListResponse(BaseModel):
    subjects: List[SubjectResponse]


async def _verify_exam_ownership(exam_id: str, user_email: str):
    """Verify that the exam exists and belongs to the current user."""
    if exams_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    exam = await exams_collection.find_one({
        "_id": ObjectId(exam_id),
        "user_id": user_email
    })

    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exam not found"
        )

    return exam


@router.get("/{exam_id}/subjects", response_model=SubjectListResponse)
async def list_subjects(exam_id: str, user_email: str = Depends(get_current_user)):
    """List all subjects for a given exam."""
    if subjects_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        # Verify exam ownership
        await _verify_exam_ownership(exam_id, user_email)

        cursor = subjects_collection.find({"exam_id": exam_id}).sort("created_at", -1)
        subjects = []
        async for subject in cursor:
            subject["id"] = str(subject["_id"])
            del subject["_id"]
            subjects.append(SubjectResponse(**subject))

        return SubjectListResponse(subjects=subjects)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing subjects: {str(e)}"
        )


@router.post("/{exam_id}/subjects", response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
async def create_subject(
    exam_id: str,
    subject_data: SubjectCreate,
    user_email: str = Depends(get_current_user)
):
    """Create a new subject under a given exam."""
    if subjects_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        # Verify exam ownership
        await _verify_exam_ownership(exam_id, user_email)

        subject = Subject(
            exam_id=exam_id,
            name=subject_data.name,
            icon=subject_data.icon,
        )

        result = await subjects_collection.insert_one(subject.model_dump(by_alias=True))
        created_subject = await subjects_collection.find_one({"_id": result.inserted_id})
        created_subject["id"] = str(created_subject["_id"])
        del created_subject["_id"]

        return SubjectResponse(**created_subject)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating subject: {str(e)}"
        )
