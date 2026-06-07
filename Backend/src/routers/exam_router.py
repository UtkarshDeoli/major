from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.models import Exam
from src.core.security import get_current_user
from src.core.data_store import exams_collection, users_collection
from bson import ObjectId

router = APIRouter(prefix="/api/exams", tags=["Exams"])


class ExamCreate(BaseModel):
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    is_active: bool = False


class ExamResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    is_active: bool = False
    created_at: datetime
    updated_at: datetime


class ExamListResponse(BaseModel):
    exams: List[ExamResponse]


class ActiveExamResponse(BaseModel):
    exam_id: str
    is_active: bool


@router.get("/", response_model=ExamListResponse)
async def list_exams(user_email: str = Depends(get_current_user)):
    """List all exams for the current user."""
    if exams_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    
    try:
        cursor = exams_collection.find({"user_id": user_email}).sort("created_at", -1)
        exams = []
        async for exam in cursor:
            exam["id"] = str(exam["_id"])
            del exam["_id"]
            exams.append(ExamResponse(**exam))
        
        return ExamListResponse(exams=exams)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing exams: {str(e)}"
        )


@router.post("/", response_model=ExamResponse, status_code=status.HTTP_201_CREATED)
async def create_exam(exam_data: ExamCreate, user_email: str = Depends(get_current_user)):
    """Create a new exam for the current user."""
    if exams_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    
    try:
        exam = Exam(
            user_id=user_email,
            name=exam_data.name,
            description=exam_data.description,
            icon=exam_data.icon,
            color=exam_data.color,
            is_active=exam_data.is_active,
        )
        
        result = await exams_collection.insert_one(exam.model_dump(by_alias=True))
        created_exam = await exams_collection.find_one({"_id": result.inserted_id})
        created_exam["id"] = str(created_exam["_id"])
        del created_exam["_id"]
        
        return ExamResponse(**created_exam)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating exam: {str(e)}"
        )


@router.patch("/{exam_id}/active", response_model=ActiveExamResponse)
async def set_active_exam(exam_id: str, user_email: str = Depends(get_current_user)):
    """Set an exam as active and deactivate all others."""
    if exams_collection is None or users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    
    try:
        # Verify the exam exists and belongs to the user
        exam = await exams_collection.find_one({
            "_id": ObjectId(exam_id),
            "user_id": user_email
        })
        
        if not exam:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exam not found"
            )
        
        # Deactivate all other exams for this user
        await exams_collection.update_many(
            {"user_id": user_email},
            {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}}
        )
        
        # Activate the selected exam
        await exams_collection.update_one(
            {"_id": ObjectId(exam_id)},
            {"$set": {"is_active": True, "updated_at": datetime.now(timezone.utc)}}
        )
        
        # Update user's active_exam_id
        await users_collection.update_one(
            {"email": user_email},
            {"$set": {"active_exam_id": exam_id, "updated_at": datetime.now(timezone.utc)}}
        )
        
        return ActiveExamResponse(exam_id=exam_id, is_active=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting active exam: {str(e)}"
        )
