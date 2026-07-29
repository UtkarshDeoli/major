"""ClassSubject endpoints."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from src.core.security import require_role
from src.services import class_subject_service as svc

router = APIRouter(prefix="/classes/{class_id}/subjects", tags=["Class Subjects"])


class SubjectCreateRequest(BaseModel):
    name: str
    icon: Optional[str] = None


class SubjectResponse(BaseModel):
    id: str
    class_id: str
    name: str
    icon: Optional[str] = None
    created_by: str
    created_at: datetime


class SubjectListResponse(BaseModel):
    subjects: List[SubjectResponse]


@router.post("/", response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
async def create_subject(
    request: SubjectCreateRequest,
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    result = await svc.create_class_subject(class_id, request.name, request.icon, teacher["email"])
    return SubjectResponse(**result)


@router.get("/", response_model=SubjectListResponse)
async def list_subjects(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    subs = await svc.list_subjects(class_id, teacher["email"])
    return SubjectListResponse(subjects=[SubjectResponse(**s) for s in subs])


@router.delete("/{subject_id}", status_code=status.HTTP_200_OK)
async def delete_subject(
    subject_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    return await svc.remove_class_subject(class_id, subject_id, teacher["email"])
