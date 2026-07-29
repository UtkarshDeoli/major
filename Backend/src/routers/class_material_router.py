"""ClassMaterial endpoints."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from pydantic import BaseModel

from src.core.security import require_role
from src.core.plan_enforcement import enforce_limit
from src.services import class_material_service as svc

router = APIRouter(prefix="/classes/{class_id}", tags=["Class Materials"])


class ClassMaterialResponse(BaseModel):
    id: str
    class_id: str
    class_subject_id: str
    teacher_id: str
    name: str
    type: str
    size: int
    doc_id: Optional[str] = None
    rag_indexed: bool = False
    page_count: Optional[int] = None
    created_at: datetime


class ClassMaterialListResponse(BaseModel):
    materials: List[ClassMaterialResponse]


@router.post("/subjects/{subject_id}/materials", response_model=ClassMaterialResponse, status_code=status.HTTP_201_CREATED)
async def upload_material(
    file: UploadFile = File(...),
    class_id: str = Path(...),
    subject_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("doc_storage")),
):
    result = await svc.upload_class_material(class_id, subject_id, file, teacher["email"], _plan)
    return ClassMaterialResponse(**result)


@router.get("/subjects/{subject_id}/materials", response_model=ClassMaterialListResponse)
async def list_materials(
    class_id: str = Path(...),
    subject_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    mats = await svc.list_materials(class_id, subject_id, teacher["email"])
    return ClassMaterialListResponse(materials=[ClassMaterialResponse(**m) for m in mats])


@router.delete("/materials/{material_id}", status_code=status.HTTP_200_OK)
async def delete_material(
    material_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    return await svc.remove_class_material(class_id, material_id, teacher["email"])
