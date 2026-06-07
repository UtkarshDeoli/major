from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel

from src.core.models import Material
from src.core.security import get_current_user
from src.core.data_store import (
    materials_collection,
    collections_collection,
    subjects_collection,
    exams_collection,
)
from bson import ObjectId

router = APIRouter(tags=["Materials"])


class MaterialResponse(BaseModel):
    id: str
    collection_id: str
    name: str
    type: str
    size: int
    url: str
    created_at: datetime
    updated_at: datetime
    rag_indexed: bool = False


class MaterialListResponse(BaseModel):
    materials: List[MaterialResponse]


class DeleteMaterialResponse(BaseModel):
    material_id: str
    deleted: bool


async def _verify_collection_access(collection_id: str, user_email: str):
    """Verify that the collection (and its exam chain) belongs to the current user."""
    if collections_collection is None or subjects_collection is None or exams_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    collection = await collections_collection.find_one({"_id": ObjectId(collection_id)})
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    subject = await subjects_collection.find_one({"_id": ObjectId(collection["subject_id"])})
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subject not found"
        )

    exam = await exams_collection.find_one({
        "_id": ObjectId(subject["exam_id"]),
        "user_id": user_email
    })
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exam not found"
        )

    return collection


@router.get("/api/collections/{collection_id}/materials", response_model=MaterialListResponse)
async def list_materials(collection_id: str, user_email: str = Depends(get_current_user)):
    """List all materials for a given collection."""
    if materials_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        await _verify_collection_access(collection_id, user_email)

        cursor = materials_collection.find({"collection_id": collection_id}).sort("created_at", -1)
        materials = []
        async for mat in cursor:
            mat["id"] = str(mat["_id"])
            del mat["_id"]
            materials.append(MaterialResponse(**mat))

        return MaterialListResponse(materials=materials)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing materials: {str(e)}"
        )


@router.post("/api/collections/{collection_id}/materials", response_model=MaterialResponse, status_code=status.HTTP_201_CREATED)
async def upload_material(
    collection_id: str,
    file: UploadFile = File(...),
    user_email: str = Depends(get_current_user)
):
    """Upload a new material to a collection."""
    if materials_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        await _verify_collection_access(collection_id, user_email)

        file_type = "pdf" if file.filename.lower().endswith(".pdf") else "text"
        # Read file size
        content = await file.read()
        size = len(content)

        material = Material(
            collection_id=collection_id,
            name=file.filename,
            type=file_type,  # type: ignore[arg-type]
            size=size,
            url=f"/uploads/{file.filename}",
        )

        result = await materials_collection.insert_one(material.model_dump(by_alias=True))
        created = await materials_collection.find_one({"_id": result.inserted_id})
        created["id"] = str(created["_id"])
        del created["_id"]

        return MaterialResponse(**created)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading material: {str(e)}"
        )


@router.delete("/api/materials/{material_id}", response_model=DeleteMaterialResponse)
async def delete_material(
    material_id: str,
    user_email: str = Depends(get_current_user)
):
    """Delete a material."""
    if materials_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        material = await materials_collection.find_one({"_id": ObjectId(material_id)})

        if not material:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Material not found"
            )

        # Verify access through the collection chain
        await _verify_collection_access(material["collection_id"], user_email)

        await materials_collection.delete_one({"_id": ObjectId(material_id)})

        return DeleteMaterialResponse(material_id=material_id, deleted=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting material: {str(e)}"
        )
