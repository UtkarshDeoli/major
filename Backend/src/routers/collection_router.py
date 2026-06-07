from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.models import Collection
from src.core.security import get_current_user
from src.core.data_store import collections_collection, subjects_collection, exams_collection
from bson import ObjectId

router = APIRouter(tags=["Collections"])


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CollectionResponse(BaseModel):
    id: str
    subject_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CollectionListResponse(BaseModel):
    collections: List[CollectionResponse]


async def _verify_subject_access(subject_id: str, user_email: str):
    """Verify that the subject (and its exam) belongs to the current user."""
    if subjects_collection is None or exams_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    subject = await subjects_collection.find_one({"_id": ObjectId(subject_id)})
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

    return subject


@router.get("/api/subjects/{subject_id}/collections", response_model=CollectionListResponse)
async def list_collections(subject_id: str, user_email: str = Depends(get_current_user)):
    """List all collections for a given subject."""
    if collections_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        await _verify_subject_access(subject_id, user_email)

        cursor = collections_collection.find({"subject_id": subject_id}).sort("created_at", -1)
        collections = []
        async for col in cursor:
            col["id"] = str(col["_id"])
            del col["_id"]
            collections.append(CollectionResponse(**col))

        return CollectionListResponse(collections=collections)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )


@router.post("/api/subjects/{subject_id}/collections", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    subject_id: str,
    collection_data: CollectionCreate,
    user_email: str = Depends(get_current_user)
):
    """Create a new collection under a given subject."""
    if collections_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        await _verify_subject_access(subject_id, user_email)

        collection = Collection(
            subject_id=subject_id,
            name=collection_data.name,
            description=collection_data.description,
        )

        result = await collections_collection.insert_one(collection.model_dump(by_alias=True))
        created = await collections_collection.find_one({"_id": result.inserted_id})
        created["id"] = str(created["_id"])
        del created["_id"]

        return CollectionResponse(**created)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating collection: {str(e)}"
        )
