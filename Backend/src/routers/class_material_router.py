"""ClassMaterial endpoints."""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from pydantic import BaseModel

from src.core.security import require_role
from src.core.plan_enforcement import enforce_limit
from src.services import class_material_service as svc
from src.services.gemini_service import gemini_service, generate_flashcards
from src.services.mock_test_service import generate_mock_test_from_docs_service
from src.core.models import MockTestResponse
from src.core.data_store import (
    store_flashcard_deck,
    store_flashcards,
)

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


class GenerateFlashcardsFromMaterialResponse(BaseModel):
    deck_id: str
    card_count: int


class GenerateMockTestFromMaterialResponse(MockTestResponse):
    pass


async def _get_doc_content(doc_id: str, teacher_email: str) -> str:
    from src.core.data_store import get_pdf_metadata
    pdf = await get_pdf_metadata(doc_id)
    if not pdf or pdf.get("user_id") != teacher_email:
        raise HTTPException(status_code=404, detail="Source document not found")
    file_path = pdf.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="Source document has no file")
    try:
        if file_path.lower().endswith(".pdf"):
            text = await gemini_service.extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read source document: {e}")
    return text[:8000]


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


@router.post("/materials/{material_id}/generate-flashcards", response_model=GenerateFlashcardsFromMaterialResponse, status_code=status.HTTP_201_CREATED)
async def generate_flashcards_from_material(
    material_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("flashcard")),
):
    mat = await svc.get_class_material(class_id, material_id)
    if not mat or not mat.get("doc_id"):
        raise HTTPException(status_code=400, detail="Material is not indexed for AI generation")

    content = await _get_doc_content(mat["doc_id"], teacher["email"])
    cards_data = await generate_flashcards(content, num_cards=15, subject=mat.get("name"))
    if not cards_data:
        raise HTTPException(status_code=502, detail="Failed to generate flashcards")

    deck_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    deck_doc = {
        "id": deck_id,
        "user_id": teacher["email"],
        "title": f"Flashcards — {mat['name']}",
        "subject": mat.get("name"),
        "source_material_ids": [material_id],
        "source_type": "ai",
        "created_by": teacher["email"],
        "card_count": len(cards_data),
        "class_id": class_id,
        "class_subject_id": mat["class_subject_id"],
        "created_at": now,
        "updated_at": now,
    }
    await store_flashcard_deck(deck_doc)
    card_docs = [{
        "id": str(uuid.uuid4()),
        "deck_id": deck_id,
        "front": c["front"],
        "back": c["back"],
        "ease": 2,
        "interval_days": 0,
        "reps": 0,
        "due_at": now,
        "created_at": now,
    } for c in cards_data]
    await store_flashcards(card_docs)
    return GenerateFlashcardsFromMaterialResponse(deck_id=deck_id, card_count=len(card_docs))


@router.post("/materials/{material_id}/generate-mock-test", response_model=GenerateMockTestFromMaterialResponse, status_code=status.HTTP_201_CREATED)
async def generate_mock_test_from_material(
    material_id: str = Path(...),
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("mock_test")),
):
    mat = await svc.get_class_material(class_id, material_id)
    if not mat or not mat.get("doc_id"):
        raise HTTPException(status_code=400, detail="Material is not indexed for AI generation")

    mock_test = await generate_mock_test_from_docs_service(
        doc_ids=[mat["doc_id"]],
        num_mcq=10,
        num_text=3,
        total_marks=30,
        difficulty_level="medium",
        user_id=teacher["email"],
        subject=mat.get("name"),
        class_id=class_id,
        class_subject_id=mat["class_subject_id"],
        created_by=teacher["email"],
    )
    return mock_test
