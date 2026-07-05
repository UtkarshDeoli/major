"""AI-generated study material endpoints (summaries, notes).

These power the right-hand sidebar in the NotebookLM-style chat: a student (or
their teacher) can generate a summary grounded on selected materials, and it is
stored and listed alongside teacher-created material.
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from src.core.models import GenerateSummaryRequest, AIStudyMaterialResponse, AIStudyMaterialListResponse
from src.core.security import get_current_user
from src.core.data_store import (
    materials_collection,
    store_ai_material,
    get_user_ai_materials,
    get_ai_material,
    delete_ai_material,
)
from src.services.gemini_service import gemini_service, generate_summary

router = APIRouter(prefix="/ai-materials", tags=["AI Study Materials"])


async def _gather_content(user_id: str, material_ids: List[str], doc_ids: List[str]) -> str:
    from bson import ObjectId
    from src.core.data_store import get_pdf_metadata

    parts: List[str] = []
    resolved_doc_ids: List[str] = list(doc_ids or [])
    if material_ids and materials_collection is not None:
        try:
            cursor = materials_collection.find({"_id": {"$in": [ObjectId(m) for m in material_ids]}})
            async for mat in cursor:
                did = mat.get("doc_id")
                if did:
                    resolved_doc_ids.append(did)
        except Exception:
            pass

    for did in resolved_doc_ids:
        pdf = await get_pdf_metadata(did)
        if not pdf or pdf.get("user_id") != user_id:
            continue
        file_path = pdf.get("file_path")
        if not file_path:
            continue
        try:
            if file_path.lower().endswith(".pdf"):
                text = await gemini_service.extract_text_from_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if text:
                parts.append(text)
        except Exception as e:
            print(f"Skipping doc {did} for summary: {e}")
    return "\n\n".join(parts)[:8000]


@router.post("/summarize", response_model=AIStudyMaterialResponse, status_code=status.HTTP_201_CREATED)
async def summarize_material(
    request: GenerateSummaryRequest,
    user_id: str = Depends(get_current_user),
):
    """Generate a study summary from materials/documents."""
    content = await _gather_content(user_id, request.material_ids, request.doc_ids)
    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable material found to summarize")

    summary = await generate_summary(content, style=request.style, subject=request.subject)
    if not summary.strip():
        raise HTTPException(status_code=502, detail="Failed to generate summary from the material")

    now = datetime.now(timezone.utc)
    title = request.title or "Summary"
    doc = {
        "user_id": user_id,
        "created_by": None,
        "kind": "summary",
        "title": title,
        "subject": request.subject,
        "source_material_ids": request.material_ids,
        "content": summary,
        "ref_id": None,
        "created_at": now,
    }
    material_id = await store_ai_material(doc)
    doc["id"] = material_id
    return AIStudyMaterialResponse(
        id=material_id, user_id=user_id, created_by=None, kind="summary",
        title=title, subject=request.subject, source_material_ids=request.material_ids,
        content=summary, ref_id=None, created_at=now,
    )


@router.get("/", response_model=AIStudyMaterialListResponse)
async def list_ai_materials(user_id: str = Depends(get_current_user)):
    mats = await get_user_ai_materials(user_id)
    return AIStudyMaterialListResponse(materials=[
        AIStudyMaterialResponse(
            id=m["id"], user_id=m["user_id"], created_by=m.get("created_by"),
            kind=m.get("kind", "summary"), title=m.get("title", "Material"),
            subject=m.get("subject"), source_material_ids=m.get("source_material_ids", []),
            content=m.get("content", ""), ref_id=m.get("ref_id"),
            created_at=m["created_at"],
        ) for m in mats
    ])


@router.get("/{material_id}", response_model=AIStudyMaterialResponse)
async def get_one_material(material_id: str = Path(...), user_id: str = Depends(get_current_user)):
    mat = await get_ai_material(material_id)
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    allowed = {mat.get("user_id"), mat.get("created_by")}
    if user_id not in allowed:
        raise HTTPException(status_code=403, detail="Not authorized to view this material")
    return AIStudyMaterialResponse(
        id=mat["id"], user_id=mat["user_id"], created_by=mat.get("created_by"),
        kind=mat.get("kind", "summary"), title=mat.get("title", "Material"),
        subject=mat.get("subject"), source_material_ids=mat.get("source_material_ids", []),
        content=mat.get("content", ""), ref_id=mat.get("ref_id"),
        created_at=mat["created_at"],
    )


@router.delete("/{material_id}", status_code=status.HTTP_200_OK)
async def remove_material(material_id: str = Path(...), user_id: str = Depends(get_current_user)):
    mat = await get_ai_material(material_id)
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    if mat.get("user_id") != user_id and mat.get("created_by") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this material")
    await delete_ai_material(material_id)
    return {"material_id": material_id, "deleted": True}