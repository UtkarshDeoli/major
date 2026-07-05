from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel

from src.core.models import Material
from src.core.security import get_current_user
from src.core.plan_enforcement import enforce_limit
from src.core.data_store import (
    materials_collection,
    collections_collection,
    subjects_collection,
    exams_collection,
    pdfs_collection,
    store_pdf_metadata,
    update_pdf_metadata,
    store_document_chunks,
)
from src.services.vector_store import VectorStore
from src.services.document_processor import detect_doc_type, extract_text_from_pdf, chunk_document
from bson import ObjectId
import uuid
import os

router = APIRouter(tags=["Materials"])

vector_store = VectorStore()


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
    doc_id: Optional[str] = None
    processed: bool = False
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None


class MaterialListResponse(BaseModel):
    materials: List[MaterialResponse]


class DeleteMaterialResponse(BaseModel):
    material_id: str
    deleted: bool


async def _verify_collection_access(collection_id: str, user_email: str):
    """Verify that the collection (and its exam chain) belongs to the current user.

    Returns (collection, subject, exam) for downstream use.
    """
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

    return collection, subject, exam


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
    user_email: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("doc_storage")),
):
    """Upload a new material to a collection.

    The file is saved to disk, text is extracted + chunked + embedded, and the
    chunks are indexed into ChromaDB + MongoDB so the material is searchable via
    RAG chat. The material record stores `doc_id` (the pdfs metadata id used as
    the RAG doc scope) and `rag_indexed=True` on success.
    """
    if materials_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    try:
        collection, subject, exam = await _verify_collection_access(collection_id, user_email)
        subject_name = subject.get("name")
        tags = [collection.get("name", "")] if collection.get("name") else []

        doc_type = detect_doc_type(file.filename)
        if doc_type == "unknown":
            # Fall back to text for unsupported extensions (still indexable if readable)
            doc_type = "text"

        content = await file.read()
        size = len(content)

        # Save file to disk under the owner's directory
        user_dir = os.path.join("uploads", user_email)
        os.makedirs(user_dir, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(user_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(content)

        file_type = "pdf" if file.filename.lower().endswith(".pdf") else "text"

        material = Material(
            collection_id=collection_id,
            name=file.filename,
            type=file_type,  # type: ignore[arg-type]
            size=size,
            url=f"/uploads/{user_email}/{safe_name}",
        )
        result = await materials_collection.insert_one(material.model_dump(by_alias=True))
        material_id = str(result.inserted_id)

        doc_id = None
        chunk_count = 0
        rag_indexed = False
        page_count = None

        try:
            # Store a pdfs metadata record so RAG can scope by doc_id
            pdf_meta = await store_pdf_metadata(
                filename=file.filename,
                size=size,
                user_id=user_email,
                file_path=file_path,
                title=file.filename,
                tags=tags,
            )
            doc_id = pdf_meta["id"]

            # Extract + chunk
            if doc_type == "pdf":
                text, page_count = extract_text_from_pdf(content)
            else:
                text = content.decode("utf-8", errors="ignore")
            chunks_data = chunk_document(text, doc_type=doc_type)

            model = VectorStore.get_embedding_model()
            chroma_chunks = []
            for chunk in chunks_data:
                chroma_id = str(uuid.uuid4())
                embedding = model.encode(chunk["content"]).tolist()
                chroma_chunks.append({
                    "chroma_id": chroma_id,
                    "user_id": user_email,
                    "doc_id": doc_id,
                    "doc_name": file.filename,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "embedding": embedding,
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "doc_type": doc_type,
                    "subject": subject_name,
                    "tags": tags,
                    "material_id": material_id,
                })

            if chroma_chunks:
                vector_store.add_chunks(user_email, chroma_chunks)
                await store_document_chunks(chroma_chunks)
                chunk_count = len(chroma_chunks)
                rag_indexed = True

            await update_pdf_metadata(doc_id, {
                "processed": True,
                "chunk_count": chunk_count,
                "doc_type": doc_type,
                "subject": subject_name,
                "tags": tags,
                "page_count": page_count,
                "material_id": material_id,
            })
        except Exception as index_err:
            # Indexing failure shouldn't lose the material record; surface it as a flag
            print(f"Material RAG indexing failed for {material_id}: {index_err}")

        # Update the material record with RAG linkage
        await materials_collection.update_one(
            {"_id": ObjectId(material_id)},
            {"$set": {
                "doc_id": doc_id,
                "rag_indexed": rag_indexed,
                "processed": rag_indexed,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "updated_at": datetime.now(timezone.utc),
            }}
        )

        created = await materials_collection.find_one({"_id": ObjectId(material_id)})
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
