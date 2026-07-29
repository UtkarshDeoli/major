"""Class-scoped material upload + RAG indexing."""
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, UploadFile

from src.core.data_store import (
    classes_collection,
    class_subjects_collection,
    class_materials_collection,
    get_class_by_id,
    get_class_subject_by_id,
    get_class_material_by_id,
    get_pdf_metadata,
    list_class_materials,
    store_class_material,
    delete_class_material,
    store_pdf_metadata,
    update_pdf_metadata,
    delete_pdf_metadata,
    store_document_chunks,
    delete_document_chunks,
)
from src.services.vector_store import VectorStore
from src.services.document_processor import detect_doc_type, extract_text_from_pdf, chunk_document

vector_store = VectorStore()


async def _require_class_teacher(class_id: str, teacher_email: str) -> dict:
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to manage this class")
    return cls


async def upload_class_material(
    class_id: str,
    class_subject_id: str,
    file: UploadFile,
    teacher_email: str,
    _plan: dict,  # passed from router dependency for limit enforcement
) -> dict:
    if any(c is None for c in (classes_collection, class_subjects_collection, class_materials_collection)):
        raise HTTPException(status_code=503, detail="Database connection not available")

    cls = await _require_class_teacher(class_id, teacher_email)
    subject = await get_class_subject_by_id(class_subject_id)
    if not subject or subject.get("class_id") != class_id:
        raise HTTPException(status_code=404, detail="Subject not found in this class")

    subject_name = subject.get("name")
    tags = [cls.get("name", ""), subject_name] if cls.get("name") else [subject_name]

    doc_type = detect_doc_type(file.filename)
    if doc_type == "unknown":
        doc_type = "text"

    content = await file.read()
    size = len(content)

    user_dir = os.path.join("uploads", teacher_email)
    os.makedirs(user_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(user_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(content)

    file_type = "pdf" if file.filename.lower().endswith(".pdf") else "text"

    now = datetime.now(timezone.utc)
    material_doc = {
        "class_id": class_id,
        "class_subject_id": class_subject_id,
        "teacher_id": teacher_email,
        "name": file.filename,
        "type": file_type,
        "size": size,
        "doc_id": None,
        "rag_indexed": False,
        "page_count": None,
        "created_at": now,
        "updated_at": now,
    }
    material_id = await store_class_material(material_doc)

    doc_id = None
    chunk_count = 0
    rag_indexed = False
    page_count = None

    try:
        pdf_meta = await store_pdf_metadata(
            filename=file.filename,
            size=size,
            user_id=teacher_email,
            file_path=file_path,
            title=file.filename,
            tags=tags,
        )
        doc_id = pdf_meta["id"]

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
                "user_id": teacher_email,
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
            vector_store.add_chunks(teacher_email, chroma_chunks)
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
        print(f"Class material RAG indexing failed for {material_id}: {index_err}")

    await class_materials_collection.update_one(
        {"_id": __import__("bson").ObjectId(material_id)},
        {"$set": {
            "doc_id": doc_id,
            "rag_indexed": rag_indexed,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "updated_at": datetime.now(timezone.utc),
        }}
    )

    created = await get_class_material_by_id(material_id)
    if created:
        created["id"] = created.pop("_id")
    return created


async def list_materials(class_id: str, class_subject_id: Optional[str], teacher_email: str) -> List[dict]:
    await _require_class_teacher(class_id, teacher_email)
    mats = await list_class_materials(class_id, class_subject_id)
    for m in mats:
        m["id"] = m.pop("_id")
    return mats


async def _delete_material_cascade(mat: dict) -> None:
    """Remove a class material and all associated PDF/file/vector data."""
    doc_id = mat.get("doc_id")
    if doc_id:
        pdf = await get_pdf_metadata(doc_id)
        if pdf:
            file_path = pdf.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            vector_store.delete_document_chunks(pdf.get("user_id"), doc_id)
            await delete_document_chunks(doc_id)
            await delete_pdf_metadata(doc_id)
    await delete_class_material(mat["id"])


async def remove_class_subject_materials(class_id: str, class_subject_id: str, teacher_email: str) -> int:
    """Delete all materials for a subject and their associated PDF/file/vector data."""
    await _require_class_teacher(class_id, teacher_email)
    mats = await list_class_materials(class_id, class_subject_id)
    deleted = 0
    for m in mats:
        if "_id" in m and "id" not in m:
            m["id"] = str(m.pop("_id"))
        await _delete_material_cascade(m)
        deleted += 1
    return deleted


async def remove_class_material(class_id: str, material_id: str, teacher_email: str) -> dict:
    if class_materials_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    await _require_class_teacher(class_id, teacher_email)
    mat = await get_class_material_by_id(material_id)
    if not mat or mat.get("class_id") != class_id:
        raise HTTPException(status_code=404, detail="Material not found in this class")
    if "_id" in mat and "id" not in mat:
        mat["id"] = str(mat.pop("_id"))
    await _delete_material_cascade(mat)
    return {"material_id": material_id, "deleted": True}


async def get_class_material(class_id: str, material_id: str) -> Optional[dict]:
    mat = await get_class_material_by_id(material_id)
    if not mat or mat.get("class_id") != class_id:
        return None
    return mat
