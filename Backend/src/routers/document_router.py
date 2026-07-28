from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Path, Form
from typing import Dict, List, Optional
from pydantic import BaseModel

from src.core.models import DocumentUploadResponse, DocumentListResponse, PDFMetadata
from src.core.security import get_current_user
from src.core.plan_enforcement import enforce_limit
from src.core.data_store import (
    store_pdf_metadata,
    update_pdf_metadata,
    get_user_pdfs,
    get_pdf_metadata,
    store_document_chunks,
    update_chunk_tags,
)
from src.services.vector_store import VectorStore
from src.services.document_processor import detect_doc_type, extract_text_from_pdf, chunk_document
import uuid
import os

router = APIRouter(prefix="/documents", tags=["Documents"])

vector_store = VectorStore()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # comma-separated
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("doc_storage")),
):
    """Upload any document type."""
    doc_type = detect_doc_type(file.filename)
    if doc_type == "unknown":
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Read file content
    content = await file.read()
    
    # Save file to disk
    user_dir = os.path.join("uploads", user_id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Store metadata
    pdf_metadata = await store_pdf_metadata(
        filename=file.filename,
        size=len(content),
        user_id=user_id,
        file_path=file_path,
        title=file.filename,
        tags=tags.split(",") if tags else [],
    )
    
    # Extract and chunk
    page_count = None
    if doc_type == "pdf":
        text, page_count = extract_text_from_pdf(content)
        chunks_data = chunk_document(text, doc_type="pdf")
    elif doc_type in ["txt", "md"]:
        text = content.decode('utf-8', errors='ignore')
        chunks_data = chunk_document(text, doc_type=doc_type)
    else:
        # Other types - basic extraction
        text = content.decode('utf-8', errors='ignore')
        chunks_data = chunk_document(text, doc_type=doc_type)
    
    # Generate embeddings and store
    model = VectorStore.get_embedding_model()
    
    chroma_chunks = []
    for chunk in chunks_data:
        chroma_id = str(uuid.uuid4())
        embedding = model.encode(chunk["content"]).tolist()
        chroma_chunks.append({
            "chroma_id": chroma_id,
            "user_id": user_id,
            "doc_id": pdf_metadata["id"],
            "doc_name": file.filename,
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "embedding": embedding,
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "doc_type": doc_type,
            "subject": subject,
            "tags": tags.split(",") if tags else [],
        })
    
    # Store in ChromaDB
    vector_store.add_chunks(user_id, chroma_chunks)
    
    # Store in MongoDB
    await store_document_chunks(chroma_chunks)
    
    # Update metadata
    await update_pdf_metadata(
        pdf_metadata["id"],
        {
            "processed": True,
            "chunk_count": len(chroma_chunks),
            "doc_type": doc_type,
            "subject": subject,
            "tags": tags.split(",") if tags else [],
            "page_count": page_count,
        }
    )
    
    return DocumentUploadResponse(
        id=pdf_metadata["id"],
        filename=file.filename,
        doc_type=doc_type,
        size=len(content),
        chunk_count=len(chroma_chunks),
        page_count=page_count,
        subject=subject,
        tags=tags.split(",") if tags else [],
        processed=True,
        upload_date=pdf_metadata["upload_date"],
    )

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    subject: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user_id: str = Depends(get_current_user)
):
    """List all documents for the current user."""
    pdfs = await get_user_pdfs(user_id)

    # Apply filters
    if subject:
        pdfs = [p for p in pdfs if p.get("subject") == subject]
    if tags:
        pdfs = [p for p in pdfs if any(t in p.get("tags", []) for t in tags)]

    return DocumentListResponse(
        documents=[PDFMetadata(**p) for p in pdfs]
    )


class SubjectDocumentGroup(BaseModel):
    name: str
    documents: List[PDFMetadata]


class DocumentsBySubjectResponse(BaseModel):
    subjects: List[SubjectDocumentGroup]
    others: SubjectDocumentGroup


@router.get("/by-subject", response_model=DocumentsBySubjectResponse)
async def list_documents_by_subject(user_id: str = Depends(get_current_user)):
    """List all user documents grouped by subject; uncategorized docs go to Others."""
    pdfs = await get_user_pdfs(user_id)

    subject_buckets: Dict[str, List[PDFMetadata]] = {}
    others: List[PDFMetadata] = []

    for pdf in pdfs:
        meta = PDFMetadata(**pdf)
        if meta.subject:
            subject_buckets.setdefault(meta.subject, []).append(meta)
        else:
            others.append(meta)

    subjects = [
        SubjectDocumentGroup(name=name, documents=docs)
        for name, docs in sorted(subject_buckets.items(), key=lambda x: x[0].lower())
    ]

    return DocumentsBySubjectResponse(
        subjects=subjects,
        others=SubjectDocumentGroup(name="Others", documents=others),
    )

@router.post("/{doc_id}/tags")
async def update_document_tags(
    doc_id: str = Path(...),
    tags: List[str] = Form(...),
    user_id: str = Depends(get_current_user)
):
    """Update tags for a document and its chunks."""
    pdf = await get_pdf_metadata(doc_id)
    if not pdf or pdf["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Update document tags
    await update_pdf_metadata(doc_id, {"tags": tags})
    
    # Update chunk tags
    await update_chunk_tags(doc_id, tags)
    
    return {"success": True}
