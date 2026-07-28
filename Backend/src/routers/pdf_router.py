from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Path, Form, Request
from fastapi.responses import FileResponse
from typing import List, Optional
import os
from src.core.models import PDFMetadata, PDFListResponse, PDFUploadResponse
from src.core.security import get_current_user
from src.core.plan_enforcement import enforce_limit
from src.core.limiter import limiter, UPLOAD_LIMIT
from src.core.data_store import (
    get_user_pdfs,
    get_pdf_metadata,
    store_pdf_metadata,
    update_pdf_metadata,
    store_document_chunks,
)
from src.services.document_processor import extract_text_from_pdf, chunk_document
from src.services.vector_store import VectorStore
import uuid

router = APIRouter(prefix="/pdfs", tags=["PDFs"])


@router.post(
    "/upload",
    response_model=PDFUploadResponse,
    summary="Upload a PDF file",
    description="Upload a PDF file to be processed and stored. The file will be indexed for RAG chat and made available for querying.",
)
@limiter.limit(UPLOAD_LIMIT)
async def upload_pdf(
    request: Request,
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    tags: Optional[List[str]] = Form(None),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("doc_storage")),
):
    """
    Upload a PDF file for processing, storage, and RAG indexing.
    """
    # Validate the file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:
        # Read the file content
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
            title=title,
            description=description,
            tags=tags or [],
        )

        # Extract and chunk
        doc_type = "pdf"
        text, page_count = extract_text_from_pdf(content)
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
                "tags": tags or [],
            })

        # Store in ChromaDB (use a fresh singleton lookup so test resets are respected)
        VectorStore().add_chunks(user_id, chroma_chunks)

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
                "tags": tags or [],
                "page_count": page_count,
            }
        )

        updated = await get_pdf_metadata(pdf_metadata["id"])

        return PDFUploadResponse(
            id=updated["id"],
            filename=updated["filename"],
            size=updated["size"],
            upload_date=updated["upload_date"],
            user_id=updated["user_id"],
            file_path=updated["file_path"],
            processed=updated["processed"],
            tags=updated.get("tags", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading PDF: {str(e)}"
        )

@router.get(
    "/", 
    response_model=PDFListResponse,
    summary="List all PDFs for the current user",
    description="List all PDFs that have been uploaded by the current user."
)
async def list_pdfs(user_id: str = Depends(get_current_user)):
    """
    List all PDFs for the current user.
    """
    try:
        # Get all PDFs for the user
        pdf_list = await get_user_pdfs(user_id)
        
        return PDFListResponse(
            pdfs=[
                PDFMetadata(
                    id=pdf["id"],
                    filename=pdf["filename"],
                    size=pdf["size"],
                    upload_date=pdf["upload_date"],
                    user_id=pdf["user_id"],
                    file_path=pdf["file_path"],
                    processed=pdf["processed"],
                    title=pdf.get("title"),
                    description=pdf.get("description"),
                    page_count=pdf.get("page_count"),
                    vector_db_path=pdf.get("vector_db_path"),
                    tags=pdf.get("tags", [])
                )
                for pdf in pdf_list
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error listing PDFs: {str(e)}"
        )

@router.get(
    "/{pdf_id}", 
    response_model=PDFMetadata,
    summary="Get PDF metadata",
    description="Get metadata for a specific PDF."
)
async def get_pdf(
    pdf_id: str = Path(..., description="The ID of the PDF"),
    user_id: str = Depends(get_current_user)
):
    """
    Get metadata for a specific PDF.
    """
    try:
        # Get PDF metadata
        pdf = await get_pdf_metadata(pdf_id)
        
        if not pdf:
            raise HTTPException(
                status_code=404, 
                detail=f"PDF with ID {pdf_id} not found"
            )
        
        # Check if the PDF belongs to the user
        if pdf["user_id"] != user_id:
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to access this PDF"
            )
        
        return PDFMetadata(
            id=pdf["id"],
            filename=pdf["filename"],
            size=pdf["size"],
            upload_date=pdf["upload_date"],
            user_id=pdf["user_id"],
            file_path=pdf["file_path"],
            processed=pdf["processed"],
            title=pdf.get("title"),
            description=pdf.get("description"),
            page_count=pdf.get("page_count"),
            vector_db_path=pdf.get("vector_db_path"),
            tags=pdf.get("tags", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting PDF metadata: {str(e)}"
        )

@router.get(
    "/{pdf_id}/download",
    summary="Download a PDF file",
    description="Download the original PDF file."
)
async def download_pdf(
    pdf_id: str = Path(..., description="The ID of the PDF"),
    user_id: str = Depends(get_current_user)
):
    """
    Download the original PDF file.
    """
    try:
        # Get PDF metadata
        pdf = await get_pdf_metadata(pdf_id)
        
        if not pdf:
            raise HTTPException(
                status_code=404, 
                detail=f"PDF with ID {pdf_id} not found"
            )
        
        # Check if the PDF belongs to the user
        if pdf["user_id"] != user_id:
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to access this PDF"
            )
        
        # Check if the file exists
        if not os.path.exists(pdf["file_path"]):
            raise HTTPException(
                status_code=404, 
                detail=f"PDF file not found on server"
            )
        
        # Return the file
        return FileResponse(
            path=pdf["file_path"],
            filename=pdf["filename"],
            media_type="application/pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error downloading PDF: {str(e)}"
        )
