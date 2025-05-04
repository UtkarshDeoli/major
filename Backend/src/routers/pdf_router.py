from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Path
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import os
from datetime import datetime
import jwt
from pydantic import BaseModel

from src.core.models import PDFMetadata, PDFListResponse, PDFUploadResponse
from src.core.config import SECRET_KEY, ALGORITHM
from src.services.pdf_service import process_and_store_pdf
from src.core.data_store import get_user_pdfs, get_pdf_metadata

router = APIRouter(prefix="/pdfs", tags=["PDFs"])

# Helper function to get the current user from JWT token
async def get_current_user(token: str = Depends(lambda authorization: authorization)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token.replace("Bearer ", "")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication credentials"
        )


class UploadPDFRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@router.post(
    "/upload", 
    response_model=PDFUploadResponse,
    summary="Upload a PDF file",
    description="Upload a PDF file to be processed and stored. The file will be processed in the background and made available for querying.",
)
async def upload_pdf(
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Upload a PDF file for processing and storage.
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
        
        # Process and store the PDF
        pdf_metadata = await process_and_store_pdf(
            file_content=content,
            filename=file.filename,
            user_id=user_id,
            title=title,
            description=description,
            tags=tags
        )
        
        return PDFUploadResponse(
            id=pdf_metadata["id"],
            filename=pdf_metadata["filename"],
            size=pdf_metadata["size"],
            upload_date=pdf_metadata["upload_date"],
            user_id=pdf_metadata["user_id"],
            file_path=pdf_metadata["file_path"],
            processed=pdf_metadata["processed"],
            tags=pdf_metadata.get("tags", [])
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
