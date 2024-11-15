from fastapi import APIRouter, File, UploadFile
from ..services.pdf_service import process_pdf
from ..services.pdf_service import process_questions
router = APIRouter()

@router.post("/process-pdf")
async def process_pdf_route(file: UploadFile = File(...)):
    """
    Process a PDF file and extract embeddings
    
    This endpoint accepts a PDF file, extracts text and creates embeddings
    using Sentence-BERT for later question answering.
    """
    paragraphs, message = await process_pdf(file)
    return {"status": "success", "paragraphs": paragraphs, "message": message}



@router.post("/process-questions")
async def process_questions_route(file: UploadFile = File(...)):
    """
    Process a PDF file of questions and extract embeddings
    
    This endpoint accepts a PDF file, extracts text and creates embeddings
    using Sentence-BERT for later question answering.
    """
    paragraphs, message = await process_questions(file)
    return {"status": "success", "paragraphs": paragraphs, "message": message}
