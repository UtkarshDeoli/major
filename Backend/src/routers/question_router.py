from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from ..core.models import QuestionRequest, QuestionResponse
from ..services.llm_service import get_answer, stream_answer
from ..core.data_store import data_store

router = APIRouter()

async def verify_pdf_processed():
    """Dependency to verify PDF is processed before answering questions"""
    if not data_store.has_data():
        raise HTTPException(status_code=400, detail="No PDF has been processed. Please upload a PDF first.")
    return True

@router.post("/ask-question", response_model=QuestionResponse)
async def ask_question_route(request: QuestionRequest, _: bool = Depends(verify_pdf_processed)):
    """
    Answer a question based on processed PDF content
    
    This endpoint accepts a question and generates an answer using the LLM,
    providing relevant context from the processed PDF.
    """
    return await get_answer(request.question)

@router.post("/ask-question/stream")
async def ask_question_stream_route(request: QuestionRequest, _: bool = Depends(verify_pdf_processed)):
    """
    Stream an answer to a question based on processed PDF content
    
    This endpoint accepts a question and streams the generated answer from the LLM
    in real-time, providing relevant context from the processed PDF.
    """
    return StreamingResponse(
        stream_answer(request.question),
        media_type="application/x-ndjson"
    )
