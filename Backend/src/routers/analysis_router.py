from fastapi import APIRouter, Depends, HTTPException, Request

from src.core.models import QuestionPaperAnalysisRequest, QuestionPaperAnalysisResponse
from src.core.security import get_current_user
from src.core.limiter import limiter
from src.services.question_paper_analysis_service import analyze_question_papers_service

router = APIRouter(prefix="/analysis", tags=["Question Paper Analysis"])

@router.post(
    "/question-papers",
    response_model=QuestionPaperAnalysisResponse,
    summary="Analyze Question Papers",
    description="Analyze question paper patterns using syllabus and previous year question papers with Gemini 2.0 Flash"
)
@limiter.limit("10/hour")
async def analyze_question_papers(
    request: Request,
    analysis_request: QuestionPaperAnalysisRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Analyze question papers to identify patterns, generate sample questions, and provide study recommendations.
    
    This endpoint uses Gemini 2.0 Flash to:
    - Analyze the syllabus structure
    - Identify patterns in previous year question papers
    - Calculate unit-wise weightage
    - Generate sample questions following the same patterns
    - Provide preparation strategy and focus areas
    
    **Required inputs:**
    - syllabus_pdf_id: ID of the uploaded syllabus PDF
    - question_paper_pdf_ids: List of IDs of uploaded question paper PDFs
    
    **Returns:**
    - Comprehensive analysis with unit-wise breakdown
    - Question patterns and difficulty analysis
    - Generated sample questions
    - Study recommendations and preparation strategy
    """
    try:
        # Validate input
        if not analysis_request.syllabus_pdf_id:
            raise HTTPException(
                status_code=400,
                detail="Syllabus PDF ID is required"
            )
        
        if not analysis_request.question_paper_pdf_ids or len(analysis_request.question_paper_pdf_ids) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one question paper PDF ID is required"
            )
        
        # Perform analysis
        analysis_result = await analyze_question_papers_service(
            syllabus_pdf_id=analysis_request.syllabus_pdf_id,
            question_paper_pdf_ids=analysis_request.question_paper_pdf_ids,
            user_id=user_id
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing question papers: {str(e)}"
        )
