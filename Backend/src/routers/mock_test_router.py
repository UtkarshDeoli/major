from fastapi import APIRouter, Depends, HTTPException, Path

from src.core.data_store import users_collection
from src.core.models import (
    MockTestGenerationRequest,
    MockTestResponse,
    MockTestSubmission,
    MockTestAnalysisResponse,
    MockTestListResponse
)
from src.core.security import get_current_user
from src.services.mock_test_service import (
    generate_mock_test_service,
    analyze_mock_test_submission_service,
    get_user_mock_tests_service,
    get_mock_test_service
)

router = APIRouter(prefix="/mock-tests", tags=["Mock Tests"])

@router.post(
    "/generate",
    response_model=MockTestResponse,
    summary="Generate Mock Test",
    description="Generate a mock test using syllabus, previous year question papers, and optional study notes with Gemini AI"
)
async def generate_mock_test(
    request: MockTestGenerationRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Generate a personalized mock test based on syllabus and previous year papers.
    
    This endpoint uses Gemini AI to:
    - Analyze the syllabus and question patterns
    - Generate MCQ and descriptive questions
    - Create a balanced test with proper difficulty distribution
    - Set appropriate marks and time limits
    
    **Required inputs:**
    - syllabus_pdf_id: ID of the uploaded syllabus PDF
    - question_paper_pdf_ids: List of IDs of uploaded question paper PDFs
    - notes_pdf_id: Optional ID of study notes PDF
    - num_mcq: Number of MCQ questions (default: 15)
    - num_text: Number of descriptive questions (default: 5)
    - total_marks: Total marks for the test (default: 50)
    - difficulty_level: Test difficulty level (default: medium)
    
    **Returns:**
    - Generated mock test with questions, marks, and time limit
    """
    try:
        # Validate input
        if not request.syllabus_pdf_id:
            raise HTTPException(
                status_code=400,
                detail="Syllabus PDF ID is required"
            )
        
        if not request.question_paper_pdf_ids or len(request.question_paper_pdf_ids) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one question paper PDF ID is required"
            )
        
        # Teachers can generate tests targeting a specific student under them
        if request.student_email and user_id != request.student_email:
            if users_collection is None:
                raise HTTPException(status_code=503, detail="Database unavailable")
            student = await users_collection.find_one({
                "email": request.student_email,
                "teacher_id": user_id,
            })
            if not student:
                raise HTTPException(status_code=403, detail="You can only target students you manage")
        
        # Generate mock test
        mock_test = await generate_mock_test_service(
            syllabus_pdf_id=request.syllabus_pdf_id,
            question_paper_pdf_ids=request.question_paper_pdf_ids,
            notes_pdf_id=request.notes_pdf_id,
            num_mcq=request.num_mcq,
            num_text=request.num_text,
            total_marks=request.total_marks,
            difficulty_level=request.difficulty_level,
            user_id=user_id,
            focus_topics=request.focus_topics,
            weak_topics=request.weak_topics,
            subject=request.subject,
            student_email=request.student_email,
        )
        
        return mock_test
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating mock test: {str(e)}"
        )

@router.get(
    "/",
    response_model=MockTestListResponse,
    summary="List Mock Tests",
    description="Get all mock tests created by the current user"
)
async def list_mock_tests(
    user_id: str = Depends(get_current_user)
):
    """
    List all mock tests created by the current user.
    """
    try:
        tests = await get_user_mock_tests_service(user_id)
        return MockTestListResponse(tests=tests)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching mock tests: {str(e)}"
        )

@router.get(
    "/{test_id}",
    response_model=MockTestResponse,
    summary="Get Mock Test",
    description="Get a specific mock test by ID"
)
async def get_mock_test(
    test_id: str = Path(..., description="The ID of the mock test"),
    user_id: str = Depends(get_current_user)
):
    """
    Get a specific mock test by ID.
    """
    try:
        test = await get_mock_test_service(test_id, user_id)
        if not test:
            raise HTTPException(
                status_code=404,
                detail="Mock test not found"
            )
        return test
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching mock test: {str(e)}"
        )

@router.post(
    "/{test_id}/submit",
    response_model=MockTestAnalysisResponse,
    summary="Submit Mock Test",
    description="Submit a mock test and get detailed analysis with feedback using Gemini AI"
)
async def submit_mock_test(
    submission: MockTestSubmission,
    test_id: str = Path(..., description="The ID of the mock test"),
    user_id: str = Depends(get_current_user)
):
    """
    Submit a mock test and get detailed AI-powered analysis and feedback.
    
    This endpoint:
    - Evaluates MCQ answers automatically
    - Uses Gemini AI to analyze descriptive answers
    - Provides detailed feedback for each question
    - Generates overall performance analysis
    - Suggests study improvements and recommendations
    
    **Returns:**
    - Comprehensive analysis with scores, feedback, and recommendations
    """
    try:
        # Validate submission
        if submission.test_id != test_id:
            raise HTTPException(
                status_code=400,
                detail="Test ID mismatch in submission"
            )
        
        # Get the mock test to validate it belongs to the user
        test = await get_mock_test_service(test_id, user_id)
        if not test:
            raise HTTPException(
                status_code=404,
                detail="Mock test not found"
            )
        
        # Analyze the submission
        analysis = await analyze_mock_test_submission_service(
            test=test,
            submission=submission,
            user_id=user_id
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing mock test submission: {str(e)}"
        )

@router.get(
    "/submissions/{submission_id}/analysis",
    response_model=MockTestAnalysisResponse,
    summary="Get Mock Test Analysis",
    description="Get the analysis results for a specific mock test submission"
)
async def get_mock_test_analysis(
    submission_id: str = Path(..., description="The ID of the mock test submission"),
    user_id: str = Depends(get_current_user)
):
    """
    Get the analysis results for a specific mock test submission.
    """
    try:
        # Get the submission analysis from database
        from src.core.data_store import mock_test_submissions_collection
        
        if mock_test_submissions_collection is None:
            raise HTTPException(
                status_code=503,
                detail="Database connection unavailable"
            )
        
        # Find the submission
        submission = await mock_test_submissions_collection.find_one({
            "submission_id": submission_id,
            "user_id": user_id
        })
        
        if not submission:
            raise HTTPException(
                status_code=404,
                detail="Mock test submission not found"
            )
        
        # Convert MongoDB document to response model
        analysis_data = {
            "submission_id": submission["submission_id"],
            "test_id": submission["test_id"],
            "total_score": submission["total_score"],
            "max_score": submission["max_score"],
            "percentage": submission["percentage"],
            "time_taken": submission["time_taken"],
            "feedback_summary": submission["feedback_summary"],
            "question_feedback": submission["question_feedback"],
            "strengths": submission["strengths"],
            "improvements": submission["improvements"],
            "study_recommendations": submission["study_recommendations"],
            "created_at": submission["created_at"]
        }
        
        return MockTestAnalysisResponse(**analysis_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching mock test analysis: {str(e)}"
        )
