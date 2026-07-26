from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from pydantic import BaseModel
from typing import Dict, List, Optional

from src.core.models import (
    MockTestGenerationRequest,
    MockTestResponse,
    MockTestQuestion,
    MockTestSubmission,
    MockTestAnalysisResponse,
    MockTestListResponse
)
from src.core.security import get_current_user, require_role
from src.core.plan_enforcement import enforce_limit
from src.core.data_store import get_mock_test as fetch_mock_test_data
from src.services.auth_service import get_user_by_email
from src.services.mock_test_service import (
    generate_mock_test_service,
    analyze_mock_test_submission_service,
    get_user_mock_tests_service,
    get_mock_test_service
)
from src.core.data_store import update_mock_test_assignment
from src.core.limiter import limiter

router = APIRouter(prefix="/mock-tests", tags=["Mock Tests"])

@router.post(
    "/generate",
    response_model=MockTestResponse,
    summary="Generate Mock Test",
    description="Generate a mock test using syllabus, previous year question papers, and optional study notes with Gemini AI"
)
@limiter.limit("10/hour")
async def generate_mock_test(
    request: Request,
    req: MockTestGenerationRequest,
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("mock_test")),
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
        if not req.syllabus_pdf_id:
            raise HTTPException(
                status_code=400,
                detail="Syllabus PDF ID is required"
            )

        if not req.question_paper_pdf_ids or len(req.question_paper_pdf_ids) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one question paper PDF ID is required"
            )

        # Determine ownership / assignment
        created_by = user_id
        assigned_to = None

        if req.student_email and req.student_email != user_id:
            student = await get_user_by_email(req.student_email)
            if not student:
                raise HTTPException(
                    status_code=404,
                    detail="Student not found"
                )
            if student.get("teacher_id") != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to assign test to this student"
                )
            assigned_to = req.student_email

        # Generate mock test
        mock_test = await generate_mock_test_service(
            syllabus_pdf_id=req.syllabus_pdf_id,
            question_paper_pdf_ids=req.question_paper_pdf_ids,
            notes_pdf_id=req.notes_pdf_id,
            num_mcq=req.num_mcq,
            num_text=req.num_text,
            total_marks=req.total_marks,
            difficulty_level=req.difficulty_level,
            focus_topics=req.focus_topics,
            weak_topics=req.weak_topics,
            subject=req.subject,
            user_id=user_id,
            created_by=created_by,
            assigned_to=assigned_to,
            grading_mode=req.grading_mode,
            source_material_ids=req.source_material_ids,
            adaptive=req.adaptive or req.difficulty_level == "adaptive",
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
        
        # Validate access to the mock test
        test_check = await get_mock_test_service(test_id, user_id)
        if not test_check:
            raise HTTPException(
                status_code=404,
                detail="Mock test not found"
            )

        # Determine who is allowed to submit
        submitter_email = user_id
        if test_check.assigned_to:
            if test_check.assigned_to != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only the assigned student can submit this test"
                )
            submitter_email = test_check.assigned_to

        # Load the full test data (with correct answers) for grading
        test_data = await fetch_mock_test_data(test_id)
        if not test_data:
            raise HTTPException(
                status_code=404,
                detail="Mock test not found"
            )

        questions = [MockTestQuestion(**q) for q in test_data["questions"]]
        test = MockTestResponse(
            test_id=test_data["test_id"],
            title=test_data["title"],
            questions=questions,
            total_marks=test_data["total_marks"],
            time_limit=test_data["time_limit"],
            created_at=test_data["created_at"],
            user_id=test_data["user_id"],
            difficulty_level=test_data.get("difficulty_level", "medium"),
            created_by=test_data.get("created_by"),
            assigned_to=test_data.get("assigned_to"),
            subject=test_data.get("subject"),
            grading_mode=test_data.get("grading_mode", "auto"),
            status=test_data.get("status"),
        )

        # Analyze the submission
        analysis = await analyze_mock_test_submission_service(
            test=test,
            submission=submission,
            user_id=user_id,
            submitter_email=submitter_email
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing mock test submission: {str(e)}"
        )

@router.post(
    "/{test_id}/assign",
    summary="Assign Existing Mock Test",
    description="Assign an existing mock test to a managed student"
)
async def assign_mock_test(
    test_id: str = Path(..., description="The ID of the mock test"),
    student_email: str = Query(..., description="The email of the student to assign"),
    teacher=Depends(require_role("teacher")),
):
    """
    Assign an already-created mock test to a managed student.
    """
    try:
        user_id = teacher["email"]
        test = await get_mock_test_service(test_id, user_id)
        if not test:
            raise HTTPException(status_code=404, detail="Mock test not found")

        is_owner = test.user_id == user_id or test.created_by == user_id
        if not is_owner:
            raise HTTPException(status_code=403, detail="Not authorized to assign this test")

        if test.assigned_to:
            raise HTTPException(status_code=400, detail="Test is already assigned to a student")

        student = await get_user_by_email(student_email)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        teacher_ids = student.get("teacher_ids") or ([student["teacher_id"]] if student.get("teacher_id") else [])
        if user_id not in teacher_ids:
            raise HTTPException(status_code=403, detail="Not authorized to assign test to this student")

        updated = await update_mock_test_assignment(test_id, student_email)
        if not updated:
            raise HTTPException(
                status_code=400, detail="Test is already assigned or could not be updated"
            )

        return {"test_id": test_id, "assigned_to": student_email}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assigning mock test: {str(e)}")


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
        from src.core.data_store import mock_test_submissions_collection, get_mock_test

        if mock_test_submissions_collection is None:
            raise HTTPException(
                status_code=503,
                detail="Database connection unavailable"
            )

        # Find the submission by ID only, then authorize
        submission = await mock_test_submissions_collection.find_one({
            "submission_id": submission_id
        })

        if not submission:
            raise HTTPException(
                status_code=404,
                detail="Mock test submission not found"
            )

        test_data = await get_mock_test(submission["test_id"])
        is_owner = submission.get("user_id") == user_id
        is_creator = test_data is not None and (
            test_data.get("created_by") == user_id or
            test_data.get("user_id") == user_id
        )
        if not is_owner and not is_creator:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this submission"
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


class SubmissionSummary(BaseModel):
    submission_id: str
    test_id: str
    user_id: str
    total_score: float
    max_score: float
    percentage: float
    time_taken: int
    status: Optional[str] = None
    grading_mode: Optional[str] = None
    created_at: Optional[str] = None


class SubmissionListResponse(BaseModel):
    submissions: List[SubmissionSummary]


@router.get(
    "/{test_id}/submissions",
    response_model=SubmissionListResponse,
    summary="List attempts for a mock test",
)
async def list_test_submissions(
    test_id: str = Path(...),
    user_id: str = Depends(get_current_user),
):
    """List all attempts for a test. Students see their own; teachers see the
    assigned student's attempts (so both can track attempts)."""
    from src.core.data_store import mock_test_submissions_collection

    if mock_test_submissions_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    test = await get_mock_test_service(test_id, user_id)
    if not test:
        raise HTTPException(status_code=404, detail="Mock test not found")

    # Whose submissions to show: assigned student's if teacher/owner viewing an assigned test, else own
    target_user = test.assigned_to if test.assigned_to and test.assigned_to != user_id else user_id
    # A student may only view their own submissions
    if test.assigned_to and user_id == test.assigned_to:
        target_user = user_id
    # If the caller is neither the assigned student nor the creator/owner, deny
    allowed = {test.user_id, test.created_by, test.assigned_to}
    if user_id not in allowed:
        raise HTTPException(status_code=403, detail="Not authorized to view these submissions")

    cursor = mock_test_submissions_collection.find({"test_id": test_id, "user_id": target_user}).sort("created_at", -1)
    subs = await cursor.to_list(length=None)
    out = []
    for s in subs:
        ca = s.get("created_at")
        out.append(SubmissionSummary(
            submission_id=s.get("submission_id"), test_id=s.get("test_id"),
            user_id=s.get("user_id"), total_score=s.get("total_score", 0),
            max_score=s.get("max_score", 0), percentage=s.get("percentage", 0),
            time_taken=s.get("time_taken", 0), status=s.get("status"),
            grading_mode=s.get("grading_mode"),
            created_at=ca.isoformat() if hasattr(ca, "isoformat") else (str(ca) if ca else None),
        ))
    return SubmissionListResponse(submissions=out)


class GradeItem(BaseModel):
    question_id: str
    marks_awarded: float
    feedback: Optional[str] = None


class GradeRequest(BaseModel):
    grades: List[GradeItem]


@router.post(
    "/submissions/{submission_id}/grade",
    response_model=MockTestAnalysisResponse,
    summary="Teacher grades a pending-review submission",
)
async def grade_submission(
    submission_id: str = Path(...),
    request: GradeRequest = ...,
    teacher=Depends(require_role("teacher")),
):
    """Teacher grades the text answers of a pending-review (teacher-marked) test."""
    from src.core.data_store import mock_test_submissions_collection, get_mock_test

    if mock_test_submissions_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    teacher_email = teacher["email"]

    submission = await mock_test_submissions_collection.find_one({"submission_id": submission_id})
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    test_data = await get_mock_test(submission["test_id"])
    if not test_data:
        raise HTTPException(status_code=404, detail="Test not found")
    if test_data.get("created_by") != teacher_email and test_data.get("user_id") != teacher_email:
        raise HTTPException(status_code=403, detail="Not authorized to grade this submission")

    grade_map = {g.question_id: g for g in request.grades}
    qf = submission.get("question_feedback", []) or []
    total_score = 0.0
    for item in qf:
        gid = item.get("question_id")
        if gid in grade_map:
            g = grade_map[gid]
            item["marks_awarded"] = g.marks_awarded
            if g.feedback is not None:
                item["feedback"] = g.feedback
        total_score += float(item.get("marks_awarded", 0) or 0)

    max_score = float(submission.get("max_score", 0) or 0)
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    await mock_test_submissions_collection.update_one(
        {"submission_id": submission_id},
        {"$set": {
            "question_feedback": qf,
            "total_score": total_score,
            "percentage": percentage,
            "status": "graded",
            "graded_by": teacher_email,
            "graded_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        }},
    )

    return MockTestAnalysisResponse(
        submission_id=submission["submission_id"],
        test_id=submission["test_id"],
        total_score=total_score,
        max_score=max_score,
        percentage=percentage,
        time_taken=submission.get("time_taken", 0),
        feedback_summary=submission.get("feedback_summary", "Graded by teacher."),
        question_feedback=qf,
        strengths=submission.get("strengths", []),
        improvements=submission.get("improvements", []),
        study_recommendations=submission.get("study_recommendations", []),
        created_at=submission.get("created_at"),
    )
