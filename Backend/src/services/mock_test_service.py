import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import HTTPException

from src.services.gemini_service import gemini_service
from src.services.student_mastery_service import build_adaptive_bias, update_mastery_from_submission
from src.core.data_store import (
    get_pdf_metadata,
    store_mock_test,
    get_user_mock_tests,
    get_mock_test,
    store_mock_test_submission
)
from src.core.models import (
    MockTestResponse,
    MockTestQuestion,
    MockTestSubmission,
    MockTestAnalysisResponse,
    AnswerFeedback
)

async def generate_mock_test_service(
    syllabus_pdf_id: str,
    question_paper_pdf_ids: List[str],
    notes_pdf_id: Optional[str],
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    user_id: str,
    created_by: Optional[str] = None,
    assigned_to: Optional[str] = None,
    focus_topics: Optional[List[str]] = None,
    weak_topics: Optional[List[str]] = None,
    subject: Optional[str] = None,
    grading_mode: str = "auto",
    source_material_ids: Optional[List[str]] = None,
    adaptive: bool = False,
) -> MockTestResponse:
    """Generate a mock test using syllabus + previous year papers."""

    if not gemini_service:
        raise HTTPException(
            status_code=503,
            detail="Gemini service is not available"
        )
    
    try:
        # Get PDF contents
        syllabus_metadata = await get_pdf_metadata(syllabus_pdf_id)
        if not syllabus_metadata or syllabus_metadata["user_id"] != user_id:
            raise HTTPException(
                status_code=404,
                detail="Syllabus PDF not found or access denied"
            )
        
        # Extract text from syllabus
        syllabus_content = await gemini_service.extract_text_from_pdf(syllabus_metadata["file_path"])
        
        # Extract text from question papers
        question_papers_content = []
        for qp_id in question_paper_pdf_ids:
            qp_metadata = await get_pdf_metadata(qp_id)
            if not qp_metadata or qp_metadata["user_id"] != user_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Question paper PDF {qp_id} not found or access denied"
                )
            qp_content = await gemini_service.extract_text_from_pdf(qp_metadata["file_path"])
            question_papers_content.append(qp_content)
        
        # Extract text from notes if provided
        notes_content = ""
        if notes_pdf_id:
            notes_metadata = await get_pdf_metadata(notes_pdf_id)
            if notes_metadata and notes_metadata["user_id"] == user_id:
                notes_content = await gemini_service.extract_text_from_pdf(notes_metadata["file_path"])

        # Adaptive difficulty: bias toward weak topics and appropriate difficulty
        effective_difficulty = difficulty_level
        if adaptive or difficulty_level == "adaptive":
            bias = await build_adaptive_bias(user_id, "adaptive")
            effective_difficulty = bias["difficulty"]
            focus_topics = list(set((focus_topics or []) + bias["focus_topics"]))
            weak_topics = list(set((weak_topics or []) + bias["weak_topics"]))

        # Generate mock test using Gemini
        mock_test_data = await _generate_mock_test_with_gemini(
            syllabus_content=syllabus_content,
            question_papers_content=question_papers_content,
            notes_content=notes_content,
            num_mcq=num_mcq,
            num_text=num_text,
            total_marks=total_marks,
            difficulty_level=effective_difficulty,
            focus_topics=focus_topics,
            weak_topics=weak_topics,
            subject=subject
        )
        
        # Create mock test object
        test_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        mock_test = MockTestResponse(
            test_id=test_id,
            title=f"Mock Test - {created_at.strftime('%B %d, %Y')}",
            questions=mock_test_data["questions"],
            total_marks=total_marks,
            time_limit=_calculate_time_limit(total_marks, num_mcq, num_text),
            created_at=created_at,
            user_id=user_id,
            difficulty_level=effective_difficulty,
            created_by=created_by,
            assigned_to=assigned_to,
            subject=subject,
            grading_mode=grading_mode,
            status="pending_review" if grading_mode == "teacher" else None
        )

        # Store in database
        await store_mock_test(mock_test.model_dump())

        return mock_test

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating mock test: {str(e)}"
        )


async def generate_mock_test_from_docs_service(
    doc_ids: List[str],
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    user_id: str,
    subject: Optional[str] = None,
    grading_mode: str = "auto",
) -> MockTestResponse:
    """Generate a practice mock test directly from uploaded document(s)."""

    if not gemini_service:
        raise HTTPException(
            status_code=503,
            detail="Gemini service is not available"
        )

    try:
        # Validate ownership and collect content
        parts: List[str] = []
        for doc_id in doc_ids:
            pdf = await get_pdf_metadata(doc_id)
            if not pdf or pdf.get("user_id") != user_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {doc_id} not found or access denied"
                )
            file_path = pdf.get("file_path")
            if not file_path:
                continue
            text = await gemini_service.extract_text_from_pdf(file_path)
            if text:
                parts.append(text)

        if not parts:
            raise HTTPException(
                status_code=400,
                detail="No readable content found in the selected documents"
            )

        combined_content = "\n\n---DOCUMENT---\n\n".join(parts)[:12000]

        mock_test_data = await _generate_mock_test_from_content_with_gemini(
            content=combined_content,
            num_mcq=num_mcq,
            num_text=num_text,
            total_marks=total_marks,
            difficulty_level=difficulty_level,
            subject=subject,
        )

        test_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        mock_test = MockTestResponse(
            test_id=test_id,
            title=f"Practice Test - {subject or 'General'} - {created_at.strftime('%B %d, %Y')}",
            questions=mock_test_data["questions"],
            total_marks=total_marks,
            time_limit=_calculate_time_limit(total_marks, num_mcq, num_text),
            created_at=created_at,
            user_id=user_id,
            difficulty_level=difficulty_level,
            subject=subject,
            grading_mode=grading_mode,  # type: ignore[arg-type]
        )

        await store_mock_test(mock_test.model_dump())
        return mock_test

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating practice test: {str(e)}"
        )


async def _generate_mock_test_from_content_with_gemini(
    content: str,
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """Use Gemini to generate questions from provided study content."""

    mcq_marks_per_question = 2
    num_text_safe = max(0, num_text)
    text_marks_per_question = (
        max(5, (total_marks - num_mcq * mcq_marks_per_question) // num_text_safe)
        if num_text_safe > 0 else 5
    )

    prompt = f"""You are an expert exam paper setter. Create a realistic practice test based ONLY on the study material provided below.

INSTRUCTIONS:
1. Read the study material carefully.
2. Generate questions that are directly based on the material.
3. Cover important concepts, facts, and examples from the material.
4. Maintain the requested difficulty level and balance across topics.

STUDY MATERIAL:
{content}

SUBJECT: {subject or "Not specified"}
DIFFICULTY: {difficulty_level}

REQUIREMENTS:
- Generate {num_mcq} Multiple Choice Questions (MCQ) worth {mcq_marks_per_question} marks each
- Generate {num_text} Descriptive/Text questions worth {text_marks_per_question} marks each
- Total marks: {total_marks}
- MCQ should have 4 options with one correct answer
- Options must start with A), B), C), D)

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "questions": [
        {{
            "id": "1",
            "type": "mcq",
            "question": "Which of the following correctly describes...",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correctAnswer": "A) Option 1",
            "marks": {mcq_marks_per_question},
            "unit": "Unit 1",
            "topic": "Specific topic",
            "difficulty": "Easy/Medium/Hard"
        }},
        {{
            "id": "{num_mcq + 1}",
            "type": "text",
            "question": "Explain the concept of... with suitable examples.",
            "marks": {text_marks_per_question},
            "unit": "Unit 2",
            "topic": "Specific topic",
            "difficulty": "Medium/Hard"
        }}
    ]
}}

CRITICAL REQUIREMENTS:
- Generate exactly {num_mcq + num_text} questions total
- Every question MUST be answerable from the provided study material
- NO generic questions unrelated to the material
- MCQ options must start with A), B), C), D)
- Ensure sequential question IDs as strings
- Return ONLY the JSON object, no additional text"""

    try:
        response = gemini_service.model.generate_content(prompt)

        if not response or not response.text:
            return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)

        response_text = response.text.strip()
        print(f"Practice test generation response: {response_text[:500]}...")

        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)

        json_text = response_text[start_idx:end_idx]
        mock_test_data = json.loads(json_text)

        questions = []
        for i, q_data in enumerate(mock_test_data.get("questions", [])):
            question = MockTestQuestion(
                id=str(i + 1),
                type=q_data.get("type", "mcq"),
                question=q_data.get("question", f"Sample question {i + 1}"),
                options=q_data.get("options") if q_data.get("type") == "mcq" else None,
                correctAnswer=q_data.get("correctAnswer") if q_data.get("type") == "mcq" else None,
                marks=q_data.get("marks", mcq_marks_per_question if q_data.get("type") == "mcq" else text_marks_per_question),
                unit=q_data.get("unit"),
                topic=q_data.get("topic"),
                difficulty=q_data.get("difficulty", difficulty_level),
            )
            questions.append(question)

        return {"questions": questions}

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in practice test generation: {str(e)}")
        return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)
    except Exception as e:
        print(f"Error in practice test generation: {str(e)}")
        return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)


async def _generate_mock_test_with_gemini(
    syllabus_content: str,
    question_papers_content: List[str],
    notes_content: str,
    num_mcq: int,
    num_text: int,
    total_marks: int,
    difficulty_level: str,
    focus_topics: Optional[List[str]] = None,
    weak_topics: Optional[List[str]] = None,
    subject: Optional[str] = None
) -> Dict[str, Any]:
    """Use Gemini to generate mock test questions with enhanced pattern matching"""
    
    # Combine all question papers content
    combined_question_papers = "\n\n---PREVIOUS PAPER---\n\n".join(question_papers_content)
    
    # Calculate marks distribution
    mcq_marks_per_question = 2
    text_marks_per_question = max(5, (total_marks - num_mcq * mcq_marks_per_question) // num_text) if num_text > 0 else 5
    
    prompt = f"""You are an expert exam paper setter with years of experience. Create a realistic mock test paper.

INSTRUCTIONS:
1. CAREFULLY analyze the syllabus to understand course structure and learning outcomes
2. STUDY the previous year question papers to identify:
   - Question patterns and formats
   - Marks distribution and question types
   - Frequently asked topics
   - Question difficulty progression
3. REVIEW the notes to understand depth of coverage for each topic
4. GENERATE questions that:
   - Are directly relevant to syllabus topics
   - Follow the same pattern as previous papers
   - Test conceptual understanding, application, and problem-solving
   - Have realistic difficulty levels
   - Cover different units proportionally

SYLLABUS (PRIMARY REFERENCE):
{syllabus_content[:3000]}

STUDY NOTES (For topic depth):
{notes_content[:2000] if notes_content else "No additional notes provided"}

PREVIOUS YEAR QUESTION PAPERS (For pattern matching):
{combined_question_papers[:4000]}

SUBJECT:
{subject or "Not specified"}

FOCUS TOPICS (emphasize these in the test when provided):
{", ".join(focus_topics) if focus_topics else "None provided"}

WEAK TOPICS (include more questions targeting these areas when provided):
{", ".join(weak_topics) if weak_topics else "None provided"}

REQUIREMENTS:
- Generate {num_mcq} Multiple Choice Questions (MCQ) worth {mcq_marks_per_question} marks each
- Generate {num_text} Descriptive/Text questions worth {text_marks_per_question} marks each
- Difficulty level: {difficulty_level}
- Subject: {subject or "the syllabus subject"}
- ALL questions must be traceable to syllabus topics
- Prioritize focus topics and weak topics when they are provided
- MCQ should have 4 options with one correct answer
- Follow question patterns from previous papers

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "questions": [
        {{
            "id": "1",
            "type": "mcq",
            "question": "Which of the following correctly describes...",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correctAnswer": "A) Option 1",
            "marks": {mcq_marks_per_question},
            "unit": "Unit 1",
            "topic": "Specific topic from syllabus",
            "difficulty": "Easy/Medium/Hard",
            "syllabus_reference": "Exact topic from syllabus"
        }},
        {{
            "id": "{num_mcq + 1}",
            "type": "text",
            "question": "Explain the concept of... with suitable examples.",
            "marks": {text_marks_per_question},
            "unit": "Unit 2",
            "topic": "Specific topic from syllabus",
            "difficulty": "Medium/Hard",
            "syllabus_reference": "Exact topic from syllabus",
            "answer_guidelines": ["Point 1", "Point 2", "Point 3"]
        }}
    ]
}}

CRITICAL REQUIREMENTS:
- Generate exactly {num_mcq + num_text} questions total
- Every question MUST reference a specific syllabus topic
- Question patterns MUST match previous year papers
- NO generic questions like "Was this in syllabus?" - all must be subject-specific
- MCQ options must start with A), B), C), D)
- Ensure sequential question IDs as strings
- Return ONLY the JSON object, no additional text"""

    try:
        response = gemini_service.model.generate_content(prompt)
        
        if not response or not response.text:
            return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)
        
        response_text = response.text.strip()
        print(f"Enhanced mock test generation response: {response_text[:500]}...")
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)
        
        json_text = response_text[start_idx:end_idx]
        mock_test_data = json.loads(json_text)
        
        # Validate and convert to MockTestQuestion objects
        questions = []
        for i, q_data in enumerate(mock_test_data.get("questions", [])):
            question = MockTestQuestion(
                id=str(i + 1),
                type=q_data.get("type", "mcq"),
                question=q_data.get("question", f"Sample question {i + 1}"),
                options=q_data.get("options") if q_data.get("type") == "mcq" else None,
                correctAnswer=q_data.get("correctAnswer") if q_data.get("type") == "mcq" else None,
                marks=q_data.get("marks", mcq_marks_per_question if q_data.get("type") == "mcq" else text_marks_per_question),
                unit=q_data.get("unit"),
                topic=q_data.get("topic"),
                syllabus_reference=q_data.get("syllabus_reference"),
                difficulty=q_data.get("difficulty", difficulty_level),
            )
            questions.append(question)

        return {"questions": questions}
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in mock test generation: {str(e)}")
        return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)
    except Exception as e:
        print(f"Error in mock test generation: {str(e)}")
        return _create_fallback_mock_test(num_mcq, num_text, mcq_marks_per_question, text_marks_per_question)

def _create_fallback_mock_test(num_mcq: int, num_text: int, mcq_marks: int, text_marks: int) -> Dict[str, Any]:
    """Create a fallback mock test when Gemini fails"""
    questions = []
    
    # Generate MCQ questions
    for i in range(num_mcq):
        questions.append(MockTestQuestion(
            id=str(i + 1),
            type="mcq",
            question=f"Sample multiple choice question {i + 1} - Choose the correct answer:",
            options=["Option A", "Option B", "Option C", "Option D"],
            correctAnswer="Option A",
            marks=mcq_marks
        ))
    
    # Generate text questions
    for i in range(num_text):
        questions.append(MockTestQuestion(
            id=str(num_mcq + i + 1),
            type="text",
            question=f"Sample descriptive question {i + 1} - Explain the concept with examples.",
            marks=text_marks
        ))
    
    return {"questions": questions}

def _normalize_mcq_answer(answer: Optional[str]) -> str:
    """Normalize an MCQ answer for robust comparison.

    Gemini often returns options like 'A) Option 1' while the client may store
    just 'A) Option 1' or the bare letter 'A'. Compare on the leading letter
    when present, else fall back to a trimmed full-string compare.
    """
    if not answer:
        return ""
    a = answer.strip()
    # Match an optional letter/number label at the start: "A)", "A.", "A)", "1)", "a."
    if len(a) >= 2 and a[0].isalnum() and a[1] in ").-:":
        return a[0].lower()
    return a.lower()


def _calculate_time_limit(total_marks: int, num_mcq: int, num_text: int) -> int:
    """Calculate appropriate time limit in minutes"""
    # 1 minute per MCQ mark + 2 minutes per text question mark
    mcq_time = num_mcq * 2  # 2 minutes per MCQ
    text_time = num_text * 10  # 10 minutes per descriptive question
    total_time = mcq_time + text_time
    return max(60, total_time)  # Minimum 1 hour

async def get_user_mock_tests_service(user_id: str) -> List[MockTestResponse]:
    """Get all mock tests for a user with latest submission info"""
    try:
        from src.core.data_store import mock_test_submissions_collection
        
        tests_data = await get_user_mock_tests(user_id)
        tests = []
        
        for test_data in tests_data:
            questions = [MockTestQuestion(**q) for q in test_data["questions"]]

            # Hide correct answers from the assigned student
            if test_data.get("assigned_to") == user_id:
                questions = [
                    MockTestQuestion(
                        id=q.id,
                        type=q.type,
                        question=q.question,
                        options=q.options,
                        correctAnswer=None,
                        marks=q.marks,
                        unit=q.unit,
                        topic=q.topic,
                        syllabus_reference=q.syllabus_reference,
                    )
                    for q in questions
                ]

            # Get the latest submission for this test
            latest_submission = None
            if mock_test_submissions_collection is not None:
                try:
                    # Teachers viewing an assigned test should see the assigned student's submission
                    if test_data.get("assigned_to") and (
                        test_data.get("created_by") == user_id or
                        test_data.get("user_id") == user_id
                    ):
                        submission_query = {
                            "test_id": test_data["test_id"],
                            "user_id": test_data["assigned_to"]
                        }
                    else:
                        submission_query = {
                            "test_id": test_data["test_id"],
                            "user_id": user_id
                        }

                    submission = await mock_test_submissions_collection.find_one(
                        submission_query,
                        sort=[("created_at", -1)]
                    )
                    if submission:
                        latest_submission = {
                            "submission_id": submission["submission_id"],
                            "submitted_at": submission["created_at"].isoformat() if isinstance(submission["created_at"], datetime) else submission["created_at"],
                            "score": submission["total_score"],
                            "percentage": submission["percentage"]
                        }
                except Exception as e:
                    print(f"Error fetching submission for test {test_data['test_id']}: {e}")

            # Create the test response with optional submission info
            test_dict = {
                "test_id": test_data["test_id"],
                "title": test_data["title"],
                "questions": questions,
                "total_marks": test_data["total_marks"],
                "time_limit": test_data["time_limit"],
                "created_at": test_data["created_at"],
                "user_id": test_data["user_id"],
                "difficulty_level": test_data.get("difficulty_level", "medium"),
                "created_by": test_data.get("created_by"),
                "assigned_to": test_data.get("assigned_to"),
                "subject": test_data.get("subject"),
                "grading_mode": test_data.get("grading_mode", "auto"),
                "status": test_data.get("status"),
            }
            
            if latest_submission:
                test_dict["latest_submission"] = latest_submission
            
            test = MockTestResponse(**test_dict)
            tests.append(test)
            
        return tests
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching user mock tests: {str(e)}"
        )

async def get_mock_test_service(test_id: str, user_id: str) -> Optional[MockTestResponse]:
    """Get a specific mock test"""
    try:
        test_data = await get_mock_test(test_id)
        if not test_data:
            return None
        allowed = {test_data.get("user_id"), test_data.get("assigned_to"), test_data.get("created_by")}
        if user_id not in allowed:
            return None

        questions = [MockTestQuestion(**q) for q in test_data["questions"]]

        # Hide correct answers from the assigned student
        if test_data.get("assigned_to") == user_id:
            questions = [
                MockTestQuestion(
                    id=q.id,
                    type=q.type,
                    question=q.question,
                    options=q.options,
                    correctAnswer=None,
                    marks=q.marks,
                    unit=q.unit,
                    topic=q.topic,
                    syllabus_reference=q.syllabus_reference,
                )
                for q in questions
            ]

        return MockTestResponse(
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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching mock test: {str(e)}"
        )

async def analyze_mock_test_submission_service(
    test: MockTestResponse,
    submission: MockTestSubmission,
    user_id: str,
    submitter_email: Optional[str] = None
) -> MockTestAnalysisResponse:
    """Analyze a mock test submission using Gemini AI"""
    
    if not gemini_service:
        raise HTTPException(
            status_code=503,
            detail="Gemini service is not available"
        )
    
    try:
        # Calculate scores and generate feedback
        total_score = 0.0
        max_score = test.total_marks
        question_feedback = []
        grading_mode = getattr(test, "grading_mode", "auto") or "auto"
        teacher_graded_pending = grading_mode == "teacher"

        for question in test.questions:
            user_answer = submission.answers.get(question.id, "")
            feedback = None

            if question.type == "mcq":
                # Automatic grading for MCQ (robust to "A) ..." vs "A" differences)
                is_correct = _normalize_mcq_answer(user_answer) == _normalize_mcq_answer(question.correctAnswer)
                marks_awarded = question.marks if is_correct else 0
                total_score += marks_awarded

                feedback = AnswerFeedback(
                    question_id=question.id,
                    question=question.question,
                    user_answer=user_answer,
                    correct_answer=question.correctAnswer,
                    is_correct=is_correct,
                    feedback="Correct answer!" if is_correct else f"Incorrect. The correct answer is {question.correctAnswer}.",
                    marks_awarded=marks_awarded,
                    max_marks=question.marks
                )
            elif teacher_graded_pending:
                # Teacher-marked test: do not auto-grade text answers; leave for teacher review
                feedback = AnswerFeedback(
                    question_id=question.id,
                    question=question.question,
                    user_answer=user_answer,
                    feedback="Pending teacher review.",
                    marks_awarded=0,
                    max_marks=question.marks
                )
            else:
                # AI-powered grading for text questions
                ai_feedback = await _analyze_text_answer_with_gemini(
                    question.question,
                    user_answer,
                    question.marks
                )
                total_score += ai_feedback["marks_awarded"]

                feedback = AnswerFeedback(
                    question_id=question.id,
                    question=question.question,
                    user_answer=user_answer,
                    feedback=ai_feedback["feedback"],
                    marks_awarded=ai_feedback["marks_awarded"],
                    max_marks=question.marks,
                    rubric_scores=ai_feedback.get("rubric_scores"),
                    rubric_max=ai_feedback.get("rubric_max"),
                )

            question_feedback.append(feedback)

        # Generate overall analysis using Gemini (skip heavy LLM analysis for pending-review tests)
        if teacher_graded_pending:
            overall_analysis = {
                "feedback_summary": "Submitted. Awaiting teacher grading for descriptive answers.",
                "strengths": [],
                "improvements": [],
                "study_recommendations": [],
            }
        else:
            overall_analysis = await _generate_overall_analysis_with_gemini(
                test, submission, question_feedback, total_score, max_score
            )

        # Create submission record
        submission_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        analysis = MockTestAnalysisResponse(
            submission_id=submission_id,
            test_id=test.test_id,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            time_taken=submission.time_taken,
            feedback_summary=overall_analysis["feedback_summary"],
            question_feedback=question_feedback,
            strengths=overall_analysis["strengths"],
            improvements=overall_analysis["improvements"],
            study_recommendations=overall_analysis["study_recommendations"],
            created_at=created_at
        )

        # Store submission in database, recording the actual submitter + subject + grading status
        submission_record = analysis.model_dump()
        submission_record["user_id"] = submitter_email if submitter_email else user_id
        submission_record["subject"] = getattr(test, "subject", None) or submission.subject
        submission_record["grading_mode"] = grading_mode
        submission_record["status"] = "pending_review" if teacher_graded_pending else "graded"

        # Enrich question feedback with topic/difficulty for mastery update
        question_lookup = {q.id: q for q in test.questions}
        enriched_feedback = []
        for fb in submission_record.get("question_feedback", []):
            q = question_lookup.get(fb.get("question_id"))
            if q:
                fb_copy = dict(fb)
                fb_copy["topic"] = q.topic
                fb_copy["difficulty"] = q.difficulty or test.difficulty_level
                enriched_feedback.append(fb_copy)
            else:
                enriched_feedback.append(fb)

        await store_mock_test_submission(submission_record)
        await update_mastery_from_submission(submission_record["user_id"], enriched_feedback)

        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing submission: {str(e)}"
        )

async def _analyze_text_answer_with_gemini(question: str, user_answer: str, max_marks: int) -> Dict[str, Any]:
    """Use Gemini to analyze and grade a text answer with a rubric."""

    if not user_answer.strip():
        return {
            "marks_awarded": 0,
            "feedback": "No answer provided.",
            "rubric_scores": {"understanding": 0, "accuracy": 0, "completeness": 0, "examples": 0, "clarity": 0},
            "rubric_max": {"understanding": 2, "accuracy": 2, "completeness": 2, "examples": 2, "clarity": 2},
        }

    prompt = f"""You are an expert examiner. Evaluate this student's answer using a rubric.

QUESTION: {question}

STUDENT'S ANSWER: {user_answer}

MAXIMUM MARKS: {max_marks}

Evaluate the answer across these criteria (0-2 each):
- understanding: conceptual grasp
- accuracy: factual correctness
- completeness: coverage of required points
- examples: use of relevant examples
- clarity: clear expression

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "marks_awarded": 3.5,
    "feedback": "Concise overall feedback.",
    "rubric_scores": {{
        "understanding": 2,
        "accuracy": 1,
        "completeness": 2,
        "examples": 1,
        "clarity": 2
    }},
    "rubric_max": {{
        "understanding": 2,
        "accuracy": 2,
        "completeness": 2,
        "examples": 2,
        "clarity": 2
    }}
}}

IMPORTANT: Return ONLY the JSON object, no additional text."""

    default_rubric_max = {"understanding": 2, "accuracy": 2, "completeness": 2, "examples": 2, "clarity": 2}
    fallback = {
        "marks_awarded": max_marks * 0.5,
        "feedback": "Unable to analyze answer automatically. Please review with instructor.",
        "rubric_scores": {k: 1 for k in default_rubric_max},
        "rubric_max": default_rubric_max,
    }

    try:
        response = gemini_service.model.generate_content(prompt)

        if not response or not response.text:
            return fallback

        response_text = response.text.strip()

        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return fallback

        json_text = response_text[start_idx:end_idx]
        result = json.loads(json_text)

        # Ensure marks are within valid range
        marks_awarded = max(0, min(result.get("marks_awarded", 0), max_marks))
        rubric_scores = result.get("rubric_scores") or {k: 1 for k in default_rubric_max}
        rubric_max = result.get("rubric_max") or default_rubric_max

        return {
            "marks_awarded": marks_awarded,
            "feedback": result.get("feedback", "Answer evaluated."),
            "rubric_scores": rubric_scores,
            "rubric_max": rubric_max,
        }

    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return fallback

async def _generate_overall_analysis_with_gemini(
    test: MockTestResponse,
    submission: MockTestSubmission,
    question_feedback: List[AnswerFeedback],
    total_score: float,
    max_score: int
) -> Dict[str, Any]:
    """Generate overall analysis using Gemini"""
    
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    
    # Prepare data for analysis
    performance_data = []
    for feedback in question_feedback:
        performance_data.append({
            "question_type": "MCQ" if feedback.correct_answer else "Descriptive",
            "marks_awarded": feedback.marks_awarded,
            "max_marks": feedback.max_marks,
            "is_correct": feedback.is_correct if feedback.is_correct is not None else None
        })
    
    prompt = f"""Analyze this student's mock test performance and provide insights.

TEST DETAILS:
- Total Questions: {len(test.questions)}
- Total Marks: {max_score}
- Score Achieved: {total_score}
- Percentage: {percentage:.1f}%
- Time Taken: {submission.time_taken // 60} minutes {submission.time_taken % 60} seconds

QUESTION PERFORMANCE:
{json.dumps(performance_data, indent=2)}

Provide a comprehensive analysis covering strengths, areas for improvement, and study recommendations.

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "feedback_summary": "Overall performance summary with key insights",
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "improvements": ["Area for improvement 1", "Area for improvement 2", "Area for improvement 3"],
    "study_recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3", "Recommendation 4"]
}}

IMPORTANT: Return ONLY the JSON object, no additional text."""

    try:
        response = gemini_service.model.generate_content(prompt)
        
        if not response or not response.text:
            return _create_fallback_analysis(percentage)
        
        response_text = response.text.strip()
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return _create_fallback_analysis(percentage)
        
        json_text = response_text[start_idx:end_idx]
        result = json.loads(json_text)
        
        return result
        
    except Exception as e:
        print(f"Error in Gemini overall analysis: {e}")
        return _create_fallback_analysis(percentage)

def _create_fallback_analysis(percentage: float) -> Dict[str, Any]:
    """Create fallback analysis when Gemini fails"""
    if percentage >= 80:
        performance_level = "excellent"
    elif percentage >= 60:
        performance_level = "good"
    elif percentage >= 40:
        performance_level = "average"
    else:
        performance_level = "needs improvement"
    
    return {
        "feedback_summary": f"Your performance was {performance_level} with {percentage:.1f}% score. Continue practicing to improve your understanding of the concepts.",
        "strengths": [
            "Attempted all questions",
            "Showed understanding of basic concepts",
            "Completed the test within time limit"
        ],
        "improvements": [
            "Focus on conceptual clarity",
            "Practice more descriptive answers",
            "Improve time management",
            "Review incorrect answers"
        ],
        "study_recommendations": [
            "Review the syllabus thoroughly",
            "Practice previous year questions",
            "Focus on weak areas identified",
            "Seek help for difficult concepts",
            "Take regular mock tests"
        ]
    }
