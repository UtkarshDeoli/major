from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

class QuestionRequest(BaseModel):
    question: str
    pdf_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    paragraphs: Optional[int] = None

# PDF related models
class PDFUploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    upload_date: datetime
    user_id: str
    file_path: str
    processed: bool
    tags: Optional[List[str]] = None
    
class PDFMetadata(BaseModel):
    id: str
    filename: str
    size: int
    upload_date: datetime
    user_id: str
    file_path: str
    processed: bool
    title: Optional[str] = None
    description: Optional[str] = None
    page_count: Optional[int] = None
    vector_db_path: Optional[str] = None
    tags: Optional[List[str]] = None

class PDFListResponse(BaseModel):
    pdfs: List[PDFMetadata]
    
# Chat history models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    
class ChatSession(BaseModel):
    id: str
    user_id: str
    pdf_id: Optional[str] = None
    title: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    
class ChatSessionResponse(BaseModel):
    id: str
    title: str
    pdf_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int
    
class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionResponse]
    
class ChatMessageRequest(BaseModel):
    content: str
    
class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime

# Question Paper Analysis Models
class QuestionPaperAnalysisRequest(BaseModel):
    syllabus_pdf_id: str
    question_paper_pdf_ids: List[str]

class UnitAnalysis(BaseModel):
    unit_name: str
    weightage_percentage: float
    important_topics: List[str]
    difficulty_level: str
    recommendation: str

class QuestionPattern(BaseModel):
    question_type: str
    marks_distribution: Dict[str, int]
    frequency: int
    examples: List[str]

class QuestionPaperAnalysisResponse(BaseModel):
    analysis_id: str
    overall_summary: str
    focus_areas: List[str]
    unit_wise_analysis: List[UnitAnalysis]
    question_patterns: List[QuestionPattern]
    sample_questions: List[str]
    preparation_strategy: str
    created_at: datetime

# Mock Test Models
class MockTestQuestion(BaseModel):
    id: str
    type: str  # 'mcq' or 'text'
    question: str
    options: Optional[List[str]] = None
    correctAnswer: Optional[str] = None
    marks: int

class MockTestGenerationRequest(BaseModel):
    syllabus_pdf_id: str
    question_paper_pdf_ids: List[str]
    notes_pdf_id: Optional[str] = None
    num_mcq: int = 15
    num_text: int = 5
    total_marks: int = 50
    difficulty_level: str = "medium"  # easy, medium, hard

class MockTestResponse(BaseModel):
    test_id: str
    title: str
    questions: List[MockTestQuestion]
    total_marks: int
    time_limit: int  # in minutes
    created_at: datetime
    user_id: str
    difficulty_level: Optional[str] = "medium"
    latest_submission: Optional[Dict[str, Any]] = None

class MockTestSubmission(BaseModel):
    test_id: str
    answers: Dict[str, str]  # question_id -> answer
    time_taken: int  # in seconds
    submitted_at: datetime

class AnswerFeedback(BaseModel):
    question_id: str
    question: str
    user_answer: str
    correct_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback: str
    marks_awarded: float
    max_marks: int

class MockTestAnalysisResponse(BaseModel):
    submission_id: str
    test_id: str
    total_score: float
    max_score: int
    percentage: float
    time_taken: int
    feedback_summary: str
    question_feedback: List[AnswerFeedback]
    strengths: List[str]
    improvements: List[str]
    study_recommendations: List[str]
    created_at: datetime

class MockTestListResponse(BaseModel):
    tests: List[MockTestResponse]

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    email: str
    name: str
    password_hash: str
    role: str = "student"  # "student" or "teacher"
    institute: Optional[str] = None
    preferred_language: str = "en"
    onboarding_completed: bool = False
    active_exam_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
