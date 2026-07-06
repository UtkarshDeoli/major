from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
from bson import ObjectId

class QuestionRequest(BaseModel):
    question: str
    pdf_id: Optional[str] = None  # deprecated, kept for backward compat
    doc_ids: Optional[List[str]] = None  # multi-doc scope
    subject: Optional[str] = None
    tags: Optional[List[str]] = None

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    sources: Optional[List[Any]] = None

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

class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    doc_type: str
    size: int
    chunk_count: int
    page_count: Optional[int] = None
    has_scanned_pages: bool = False
    subject: Optional[str] = None
    tags: Optional[List[str]] = None
    processed: bool
    upload_date: datetime

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
    doc_type: Optional[str] = None
    subject: Optional[str] = None
    chunk_count: Optional[int] = None

class PDFListResponse(BaseModel):
    pdfs: List[PDFMetadata]

class DocumentListResponse(BaseModel):
    documents: List[PDFMetadata]
    
# Chat history models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    
class ChatSession(BaseModel):
    id: str
    user_id: str
    pdf_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    title: str
    messages: List[Message] = []
    created_at: datetime
    updated_at: datetime

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    pdf_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
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
    unit: Optional[str] = None       # chapter/unit tag for section-level analytics
    topic: Optional[str] = None      # topic tag for section-level analytics
    syllabus_reference: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard

class MockTestGenerationRequest(BaseModel):
    syllabus_pdf_id: str
    question_paper_pdf_ids: List[str]
    notes_pdf_id: Optional[str] = None
    num_mcq: int = 15
    num_text: int = 5
    total_marks: int = 50
    difficulty_level: Literal["easy", "medium", "hard", "adaptive", "mixed"] = "medium"
    student_email: Optional[str] = None
    focus_topics: Optional[List[str]] = None
    weak_topics: Optional[List[str]] = None
    subject: Optional[str] = None
    grading_mode: Literal["auto", "teacher"] = "auto"  # auto = AI/MCQ graded, teacher = teacher marks
    source_material_ids: Optional[List[str]] = None  # materials to ground generation on
    adaptive: bool = False  # when true, difficulty is chosen from mastery data

class MockTestResponse(BaseModel):
    test_id: str
    title: str
    questions: List[MockTestQuestion]
    total_marks: int
    time_limit: int  # in minutes
    created_at: datetime
    user_id: str
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    difficulty_level: Optional[str] = "medium"
    subject: Optional[str] = None
    grading_mode: Literal["auto", "teacher"] = "auto"
    status: Optional[str] = None  # for teacher-marked tests: "pending_review" | "graded"
    latest_submission: Optional[Dict[str, Any]] = None

class MockTestSubmission(BaseModel):
    test_id: str
    answers: Dict[str, str]  # question_id -> answer
    time_taken: int  # in seconds
    submitted_at: datetime
    subject: Optional[str] = None

class AnswerFeedback(BaseModel):
    question_id: str
    question: str
    user_answer: str
    correct_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback: str
    marks_awarded: float
    max_marks: int
    rubric_scores: Optional[Dict[str, int]] = None
    rubric_max: Optional[Dict[str, int]] = None

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

# Teacher analytics models
class TeacherStudentAnalytics(BaseModel):
    email: str
    name: Optional[str] = None
    tests_taken: int
    average_score: float
    last_active_at: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []


class TeacherDashboardAnalytics(BaseModel):
    total_students: int
    active_students: int
    total_tests_taken: int
    class_average: float
    student_analytics: List[TeacherStudentAnalytics]


class SubscriptionInfo(BaseModel):
    plan: Optional[str] = None
    status: Optional[str] = None
    expires_at: Optional[datetime] = None


class Exam(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    user_id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    is_active: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Subject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    exam_id: str
    name: str
    icon: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100)
    last_studied_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Collection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    subject_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Material(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    collection_id: str
    name: str
    type: Literal["pdf", "image", "text"] = "pdf"
    size: int = 0
    url: str
    doc_id: Optional[str] = None
    processed: bool = False
    page_count: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rag_indexed: bool = False


class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    email: str
    name: Optional[str] = None
    password_hash: Optional[str] = None
    auth_provider: Optional[str] = None  # "email" or "google"
    provider_uid: Optional[str] = None   # Google "sub" ID, or None for email users
    role: Literal["student", "teacher", "subadmin", "admin"] = "student"
    institute: Optional[str] = None
    preferred_language: str = "en"
    onboarding_completed: bool = False
    active_exam_id: Optional[str] = None
    # Teacher/student linkage (loose Mongo fields, now on the model)
    teacher_ids: List[str] = []          # students: teachers who manage this student (multiple allowed)
    teacher_id: Optional[str] = None     # legacy single-teacher link (kept for backward compat)
    managed_by: Optional[str] = None     # teacher -> sub-admin email
    class_ids: List[str] = []            # students: classes (batches) they belong to
    license_id: Optional[str] = None
    subscription: Optional[SubscriptionInfo] = None
    # Multi-tenant (org / coaching license) linkage
    org_id: Optional[str] = None
    member_role: Optional[Literal["teacher", "student"]] = None
    org_joined_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Teacher Class / Batch model — a teacher groups students (e.g. "JEE 2026 Batch")
class Class(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    teacher_id: str                      # teacher email who owns the class
    name: str                            # e.g. "JEE 2026 Batch"
    description: Optional[str] = None
    exam_preset: Optional[str] = None    # e.g. "jee-mains" — drives default subjects
    enroll_code: str                     # short shareable code
    student_emails: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Flashcard models
class Flashcard(BaseModel):
    id: str
    deck_id: str
    front: str
    back: str
    # Spaced-repetition lite
    ease: int = 2
    interval_days: int = 0
    reps: int = 0
    due_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FlashcardDeck(BaseModel):
    id: str
    user_id: str
    title: str
    subject: Optional[str] = None
    source_material_ids: List[str] = []
    source_type: Optional[str] = None   # "student" | "teacher" | "ai"
    created_by: Optional[str] = None    # teacher email if teacher-created
    card_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FlashcardDeckResponse(BaseModel):
    deck: FlashcardDeck
    cards: List[Flashcard] = []


class FlashcardDeckListResponse(BaseModel):
    decks: List[FlashcardDeck]


class GenerateFlashcardsRequest(BaseModel):
    material_ids: List[str] = []
    doc_ids: List[str] = []              # alternative: raw document ids
    subject: Optional[str] = None
    title: Optional[str] = None
    num_cards: int = 15


# AI-generated study material (summaries, notes) — shown in the right sidebar
class AIStudyMaterial(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    user_id: str                         # owner (student who generated, or assigned student)
    created_by: Optional[str] = None     # teacher email if teacher-created for student
    kind: Literal["summary", "notes", "flashcard_deck", "mock_test"] = "summary"
    title: str
    subject: Optional[str] = None
    source_material_ids: List[str] = []
    content: str = ""                    # markdown text for summary/notes
    ref_id: Optional[str] = None         # linked deck_id / test_id for flashcard/mock
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIStudyMaterialResponse(BaseModel):
    id: str
    user_id: str
    created_by: Optional[str] = None
    kind: str
    title: str
    subject: Optional[str] = None
    source_material_ids: List[str] = []
    content: str = ""
    ref_id: Optional[str] = None
    created_at: datetime


class AIStudyMaterialListResponse(BaseModel):
    materials: List[AIStudyMaterialResponse]


class GenerateSummaryRequest(BaseModel):
    material_ids: List[str] = []
    doc_ids: List[str] = []
    subject: Optional[str] = None
    title: Optional[str] = None
    style: Literal["brief", "detailed", "bullet"] = "detailed"
