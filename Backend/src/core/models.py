from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

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
