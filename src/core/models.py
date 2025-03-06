from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    paragraphs: Optional[int] = None
