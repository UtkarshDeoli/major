from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class TeacherStudentAnalytics(BaseModel):
    student_email: str
    tests_taken: int
    average_score: float
    last_active_at: Optional[datetime] = None
    strengths: List[str]
    weaknesses: List[str]


class TeacherDashboardAnalytics(BaseModel):
    total_students: int
    active_students: int
    total_tests_taken: int
    class_average: float
    students: List[TeacherStudentAnalytics]
