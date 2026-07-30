from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.security import require_role
from src.core.data_store import users_collection, mock_test_submissions_collection
from src.core.models import TeacherDashboardAnalytics, TeacherStudentAnalytics
from src.services.class_service import list_teacher_students

router = APIRouter(prefix="/teachers", tags=["Teachers"])


class StudentInfo(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    institute: Optional[str] = None
    onboarding_completed: bool = False


class ManagedStudentsResponse(BaseModel):
    students: List[StudentInfo]


@router.get("/students", response_model=ManagedStudentsResponse)
async def list_managed_students(user_info: dict = Depends(require_role("teacher"))):
    """List all students that share at least one class with the teacher."""
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    teacher_email = user_info["email"]
    students = await list_teacher_students(teacher_email)
    result: List[StudentInfo] = []
    for student in students:
        result.append(StudentInfo(
            id=str(student.get("_id", student.get("id", ""))),
            email=student.get("email"),
            name=student.get("name"),
            institute=student.get("institute"),
            onboarding_completed=student.get("onboarding_completed", False),
        ))
    return ManagedStudentsResponse(students=result)


@router.get("/analytics", response_model=TeacherDashboardAnalytics)
async def get_teacher_analytics(user_info: dict = Depends(require_role("teacher"))):
    """Aggregate analytics for all students in the teacher's classes."""
    teacher_email = user_info["email"]

    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )

    from src.core.data_store import get_teacher_classes
    classes = await get_teacher_classes(teacher_email)
    student_emails = {e for c in classes for e in c.get("student_emails", [])}
    students = []
    if student_emails:
        cursor = users_collection.find({"email": {"$in": list(student_emails)}})
        students = await cursor.to_list(length=None)

    student_analytics: List[TeacherStudentAnalytics] = []
    total_tests = 0
    class_score_sum = 0.0
    active_count = 0

    for student in students:
        student_email = student.get("email")
        if not student_email:
            continue
        submissions_cursor = mock_test_submissions_collection.find(
            {"user_id": student_email}
        ).sort("created_at", -1)
        submissions = await submissions_cursor.to_list(length=None)
        tests_taken = len(submissions)
        total_tests += tests_taken
        if tests_taken > 0:
            active_count += 1
        student_score_sum = 0.0
        strengths_set: set = set()
        weaknesses_set: set = set()
        last_active: Optional[str] = None
        for sub in submissions:
            score = float(sub.get("total_score", 0))
            max_score = float(sub.get("max_score", 1))
            percentage = (score / max_score) * 100 if max_score > 0 else 0
            student_score_sum += percentage
            class_score_sum += percentage
            strengths_set.update(sub.get("strengths", []) or [])
            weaknesses_set.update(sub.get("improvements", []) or [])
            created_at = sub.get("created_at")
            if created_at:
                last_active = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        avg_score = student_score_sum / tests_taken if tests_taken > 0 else 0
        student_analytics.append(
            TeacherStudentAnalytics(
                email=student_email,
                name=student.get("name"),
                tests_taken=tests_taken,
                average_score=round(avg_score, 2),
                last_active_at=last_active,
                strengths=list(strengths_set)[:5],
                weaknesses=list(weaknesses_set)[:5],
            )
        )

    class_average = class_score_sum / total_tests if total_tests > 0 else 0
    return TeacherDashboardAnalytics(
        total_students=len(students),
        active_students=active_count,
        total_tests_taken=total_tests,
        class_average=round(class_average, 2),
        student_analytics=student_analytics,
    )
