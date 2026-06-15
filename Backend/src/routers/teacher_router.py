from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.core.security import require_role
from src.core.data_store import users_collection, mock_test_submissions_collection
from src.core.models import TeacherDashboardAnalytics, TeacherStudentAnalytics

router = APIRouter(prefix="/teachers", tags=["Teachers"])


@router.get("/analytics", response_model=TeacherDashboardAnalytics)
async def get_teacher_analytics(user_info: dict = Depends(require_role("teacher"))):
    """
    Return aggregate analytics for all students assigned to the current teacher.
    """
    teacher_email = user_info["email"]

    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )

    # Find every student whose teacher_id matches the authenticated teacher
    students_cursor = users_collection.find({"teacher_id": teacher_email})
    students = await students_cursor.to_list(length=None)

    student_analytics: List[TeacherStudentAnalytics] = []
    total_tests_taken = 0
    class_score_sum = 0.0
    active_students = 0

    for student in students:
        student_email = student.get("email")
        if not student_email:
            continue

        submissions_cursor = mock_test_submissions_collection.find(
            {"user_id": student_email}
        ).sort("created_at", -1)
        submissions = await submissions_cursor.to_list(length=None)

        tests_taken = len(submissions)
        total_tests_taken += tests_taken
        if tests_taken > 0:
            active_students += 1

        student_score_sum = 0.0
        strengths_set: set = set()
        weaknesses_set: set = set()
        last_active_at: Optional[str] = None

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
                last_active_at = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

        average_score = student_score_sum / tests_taken if tests_taken > 0 else 0.0

        student_analytics.append(
            TeacherStudentAnalytics(
                email=student_email,
                name=student.get("name"),
                tests_taken=tests_taken,
                average_score=round(average_score, 2),
                last_active_at=last_active_at,
                strengths=list(strengths_set)[:5],
                weaknesses=list(weaknesses_set)[:5],
            )
        )

    total_students = len(students)
    class_average = class_score_sum / total_tests_taken if total_tests_taken > 0 else 0.0

    return TeacherDashboardAnalytics(
        total_students=total_students,
        active_students=active_students,
        total_tests_taken=total_tests_taken,
        class_average=round(class_average, 2),
        student_analytics=student_analytics
    )
