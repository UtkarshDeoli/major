from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.security import get_current_user, require_role
from src.core.data_store import (
    mock_test_submissions_collection,
    users_collection,
    materials_collection,
    pdfs_collection,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class SubjectAnalytics(BaseModel):
    subject: str
    tests_taken: int
    average_score: float
    total_marks: float
    max_marks: float
    last_test_at: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []


class StudentAnalyticsResponse(BaseModel):
    email: str
    tests_taken: int
    average_score: float
    best_score: float
    total_time_spent_seconds: int
    subject_wise: List[SubjectAnalytics]
    recent_submissions: List[Dict]


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


@router.get("/student", response_model=StudentAnalyticsResponse)
async def get_student_analytics(user_email: str = Depends(get_current_user)):
    """Aggregate analytics for the currently logged-in student."""
    if mock_test_submissions_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    cursor = mock_test_submissions_collection.find({"user_id": user_email}).sort("created_at", -1)
    submissions = await cursor.to_list(length=None)

    tests_taken = len(submissions)
    total_score = 0.0
    best_score = 0.0
    total_time = 0
    subject_map: Dict[str, Dict] = defaultdict(
        lambda: {
            "tests_taken": 0,
            "total_score": 0.0,
            "max_marks": 0,
            "strengths_set": set(),
            "weaknesses_set": set(),
            "last_test_at": None,
        }
    )

    recent_submissions: List[Dict] = []

    for sub in submissions:
        score = float(sub.get("total_score", 0))
        max_score = float(sub.get("max_score", 1))
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        total_score += percentage
        best_score = max(best_score, percentage)
        total_time += int(sub.get("time_taken", 0) or 0)

        # Normalize subject from linked test metadata if available
        subject = sub.get("subject") or "General"

        entry = subject_map[subject]
        entry["tests_taken"] += 1
        entry["total_score"] += percentage
        entry["max_marks"] += max_score
        entry["strengths_set"].update(sub.get("strengths", []) or [])
        entry["weaknesses_set"].update(sub.get("improvements", []) or [])
        created_at = sub.get("created_at")
        if created_at:
            entry["last_test_at"] = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

        recent_submissions.append({
            "submission_id": sub.get("submission_id"),
            "test_id": sub.get("test_id"),
            "score": score,
            "max_score": max_score,
            "percentage": percentage,
            "time_taken": sub.get("time_taken"),
            "subject": subject,
            "created_at": sub.get("created_at"),
        })

    subject_wise: List[SubjectAnalytics] = []
    for subject, entry in subject_map.items():
        avg = entry["total_score"] / entry["tests_taken"] if entry["tests_taken"] > 0 else 0
        subject_wise.append(
            SubjectAnalytics(
                subject=subject,
                tests_taken=entry["tests_taken"],
                average_score=round(avg, 2),
                total_marks=entry["total_score"],
                max_marks=entry["max_marks"],
                last_test_at=entry["last_test_at"],
                strengths=list(entry["strengths_set"])[:5],
                weaknesses=list(entry["weaknesses_set"])[:5],
            )
        )

    return StudentAnalyticsResponse(
        email=user_email,
        tests_taken=tests_taken,
        average_score=round(total_score / tests_taken, 2) if tests_taken > 0 else 0,
        best_score=round(best_score, 2),
        total_time_spent_seconds=total_time,
        subject_wise=subject_wise,
        recent_submissions=recent_submissions[:10],
    )


@router.get("/teacher", response_model=TeacherDashboardAnalytics)
async def get_teacher_analytics(user_info: dict = Depends(require_role("teacher"))):
    """Aggregate analytics for students managed by the logged-in teacher."""
    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    teacher_email = user_info["email"]
    cursor = users_collection.find({"teacher_id": teacher_email})
    students = await cursor.to_list(length=None)

    student_analytics: List[TeacherStudentAnalytics] = []
    total_tests = 0
    class_score_sum = 0.0
    active_count = 0
    now = datetime.now(timezone.utc)

    for student in students:
        student_email = student.get("email")
        submissions_cursor = mock_test_submissions_collection.find({"user_id": student_email}).sort("created_at", -1)
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
