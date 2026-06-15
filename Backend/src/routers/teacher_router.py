from collections import Counter
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

from src.core.config import SECRET_KEY, ALGORITHM
from src.core.data_store import users_collection, mock_test_submissions_collection
from src.routers.analytics_router import TeacherDashboardAnalytics, TeacherStudentAnalytics
from src.services.auth_service import get_user_by_email

router = APIRouter(prefix="/teachers", tags=["Teachers"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = await get_user_by_email(user_id)
    if user is None:
        raise credentials_exception

    # Strip the MongoDB password hash from the returned user object
    user.pop("password", None)
    return user


def require_role(role: str):
    async def role_checker(user: dict = Depends(get_current_user)):
        if user.get("role") != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {role} role"
            )
        return user
    return role_checker


@router.get("/analytics", response_model=TeacherDashboardAnalytics)
async def get_teacher_analytics(user: dict = Depends(require_role("teacher"))):
    """
    Return aggregate analytics for all students assigned to the current teacher.
    """
    teacher_email = user["email"]

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
    total_score_sum = 0.0
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
        if tests_taken > 0:
            active_students += 1
            total_tests_taken += tests_taken

        percentages = [sub.get("percentage", 0) or 0 for sub in submissions]
        average_score = sum(percentages) / len(percentages) if percentages else 0.0
        total_score_sum += average_score

        last_active_at = None
        if submissions:
            last_active_at = max(
                (sub.get("created_at") or datetime.min) for sub in submissions
            )

        strengths: List[str] = []
        weaknesses: List[str] = []
        for sub in submissions:
            strengths.extend(sub.get("strengths") or [])
            weaknesses.extend(sub.get("improvements") or [])

        # Deduplicate while preserving the most common items first
        strengths = [item for item, _ in Counter(strengths).most_common()]
        weaknesses = [item for item, _ in Counter(weaknesses).most_common()]

        student_analytics.append(
            TeacherStudentAnalytics(
                student_email=student_email,
                tests_taken=tests_taken,
                average_score=average_score,
                last_active_at=last_active_at,
                strengths=strengths,
                weaknesses=weaknesses
            )
        )

    total_students = len(student_analytics)
    class_average = total_score_sum / total_students if total_students > 0 else 0.0

    return TeacherDashboardAnalytics(
        total_students=total_students,
        active_students=active_students,
        total_tests_taken=total_tests_taken,
        class_average=class_average,
        students=student_analytics
    )
