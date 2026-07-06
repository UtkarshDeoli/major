from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.security import get_current_user, require_role
from src.core.data_store import (
    mock_test_submissions_collection,
    mock_tests_collection,
    users_collection,
    materials_collection,
    pdfs_collection,
)
from src.core.models import TeacherDashboardAnalytics, TeacherStudentAnalytics

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class SectionAnalytics(BaseModel):
    section: str
    attempts: int
    correct: int
    max_marks: float
    marks_awarded: float
    accuracy: float  # 0-100


class SubjectAnalytics(BaseModel):
    subject: str
    tests_taken: int
    average_score: float
    total_marks: float
    max_marks: float
    last_test_at: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []
    sections: List[SectionAnalytics] = []
    weak_sections: List[str] = []


class WeeklyActivity(BaseModel):
    day: str
    hours: float
    quizzes: int


class TrendDelta(BaseModel):
    score_delta: float
    tests_delta: int


class StudentAnalyticsResponse(BaseModel):
    email: str
    tests_taken: int
    average_score: float
    best_score: float
    total_time_spent_seconds: int
    subject_wise: List[SubjectAnalytics]
    recent_submissions: List[Dict]
    documents: int = 0
    study_streak: int = 0
    weekly_activity: List[WeeklyActivity] = []
    completion: float = 0.0
    consistency: float = 0.0
    trend: TrendDelta


def _to_dt(value) -> Optional[datetime]:
    """Coerce a stored created_at value (datetime or ISO string) to a tz-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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
            "sections": defaultdict(lambda: {"attempts": 0, "correct": 0, "max_marks": 0.0, "marks_awarded": 0.0}),
        }
    )

    recent_submissions: List[Dict] = []
    # (datetime, percentage, time_taken) per submission, for streak/weekly/trend/consistency
    activity_points: List[tuple] = []

    # Batch-fetch test metadata (for question unit/topic -> section weakness) in one query
    test_ids = {sub.get("test_id") for sub in submissions if sub.get("test_id")}
    qid_to_section: Dict[str, str] = {}
    if test_ids and mock_tests_collection is not None:
        test_cursor = mock_tests_collection.find({"test_id": {"$in": list(test_ids)}}, {"test_id": 1, "questions": 1})
        async for tdoc in test_cursor:
            for q in tdoc.get("questions", []) or []:
                qid = q.get("id")
                section = q.get("unit") or q.get("topic")
                if qid and section:
                    qid_to_section[qid] = section

    for sub in submissions:
        score = float(sub.get("total_score", 0))
        max_score = float(sub.get("max_score", 1))
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        total_score += percentage
        best_score = max(best_score, percentage)
        time_taken = int(sub.get("time_taken", 0) or 0)
        total_time += time_taken

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

        # Per-section (unit/topic) accuracy from question-level feedback
        for qf in sub.get("question_feedback", []) or []:
            qid = qf.get("question_id")
            section = qid_to_section.get(qid)
            if not section:
                continue
            sec = entry["sections"][section]
            sec["attempts"] += 1
            sec["max_marks"] += float(qf.get("max_marks", 0) or 0)
            sec["marks_awarded"] += float(qf.get("marks_awarded", 0) or 0)
            if qf.get("is_correct") is True:
                sec["correct"] += 1

        dt = _to_dt(created_at)
        if dt is not None:
            activity_points.append((dt, percentage, time_taken))

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
        sections: List[SectionAnalytics] = []
        weak_sections: List[str] = []
        for sec_name, sec in entry["sections"].items():
            accuracy = (sec["marks_awarded"] / sec["max_marks"] * 100) if sec["max_marks"] > 0 else 0
            sections.append(SectionAnalytics(
                section=sec_name,
                attempts=sec["attempts"],
                correct=sec["correct"],
                max_marks=sec["max_marks"],
                marks_awarded=sec["marks_awarded"],
                accuracy=round(accuracy, 2),
            ))
            # A section is "weak" if accuracy < 50% across at least 2 attempts
            if sec["attempts"] >= 2 and accuracy < 50:
                weak_sections.append(sec_name)
        sections.sort(key=lambda s: s.accuracy)
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
                sections=sections,
                weak_sections=weak_sections,
            )
        )

    # --- Derived metrics -----------------------------------------------------
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)

    # Weekly activity: last 7 days, oldest -> newest, labelled by short weekday.
    weekly: Dict[str, WeeklyActivity] = {}
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        weekly[day.strftime("%a")] = WeeklyActivity(day=day.strftime("%a"), hours=0.0, quizzes=0)

    active_days_last_7: set = set()
    for dt, _pct, time_taken in activity_points:
        day_key = dt.strftime("%a")
        if day_key in weekly and dt >= week_ago:
            weekly[day_key].quizzes += 1
            weekly[day_key].hours += time_taken / 3600.0
            active_days_last_7.add(dt.date())

    weekly_activity = [weekly[day.strftime("%a")] for day in (now - timedelta(days=i) for i in range(6, -1, -1))]

    # Study streak: consecutive days with >=1 activity, counting back from the most recent active day.
    activity_days = sorted({dt.date() for dt, _p, _t in activity_points}, reverse=True)
    study_streak = 0
    if activity_days:
        expected = activity_days[0]
        for d in activity_days:
            if d == expected:
                study_streak += 1
                expected = d - timedelta(days=1)
            else:
                break

    consistency = round(len(active_days_last_7) / 7 * 100, 2)

    # Completion: assigned tests that have >=1 submission / total assigned tests.
    completion = 0.0
    if mock_tests_collection is not None:
        assigned_count = await mock_tests_collection.count_documents({"assigned_to": user_email})
        if assigned_count > 0:
            assigned_cursor = mock_tests_collection.find({"assigned_to": user_email}, {"test_id": 1})
            assigned_docs = await assigned_cursor.to_list(length=None)
            assigned_test_ids = {doc.get("test_id") for doc in assigned_docs if doc.get("test_id")}
            submitted_test_ids = {sub.get("test_id") for sub in submissions if sub.get("test_id")}
            completed = len(assigned_test_ids & submitted_test_ids)
            completion = round(completed / assigned_count * 100, 2)

    # Trend: this week vs last week (avg score and submission count).
    def _window_stats(start: datetime, end: datetime) -> tuple:
        pts = [pct for dt, pct, _t in activity_points if start <= dt < end]
        count = len(pts)
        avg = sum(pts) / count if count > 0 else 0.0
        return avg, count

    this_avg, this_count = _window_stats(week_ago, now)
    last_avg, last_count = _window_stats(two_weeks_ago, week_ago)
    trend = TrendDelta(
        score_delta=round(this_avg - last_avg, 2),
        tests_delta=this_count - last_count,
    )

    # Documents: count of PDFs the user has uploaded.
    documents = 0
    if pdfs_collection is not None:
        documents = await pdfs_collection.count_documents({"user_id": user_email})

    return StudentAnalyticsResponse(
        email=user_email,
        tests_taken=tests_taken,
        average_score=round(total_score / tests_taken, 2) if tests_taken > 0 else 0,
        best_score=round(best_score, 2),
        total_time_spent_seconds=total_time,
        subject_wise=subject_wise,
        recent_submissions=recent_submissions[:10],
        documents=documents,
        study_streak=study_streak,
        weekly_activity=weekly_activity,
        completion=completion,
        consistency=consistency,
        trend=trend,
    )


@router.get("/teacher/alerts")
async def get_teacher_alerts(user_info: dict = Depends(require_role("teacher"))):
    """Return at-risk students for the logged-in teacher.

    Flags:
    - score_drop: last 2 tests avg vs previous 3 avg drops > 15 points
    - inactive: no submission in the last 7 days
    - low_mastery: more than 2 sections with accuracy < 40%
    """
    from src.core.data_store import mock_test_submissions_collection, users_collection

    teacher_email = user_info["email"]
    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(503, "Database connection not available")

    students_cursor = users_collection.find({"teacher_id": teacher_email})
    students = await students_cursor.to_list(length=None)

    alerts = []
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    for student in students:
        student_email = student.get("email")
        if not student_email:
            continue

        cursor = mock_test_submissions_collection.find({"user_id": student_email}).sort("created_at", -1)
        submissions = await cursor.to_list(length=None)

        flags = []
        if not submissions:
            alerts.append({
                "student_email": student_email,
                "name": student.get("name"),
                "flags": ["no_activity"],
                "last_active_at": None,
                "average_score": 0.0,
            })
            continue

        percentages = []
        for sub in submissions:
            score = float(sub.get("total_score", 0))
            max_score = float(sub.get("max_score", 1))
            percentages.append((score / max_score * 100) if max_score > 0 else 0)

        # Inactive flag
        latest = submissions[0].get("created_at")
        latest_dt = _to_dt(latest)
        if latest_dt is None or latest_dt < week_ago:
            flags.append("inactive")

        # Score drop flag
        if len(percentages) >= 5:
            recent_avg = sum(percentages[:2]) / 2
            previous_avg = sum(percentages[2:5]) / 3
            if previous_avg - recent_avg > 15:
                flags.append("score_drop")

        # Low mastery flag
        weak_sections = []
        for sub in submissions[:10]:  # look at recent 10
            for qf in sub.get("question_feedback", []) or []:
                max_marks = float(qf.get("max_marks", 0) or 0)
                marks_awarded = float(qf.get("marks_awarded", 0) or 0)
                topic = qf.get("topic") or qf.get("unit")
                if topic and max_marks > 0:
                    accuracy = marks_awarded / max_marks * 100
                    if accuracy < 40:
                        weak_sections.append(topic)
        if len(set(weak_sections)) > 2:
            flags.append("low_mastery")

        avg_score = sum(percentages) / len(percentages) if percentages else 0.0
        if avg_score < 40:
            flags.append("low_average")

        if flags:
            alerts.append({
                "student_email": student_email,
                "name": student.get("name"),
                "flags": list(set(flags)),
                "last_active_at": latest_dt.isoformat() if latest_dt else None,
                "average_score": round(avg_score, 2),
            })

    return {"alerts": alerts}


@router.get("/teacher/insights")
async def get_teacher_insights(user_info: dict = Depends(require_role("teacher"))):
    """Return per-student weak topics and recommended next actions."""
    from src.core.data_store import mock_test_submissions_collection, users_collection
    from src.services.student_mastery_service import get_mastery_scores

    teacher_email = user_info["email"]
    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(503, "Database connection not available")

    students_cursor = users_collection.find({"teacher_id": teacher_email})
    students = await students_cursor.to_list(length=None)

    insights = []
    for student in students:
        student_email = student.get("email")
        if not student_email:
            continue

        mastery = await get_mastery_scores(student_email)
        weak_topics = sorted(mastery.keys(), key=lambda t: mastery[t])[:5]
        recommended_tests = weak_topics[:3]

        insights.append({
            "student_email": student_email,
            "name": student.get("name"),
            "mastery_scores": mastery,
            "weak_topics": weak_topics,
            "recommended_focus": recommended_tests,
            "recommended_action": "Generate adaptive mock test on " + ", ".join(recommended_tests) if recommended_tests else "No data yet",
        })

    return {"insights": insights}


@router.get("/teacher", response_model=TeacherDashboardAnalytics)
async def get_teacher_analytics(user_info: dict = Depends(require_role("teacher"))):
    """Aggregate analytics for students managed by the logged-in teacher."""
    if users_collection is None or mock_test_submissions_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    teacher_email = user_info["email"]
    cursor = users_collection.find({
        "$or": [
            {"teacher_id": teacher_email},
            {"teacher_ids": teacher_email},
        ]
    })
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
