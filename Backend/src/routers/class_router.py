"""Teacher class / batch endpoints.

A teacher groups students into classes (e.g. "JEE 2026 Batch", "NEET Batch").
Each class has a short enroll code/link that students use to enroll with that
teacher. A student may belong to multiple teachers (teacher_ids) and multiple
classes (class_ids).
"""
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from src.core.security import require_role
from src.core.plan_enforcement import enforce_limit
from src.core.data_store import (
    users_collection,
    mock_test_submissions_collection,
    store_class,
    get_class_by_id,
    get_teacher_classes,
)
from src.services import class_service

router = APIRouter(prefix="/classes", tags=["Classes"])


class ClassCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    exam_preset: Optional[str] = None


class JoinClassRequest(BaseModel):
    enroll_code: str


class AddTeacherRequest(BaseModel):
    teacher_email: str


class ClassSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    exam_preset: Optional[str] = None
    enroll_code: str
    student_count: int
    created_at: datetime


class ClassListResponse(BaseModel):
    classes: List[ClassSummary]


class StudentInClass(BaseModel):
    email: str
    name: Optional[str] = None
    tests_taken: int = 0
    average_score: float = 0.0
    last_active_at: Optional[str] = None


class ClassDetail(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    exam_preset: Optional[str] = None
    enroll_code: str
    students: List[StudentInClass] = []
    created_at: datetime


def _gen_enroll_code() -> str:
    # 6-char alphanumeric, unambiguous chars
    import random
    import string
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(6))


@router.post("/", response_model=ClassSummary, status_code=status.HTTP_201_CREATED)
async def create_class(
    request: ClassCreateRequest,
    teacher=Depends(require_role("teacher")),
    _plan: dict = Depends(enforce_limit("class_count")),
):
    """Create a new class/batch and get a shareable enroll code."""
    teacher_email = teacher["email"]
    now = datetime.now(timezone.utc)
    enroll_code = _gen_enroll_code()

    teacher_user = None
    if users_collection is not None:
        teacher_user = await users_collection.find_one({"email": teacher_email})
    org_id = teacher_user.get("org_id") if teacher_user else None

    doc = {
        "teacher_id": teacher_email,
        "name": request.name,
        "description": request.description,
        "exam_preset": request.exam_preset,
        "enroll_code": enroll_code,
        "student_emails": [],
        "org_id": org_id,
        "teacher_ids": [teacher_email],
        "subject_ids": [],
        "created_at": now,
        "updated_at": now,
    }
    class_id = await store_class(doc)
    return ClassSummary(
        id=class_id, name=request.name, description=request.description,
        exam_preset=request.exam_preset, enroll_code=enroll_code,
        student_count=0, created_at=now,
    )


@router.post("/{class_id}/teachers", status_code=status.HTTP_200_OK)
async def add_teacher(
    request: AddTeacherRequest,
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    """Add a co-teacher to a class. Both teachers must belong to the same org."""
    teacher_email = teacher["email"]
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to modify this class")

    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    target = await users_collection.find_one({"email": request.teacher_email})
    if not target:
        raise HTTPException(status_code=404, detail="Teacher not found")
    if target.get("role") != "teacher" or target.get("member_role") != "teacher":
        raise HTTPException(status_code=400, detail="User is not a teacher in an organization")
    if target.get("org_id") != cls.get("org_id"):
        raise HTTPException(status_code=403, detail="Teacher must belong to the same organization")

    from src.core.data_store import classes_collection
    if classes_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$addToSet": {"teacher_ids": request.teacher_email}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
    return {"class_id": class_id, "teacher_email": request.teacher_email, "added": True}


@router.get("/", response_model=ClassListResponse)
async def list_classes(teacher=Depends(require_role("teacher"))):
    teacher_email = teacher["email"]
    classes = await get_teacher_classes(teacher_email)
    return ClassListResponse(classes=[
        ClassSummary(
            id=c["id"], name=c["name"], description=c.get("description"),
            exam_preset=c.get("exam_preset"), enroll_code=c["enroll_code"],
            student_count=len(c.get("student_emails", [])),
            created_at=c["created_at"],
        ) for c in classes
    ])


@router.post("/join", status_code=status.HTTP_200_OK)
async def join_class(
    request: JoinClassRequest,
    student=Depends(require_role("student")),
):
    result = await class_service.join_class_by_enroll_code(student["email"], request.enroll_code.upper().strip())
    return result


@router.get("/me", response_model=ClassListResponse)
async def list_my_classes(
    student=Depends(require_role("student")),
):
    classes = await class_service.list_student_classes(student["email"])
    return ClassListResponse(classes=[ClassSummary(**c) for c in classes])


async def _build_student_in_class(email: str) -> StudentInClass:
    student = None
    if users_collection is not None:
        student = await users_collection.find_one({"email": email})
    tests_taken = 0
    avg = 0.0
    last_active = None
    if mock_test_submissions_collection is not None:
        cursor = mock_test_submissions_collection.find({"user_id": email}).sort("created_at", -1)
        subs = await cursor.to_list(length=None)
        tests_taken = len(subs)
        pct_sum = 0.0
        for s in subs:
            score = float(s.get("total_score", 0))
            mx = float(s.get("max_score", 1))
            pct_sum += (score / mx * 100) if mx > 0 else 0
            ca = s.get("created_at")
            if ca and last_active is None:
                last_active = ca.isoformat() if hasattr(ca, "isoformat") else str(ca)
        avg = round(pct_sum / tests_taken, 2) if tests_taken > 0 else 0.0
    return StudentInClass(
        email=email, name=student.get("name") if student else None,
        tests_taken=tests_taken, average_score=avg, last_active_at=last_active,
    )


class ClassStudentAnalyticsResponse(BaseModel):
    email: str
    name: Optional[str] = None
    class_ids: List[str] = []
    class_names: List[str] = []
    tests_taken: int = 0
    average_score: float = 0.0
    last_active_at: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []


class ClassStudentsAnalyticsResponse(BaseModel):
    students: List[ClassStudentAnalyticsResponse]


async def _build_student_analytics(email: str, classes: List[dict]) -> ClassStudentAnalyticsResponse:
    student = None
    if users_collection is not None:
        student = await users_collection.find_one({"email": email})
    tests_taken = 0
    avg = 0.0
    last_active = None
    strengths_set: set = set()
    weaknesses_set: set = set()
    if mock_test_submissions_collection is not None:
        cursor = mock_test_submissions_collection.find({"user_id": email}).sort("created_at", -1)
        subs = await cursor.to_list(length=None)
        tests_taken = len(subs)
        pct_sum = 0.0
        for s in subs:
            score = float(s.get("total_score", 0))
            mx = float(s.get("max_score", 1))
            pct_sum += (score / mx * 100) if mx > 0 else 0
            ca = s.get("created_at")
            if ca and last_active is None:
                last_active = ca.isoformat() if hasattr(ca, "isoformat") else str(ca)
        avg = round(pct_sum / tests_taken, 2) if tests_taken > 0 else 0.0
        for s in subs:
            strengths_set.update(s.get("strengths", []) or [])
            weaknesses_set.update(s.get("improvements", []) or [])
    class_ids = [c["id"] for c in classes if email in c.get("student_emails", [])]
    class_names = [c["name"] for c in classes if email in c.get("student_emails", [])]
    return ClassStudentAnalyticsResponse(
        email=email, name=student.get("name") if student else None,
        class_ids=class_ids, class_names=class_names,
        tests_taken=tests_taken, average_score=avg,
        last_active_at=last_active,
        strengths=list(strengths_set)[:5], weaknesses=list(weaknesses_set)[:5],
    )


@router.get("/students", response_model=ClassStudentsAnalyticsResponse)
async def list_all_teacher_students(teacher=Depends(require_role("teacher"))):
    from src.services.class_service import list_teacher_students
    students = await list_teacher_students(teacher["email"])
    classes = await get_teacher_classes(teacher["email"])
    analytics = [await _build_student_analytics(s["email"], classes) for s in students]
    return ClassStudentsAnalyticsResponse(students=analytics)


@router.get("/{class_id}", response_model=ClassDetail)
async def get_class_detail(
    class_id: str = Path(...),
    user=Depends(require_role()),  # any authenticated user
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    user_email = user["email"]
    is_teacher = user_email in cls.get("teacher_ids", [cls.get("teacher_id")])
    is_student = user_email in cls.get("student_emails", [])
    if not is_teacher and not is_student:
        raise HTTPException(status_code=403, detail="Not authorized to view this class")
    students = []
    if is_teacher:
        students = [await _build_student_in_class(e) for e in cls.get("student_emails", [])]
    return ClassDetail(
        id=cls["id"], name=cls["name"], description=cls.get("description"),
        exam_preset=cls.get("exam_preset"), enroll_code=cls["enroll_code"],
        students=students, created_at=cls["created_at"],
    )


@router.get("/{class_id}/content")
async def get_class_content(
    class_id: str = Path(...),
    user=Depends(require_role()),
):
    result = await class_service.get_class_study_content(class_id, user["email"])
    return result


@router.get("/{class_id}/students")
async def get_class_students(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher["email"] not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized")
    students = []
    if users_collection is not None:
        cursor = users_collection.find({"email": {"$in": cls.get("student_emails", [])}})
        students = await cursor.to_list(length=None)
    return {"students": [
        {"id": str(u.get("_id")), "email": u.get("email"), "name": u.get("name")}
        for u in students
    ]}


@router.get("/{class_id}/students/analytics", response_model=ClassStudentsAnalyticsResponse)
async def get_class_students_analytics(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher["email"] not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized")
    emails = cls.get("student_emails", [])
    analytics = [await _build_student_analytics(e, [cls]) for e in emails]
    return ClassStudentsAnalyticsResponse(students=analytics)


@router.get("/{class_id}/tests")
async def get_class_tests(
    class_id: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher["email"] not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized")
    from src.core.data_store import list_mock_tests_by_class
    tests = await list_mock_tests_by_class(class_id)
    return {"tests": tests}


@router.delete("/{class_id}/students/{student_email}", status_code=status.HTTP_200_OK)
async def remove_student(
    class_id: str = Path(...),
    student_email: str = Path(...),
    teacher=Depends(require_role("teacher")),
):
    teacher_email = teacher["email"]
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to modify this class")

    from src.core.data_store import classes_collection
    if classes_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$pull": {"student_emails": student_email}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
    if users_collection is not None:
        await users_collection.update_one(
            {"email": student_email},
            {"$pull": {"teacher_ids": teacher_email, "class_ids": class_id}},
        )
    return {"class_id": class_id, "student_email": student_email, "removed": True}