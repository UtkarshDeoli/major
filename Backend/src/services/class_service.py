"""Class business logic."""
from typing import List, Optional
from fastapi import HTTPException

from src.core.data_store import (
    classes_collection,
    get_class_by_enroll_code,
    get_class_by_id,
    add_student_to_class,
    get_student_classes,
)


async def join_class_by_enroll_code(student_email: str, enroll_code: str) -> dict:
    if classes_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    cls = await get_class_by_enroll_code(enroll_code)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    class_id = cls["id"]
    if student_email in cls.get("student_emails", []):
        return {"class_id": class_id, "enrolled": True}
    updated = await add_student_to_class(class_id, student_email, cls.get("teacher_id"))
    if not updated:
        raise HTTPException(status_code=500, detail="Could not join class")
    return {"class_id": class_id, "enrolled": True}


async def list_student_classes(student_email: str) -> List[dict]:
    classes = await get_student_classes(student_email)
    for c in classes:
        c["student_count"] = len(c.get("student_emails", []))
    return classes


async def get_class_for_user(class_id: str, user_email: str, user_role: str) -> Optional[dict]:
    cls = await get_class_by_id(class_id)
    if not cls:
        return None
    is_teacher = user_email in cls.get("teacher_ids", [cls.get("teacher_id")])
    is_student = user_email in cls.get("student_emails", [])
    if user_role == "teacher" and not is_teacher:
        return None
    if user_role == "student" and not (is_teacher or is_student):
        return None
    return cls


async def is_student_in_class(class_id: Optional[str], user_email: str) -> bool:
    if not class_id:
        return False
    cls = await get_class_by_id(class_id)
    if not cls:
        return False
    return user_email in cls.get("student_emails", [])


async def get_class_study_content(class_id: str, user_email: str) -> dict:
    cls = await get_class_for_user(class_id, user_email, "student")
    if not cls:
        raise HTTPException(status_code=403, detail="Not authorized to view this class")
    from src.core.data_store import list_class_subjects, list_decks_by_class, list_mock_tests_by_class
    subjects = await list_class_subjects(class_id)
    decks = await list_decks_by_class(class_id)
    raw_tests = await list_mock_tests_by_class(class_id)
    tests = [
        {
            "id": str(t.get("_id")),
            "test_id": t.get("test_id") or str(t.get("_id")),
            "title": t.get("title"),
            "total_marks": t.get("total_marks"),
            "class_subject_id": t.get("class_subject_id"),
            "created_at": t.get("created_at"),
        }
        for t in raw_tests
    ]
    return {
        "class_id": class_id,
        "class_name": cls.get("name"),
        "subjects": [{"id": s.pop("_id", None), **s} for s in subjects],
        "decks": decks,
        "tests": tests,
    }
