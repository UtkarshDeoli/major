"""ClassSubject CRUD + ownership checks."""
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException

from src.core.data_store import (
    classes_collection,
    class_subjects_collection,
    get_class_by_id,
    list_class_subjects,
    store_class_subject,
    get_class_subject_by_id,
    delete_class_subject,
)
from src.services.class_material_service import remove_class_subject_materials


async def _require_class_teacher(class_id: str, teacher_email: str):
    cls = await get_class_by_id(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if teacher_email not in cls.get("teacher_ids", [cls.get("teacher_id")]):
        raise HTTPException(status_code=403, detail="Not authorized to manage this class")
    return cls


async def create_class_subject(class_id: str, name: str, icon: Optional[str], teacher_email: str) -> dict:
    if classes_collection is None or class_subjects_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    cls = await _require_class_teacher(class_id, teacher_email)
    now = datetime.now(timezone.utc)
    subject_doc = {
        "class_id": class_id,
        "name": name,
        "icon": icon,
        "created_by": teacher_email,
        "created_at": now,
        "updated_at": now,
    }
    subject_id = await store_class_subject(subject_doc)

    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$addToSet": {"subject_ids": subject_id}, "$set": {"updated_at": now}},
    )

    return {"id": subject_id, **subject_doc}


async def list_subjects(class_id: str, teacher_email: str) -> List[dict]:
    await _require_class_teacher(class_id, teacher_email)
    subs = await list_class_subjects(class_id)
    return [{"id": s.pop("_id"), **s} for s in subs]


async def remove_class_subject(class_id: str, subject_id: str, teacher_email: str) -> dict:
    if classes_collection is None or class_subjects_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    cls = await _require_class_teacher(class_id, teacher_email)
    subject = await get_class_subject_by_id(subject_id)
    if not subject or subject.get("class_id") != class_id:
        raise HTTPException(status_code=404, detail="Subject not found in this class")

    await remove_class_subject_materials(class_id, subject_id, teacher_email)
    await delete_class_subject(subject_id)
    await classes_collection.update_one(
        {"_id": __import__("bson").ObjectId(class_id)},
        {"$pull": {"subject_ids": subject_id}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )
    return {"subject_id": subject_id, "deleted": True}
