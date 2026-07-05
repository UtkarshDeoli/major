"""Sample-material seeding endpoint for students not enrolled with a teacher."""
from fastapi import APIRouter, Depends, HTTPException, status

from src.core.security import get_current_user
from src.core.data_store import exams_collection, users_collection
from src.services.sample_material_service import seed_sample_material

router = APIRouter(prefix="/api/sample-material", tags=["Sample Material"])


@router.post("/seed")
async def seed_sample(user_email: str = Depends(get_current_user)):
    """Seed NCERT + PYQ sample material into the student's active exam.

    Only available to students who are not enrolled with any teacher — this is
    the starter content promised to solo students.
    """
    if exams_collection is None or users_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    user = await users_collection.find_one({"email": user_email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.get("role") not in (None, "student"):
        raise HTTPException(status_code=403, detail="Sample material is for student accounts only")

    # Must be unenrolled (no teacher linkage)
    teacher_ids = user.get("teacher_ids") or []
    has_teacher = bool(teacher_ids) or bool(user.get("teacher_id"))
    if has_teacher:
        raise HTTPException(status_code=400, detail="Sample material is for students not enrolled with a teacher")

    active_exam_id = user.get("active_exam_id")
    if not active_exam_id:
        raise HTTPException(status_code=400, detail="Set an active exam before loading sample material")

    exam = await exams_collection.find_one({"_id": __import__("bson").ObjectId(active_exam_id), "user_id": user_email})
    if not exam:
        raise HTTPException(status_code=404, detail="Active exam not found")

    result = await seed_sample_material(user_email, exam)
    return result