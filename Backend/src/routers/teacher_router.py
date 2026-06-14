from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.core.security import get_current_user_with_role, require_role
from src.core.data_store import users_collection

router = APIRouter(prefix="/teachers", tags=["Teachers"])


class ManageStudentRequest(BaseModel):
    student_email: str


class UnmanageStudentRequest(BaseModel):
    student_email: str


class StudentInfo(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    institute: Optional[str] = None
    onboarding_completed: bool = False


class ManagedStudentsResponse(BaseModel):
    students: List[StudentInfo]


@router.post("/students/manage", status_code=status.HTTP_200_OK)
async def manage_student(
    request: ManageStudentRequest,
    user_info: dict = Depends(require_role("teacher")),
):
    """Link a student to the logged-in teacher."""
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    teacher = user_info["user"]
    teacher_email = teacher["email"]

    update_fields = {
        "teacher_id": teacher_email,
    }
    # Propagate the teacher's sub-admin relationship down to the student so
    # sub-admin analytics and license counting stay consistent.
    if teacher.get("managed_by"):
        update_fields["managed_by"] = teacher["managed_by"]
    if teacher.get("license_id"):
        update_fields["license_id"] = teacher["license_id"]

    await users_collection.update_one(
        {"email": request.student_email},
        {"$set": update_fields}
    )

    return {"success": True, "student_email": request.student_email, "teacher_id": teacher_email}


@router.post("/students/unmanage", status_code=status.HTTP_200_OK)
async def unmanage_student(
    request: UnmanageStudentRequest,
    user_info: dict = Depends(require_role("teacher")),
):
    """Unlink a student from the logged-in teacher."""
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    teacher_email = user_info["email"]

    student = await users_collection.find_one({
        "email": request.student_email,
        "teacher_id": teacher_email,
    })
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found or not managed by you"
        )

    await users_collection.update_one(
        {"email": request.student_email},
        {
            "$unset": {
                "teacher_id": "",
                "managed_by": "",
                "license_id": "",
            }
        }
    )

    return {"success": True, "student_email": request.student_email}


@router.get("/students", response_model=ManagedStudentsResponse)
async def list_managed_students(
    user_info: dict = Depends(require_role("teacher")),
):
    """List all students managed by the logged-in teacher."""
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

    teacher_email = user_info["email"]
    cursor = users_collection.find({
        "teacher_id": teacher_email,
        "role": "student",
    })
    students = await cursor.to_list(length=None)

    result: List[StudentInfo] = []
    for student in students:
        result.append(
            StudentInfo(
                id=str(student.get("_id", "")),
                email=student.get("email"),
                name=student.get("name"),
                institute=student.get("institute"),
                onboarding_completed=student.get("onboarding_completed", False),
            )
        )

    return ManagedStudentsResponse(students=result)
