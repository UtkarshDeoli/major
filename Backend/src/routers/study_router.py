from datetime import datetime, timezone, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.core.security import get_current_user
from src.core.data_store import focus_sessions_collection, study_plans_collection
from src.services.gemini_service import gemini_service
from src.services.auth_service import get_user_by_email

router = APIRouter(prefix="/study", tags=["Study"])


class FocusSessionCreate(BaseModel):
    task: str
    duration_minutes: int = Field(ge=1, le=180, default=25)


class FocusSessionEnd(BaseModel):
    completed: bool = True
    notes: Optional[str] = None


class StudyPlanCreate(BaseModel):
    title: str
    exam_date: Optional[str] = None
    subjects: List[str] = []
    weak_topics: List[str] = []
    hours_per_day: int = Field(ge=1, le=12, default=4)
    weeks: int = Field(ge=1, le=12, default=4)


@router.post("/focus-sessions")
async def start_focus_session(
    req: FocusSessionCreate,
    user_id: str = Depends(get_current_user),
):
    if focus_sessions_collection is None:
        raise HTTPException(503, "Database connection not available")
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "task": req.task,
        "duration_minutes": req.duration_minutes,
        "started_at": now,
        "ended_at": None,
        "completed": False,
        "notes": None,
    }
    result = await focus_sessions_collection.insert_one(doc)
    return {"session_id": str(result.inserted_id), **doc}


@router.patch("/focus-sessions/{session_id}")
async def end_focus_session(
    session_id: str,
    req: FocusSessionEnd,
    user_id: str = Depends(get_current_user),
):
    if focus_sessions_collection is None:
        raise HTTPException(503, "Database connection not available")
    from bson import ObjectId
    now = datetime.now(timezone.utc)
    result = await focus_sessions_collection.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$set": {"ended_at": now, "completed": req.completed, "notes": req.notes}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "ended_at": now.isoformat(), "completed": req.completed}


@router.get("/focus-sessions")
async def list_focus_sessions(
    limit: int = 50,
    user_id: str = Depends(get_current_user),
):
    if focus_sessions_collection is None:
        raise HTTPException(503, "Database connection not available")
    cursor = focus_sessions_collection.find({"user_id": user_id}).sort("started_at", -1).limit(limit)
    docs = await cursor.to_list(length=None)
    for d in docs:
        d["session_id"] = str(d.pop("_id"))
    return {"sessions": docs}


@router.get("/focus-stats")
async def focus_stats(user_id: str = Depends(get_current_user)):
    if focus_sessions_collection is None:
        raise HTTPException(503, "Database connection not available")
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    cursor = focus_sessions_collection.find({"user_id": user_id})
    docs = await cursor.to_list(length=None)

    total_minutes = 0
    completed = 0
    weekly_minutes = 0
    for d in docs:
        started = d.get("started_at")
        ended = d.get("ended_at")
        duration = d.get("duration_minutes", 0)
        if ended and started:
            duration = min(duration, int((ended - started).total_seconds() / 60))
        total_minutes += max(0, duration)
        if d.get("completed"):
            completed += 1
        if started and started >= week_ago:
            weekly_minutes += max(0, duration)

    return {
        "total_minutes": total_minutes,
        "weekly_minutes": weekly_minutes,
        "sessions_count": len(docs),
        "completed_sessions": completed,
    }


@router.post("/plans")
async def create_study_plan(
    req: StudyPlanCreate,
    user_id: str = Depends(get_current_user),
):
    if study_plans_collection is None:
        raise HTTPException(503, "Database connection not available")
    if not gemini_service:
        raise HTTPException(503, "Gemini service is not available")

    user = await get_user_by_email(user_id)
    language = user.get("preferred_language", "en") if user else "en"

    prompt = f"""You are an expert study coach. Create a weekly study plan for an Indian competitive exam student.

Title: {req.title}
Subjects: {', '.join(req.subjects) if req.subjects else 'All subjects'}
Weak topics to prioritize: {', '.join(req.weak_topics) if req.weak_topics else 'None specified'}
Study hours per day: {req.hours_per_day}
Number of weeks: {req.weeks}
Exam date: {req.exam_date or 'Not set'}

Respond ONLY with valid JSON in this exact format:
{{
    "weeks": [
        {{
            "week": 1,
            "focus": "Overall focus for the week",
            "days": [
                {{
                    "day": "Monday",
                    "tasks": [
                        {{
                            "subject": "Subject name",
                            "topic": "Specific topic",
                            "activity": "What to do (read, practice, revision, mock test)",
                            "minutes": 60,
                            "resource": "Optional document or note"
                        }}
                    ]
                }}
            ]
        }}
    ]
}}

Include all 7 days for each week. Spread subjects evenly and prioritize weak topics."""

    if language and language != "en":
        prompt += f"\n\nIMPORTANT: Respond entirely in language code {language}."

    try:
        response = gemini_service.model.generate_content(prompt)
        text = response.text.strip() if response and response.text else ""
        start = text.find("{")
        end = text.rfind("}") + 1
        plan_data = {"weeks": []}
        if start != -1 and end > 0:
            import json
            plan_data = json.loads(text[start:end])
    except Exception as e:
        raise HTTPException(500, f"Failed to generate study plan: {e}")

    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "title": req.title,
        "exam_date": req.exam_date,
        "subjects": req.subjects,
        "weak_topics": req.weak_topics,
        "hours_per_day": req.hours_per_day,
        "weeks": req.weeks,
        "plan": plan_data,
        "created_at": now,
        "updated_at": now,
    }
    result = await study_plans_collection.insert_one(doc)
    return {"plan_id": str(result.inserted_id), **doc}


@router.get("/plans")
async def list_study_plans(user_id: str = Depends(get_current_user)):
    if study_plans_collection is None:
        raise HTTPException(503, "Database connection not available")
    cursor = study_plans_collection.find({"user_id": user_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=None)
    for d in docs:
        d["plan_id"] = str(d.pop("_id"))
    return {"plans": docs}


@router.delete("/plans/{plan_id}")
async def delete_study_plan(plan_id: str, user_id: str = Depends(get_current_user)):
    if study_plans_collection is None:
        raise HTTPException(503, "Database connection not available")
    from bson import ObjectId
    result = await study_plans_collection.delete_one({"_id": ObjectId(plan_id), "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Plan not found")
    return {"deleted": True}


@router.patch("/plans/{plan_id}/progress")
async def update_plan_progress(
    plan_id: str,
    week: int,
    day: str,
    task_index: int,
    completed: bool,
    user_id: str = Depends(get_current_user),
):
    if study_plans_collection is None:
        raise HTTPException(503, "Database connection not available")
    from bson import ObjectId
    doc = await study_plans_collection.find_one({"_id": ObjectId(plan_id), "user_id": user_id})
    if not doc:
        raise HTTPException(404, "Plan not found")

    plan = doc.get("plan", {})
    try:
        plan["weeks"][week - 1]["days"][day]["tasks"][task_index]["completed"] = completed
    except (IndexError, KeyError):
        raise HTTPException(400, "Invalid progress path")

    await study_plans_collection.update_one(
        {"_id": ObjectId(plan_id)},
        {"$set": {"plan": plan, "updated_at": datetime.now(timezone.utc)}},
    )
    return {"updated": True}
