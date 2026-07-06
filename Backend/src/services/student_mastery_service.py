"""Student mastery tracking for adaptive difficulty.

Mastery scores per topic are updated after each mock-test submission. The mock
-test generation service reads them to bias the difficulty mix toward the
student's weak areas.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import HTTPException

from src.core.data_store import student_mastery_collection


async def update_mastery_from_submission(
    user_id: str,
    question_feedback: List[Dict],
) -> None:
    """Update topic mastery scores based on a mock-test submission.

    Each question should have a `topic` and a difficulty. Correct answers
    increase mastery; wrong answers decrease it. Hard questions contribute
    more than easy ones.
    """
    if student_mastery_collection is None:
        return

    now = datetime.now(timezone.utc)

    for qf in question_feedback:
        topic = qf.get("topic") or qf.get("unit")
        if not topic:
            continue

        difficulty = (qf.get("difficulty") or "medium").lower()
        max_marks = float(qf.get("max_marks", 1) or 1)
        marks_awarded = float(qf.get("marks_awarded", 0) or 0)
        accuracy = marks_awarded / max_marks

        # Weight by difficulty: hard = ±2, medium = ±1, easy = ±0.5
        weight = {"hard": 2.0, "medium": 1.0, "easy": 0.5}.get(difficulty, 1.0)
        delta = (accuracy * 2 - 1) * weight * 5  # range approx -10 to +10

        existing = await student_mastery_collection.find_one({"user_id": user_id, "topic": topic})
        if existing:
            current = float(existing.get("score", 50))
            new_score = max(0, min(100, current + delta))
            await student_mastery_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {"score": new_score, "updated_at": now}},
            )
        else:
            # Start at 50 and apply delta, clamped.
            start = max(0, min(100, 50 + delta))
            await student_mastery_collection.insert_one({
                "user_id": user_id,
                "topic": topic,
                "score": start,
                "updated_at": now,
            })


async def get_mastery_scores(user_id: str) -> Dict[str, float]:
    """Return a mapping of topic -> mastery score for a user."""
    if student_mastery_collection is None:
        return {}

    cursor = student_mastery_collection.find({"user_id": user_id})
    docs = await cursor.to_list(length=None)
    return {d["topic"]: float(d.get("score", 50)) for d in docs}


async def get_weak_topics(user_id: str, threshold: float = 50.0) -> List[str]:
    """Return topics where mastery is below the threshold."""
    scores = await get_mastery_scores(user_id)
    return sorted([topic for topic, score in scores.items() if score < threshold])


def recommended_difficulty(score: Optional[float]) -> str:
    """Map a mastery score to a recommended question difficulty."""
    if score is None:
        return "mixed"
    if score >= 75:
        return "hard"
    if score >= 45:
        return "medium"
    return "easy"


async def build_adaptive_bias(user_id: str, requested_difficulty: str) -> Dict[str, Any]:
    """Build a difficulty/topic bias object for mock-test generation.

    If the user requested adaptive mode, we blend their mastery data with the
    requested difficulty. Otherwise we return the requested difficulty as-is.
    """
    if requested_difficulty != "adaptive":
        return {"difficulty": requested_difficulty, "focus_topics": [], "weak_topics": []}

    scores = await get_mastery_scores(user_id)
    if not scores:
        return {"difficulty": "mixed", "focus_topics": [], "weak_topics": []}

    weak_topics = get_weak_topics_from_scores(scores)
    avg_score = sum(scores.values()) / len(scores)
    return {
        "difficulty": recommended_difficulty(avg_score),
        "focus_topics": weak_topics[:5],
        "weak_topics": weak_topics[:5],
    }


def get_weak_topics_from_scores(scores: Dict[str, float]) -> List[str]:
    return sorted(scores.keys(), key=lambda t: scores[t])[:10]
