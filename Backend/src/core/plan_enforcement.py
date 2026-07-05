"""Plan-limit enforcement for AI-generation endpoints.

Resolves a user's effective plan (own active subscription -> org tier -> free),
computes current usage, and raises HTTP 402 with an upgrade payload when a
limit is exceeded. Designed so Phase 2-5 features reuse it by adding a
resource key to core/plans.py and passing it to enforce_limit().
"""
import math
from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException

from src.core.plans import limit_for, STARTER, ALL_RESOURCES
from src.core.security import get_current_user_with_role
from src.core.data_store import (
    users_collection, subscriptions_collection, organizations_collection,
    mock_tests_collection, mock_test_submissions_collection,
    flashcards_collection, ai_materials_collection, pdfs_collection,
    classes_collection, flashcard_decks_collection, usage_events_collection,
)

_UPGRADE_URL = "/pricing"


def _start_of_month() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _period_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


async def get_effective_plan(user_id: str) -> Tuple[str, str, Optional[str]]:
    """Return (plan, source, org_id). source in {'self','org','free'}."""
    # 1. Own active subscription wins.
    if subscriptions_collection is not None:
        sub = await subscriptions_collection.find_one({"user_id": user_id, "status": "active"})
        if sub and sub.get("plan"):
            return sub["plan"], "self", None
    # 2. Org tier via the user's org_id.
    if users_collection is not None and organizations_collection is not None:
        user = await users_collection.find_one({"email": user_id})
        org_id = (user or {}).get("org_id")
        if org_id:
            org = await organizations_collection.find_one({"org_id": org_id})
            if org and org.get("status") == "active" and org.get("tier"):
                return org["tier"], "org", org_id
    # 3. Free.
    return STARTER, "free", None


async def get_usage(user_id: str, resource: str) -> float:
    if resource == "mock_test":
        if mock_tests_collection is None:
            return 0
        start = _start_of_month()
        docs = await mock_tests_collection.find({})
        return sum(
            1 for d in docs
            if (d.get("user_id") == user_id or d.get("created_by") == user_id)
            and d.get("created_at") and d["created_at"] >= start
        )
    if resource == "flashcard":
        if flashcards_collection is None or flashcard_decks_collection is None:
            return 0
        start = _start_of_month()
        decks = await flashcard_decks_collection.find({"user_id": user_id})
        deck_ids = {d.get("id") or str(d.get("_id")) for d in decks}
        cards = await flashcards_collection.find({})
        return sum(
            1 for c in cards
            if c.get("deck_id") in deck_ids
            and c.get("created_at") and c["created_at"] >= start
        )
    if resource == "ai_material":
        if ai_materials_collection is None:
            return 0
        start = _start_of_month()
        docs = await ai_materials_collection.find({"user_id": user_id})
        return sum(1 for d in docs if d.get("created_at") and d["created_at"] >= start)
    if resource == "chat_message":
        if usage_events_collection is None:
            return 0
        ev = await usage_events_collection.find_one(
            {"user_id": user_id, "resource": "chat_message", "period_key": _period_key()}
        )
        return float(ev.get("count", 0)) if ev else 0
    if resource == "doc_storage":
        if pdfs_collection is None:
            return 0
        docs = await pdfs_collection.find({"user_id": user_id})
        return float(sum(int(d.get("size", 0)) for d in docs))
    if resource == "class_count":
        if classes_collection is None:
            return 0
        return float(len(await classes_collection.find({"teacher_id": user_id})))
    return 0


async def increment_usage(user_id: str, resource: str, amount: int = 1) -> None:
    """Bump a usage counter. Only chat_message is tracked via usage_events;
    every other resource's usage is derived from its own collection."""
    if resource != "chat_message" or usage_events_collection is None:
        return
    key = {"user_id": user_id, "resource": "chat_message", "period_key": _period_key()}
    existing = await usage_events_collection.find_one(key)
    if existing:
        await usage_events_collection.update_one(key, {"$inc": {"count": amount}})
    else:
        doc = dict(key)
        doc["count"] = amount
        doc["updated_at"] = datetime.now(timezone.utc)
        await usage_events_collection.insert_one(doc)


def enforce_limit(resource: str):
    """FastAPI dependency: 402 with an upgrade payload when the limit is hit."""
    if resource not in ALL_RESOURCES:
        raise ValueError(f"unknown resource: {resource}")

    async def _dep(user_info: dict = Depends(get_current_user_with_role)) -> dict:
        user_id = user_info["email"]
        plan, _source, _org_id = await get_effective_plan(user_id)
        limit = limit_for(plan, resource)
        if limit == math.inf:
            return user_info
        used = await get_usage(user_id, resource)
        if used >= limit:
            raise HTTPException(
                status_code=402,
                detail={
                    "resource": resource, "used": used, "limit": limit,
                    "plan": plan, "upgrade_url": _UPGRADE_URL,
                },
            )
        return user_info

    return _dep