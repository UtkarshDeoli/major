"""Organization (coaching/school) + seat license logic."""
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from src.core.data_store import (
    users_collection, organizations_collection, org_invites_collection,
)
from src.services import billing_service
from src.core.config import RAZORPAY_KEY_ID
from src.core.plans import PLAN_PRICES


def _gen_code() -> str:
    return secrets.token_urlsafe(6)[:8].upper()


async def create_org(owner_id: str, name: str, brand_name: Optional[str],
                     tier: str, seats_total: int, billing_cycle: str) -> dict:
    if organizations_collection is None or users_collection is None:
        raise HTTPException(503, "Database connection not available")
    existing = await organizations_collection.find_one({"owner_user_id": owner_id})
    if existing:
        raise HTTPException(409, "You already own an organization")
    now = datetime.now(timezone.utc)
    org_id = secrets.token_urlsafe(8)
    doc = {
        "org_id": org_id, "name": name, "brand_name": brand_name,
        "owner_user_id": owner_id, "tier": tier, "seats_total": seats_total,
        "seats_used": 0, "status": "active", "billing_cycle": billing_cycle,
        "created_at": now, "updated_at": now,
    }
    await organizations_collection.insert_one(doc)
    await users_collection.update_one(
        {"email": owner_id},
        {"$set": {"role": "subadmin", "org_id": org_id, "org_joined_at": now}},
    )
    # v1: bill the owner upfront for the seat bundle via a one-time order.
    # Renewal is manual / admin-managed for now; recurring seat subscription is
    # a follow-up once per-seat Razorpay plans are created in the dashboard.
    amount = PLAN_PRICES.get((tier, billing_cycle), 0) * max(seats_total, 0)
    order = billing_service.get_client().create_order(amount, "INR", owner_id)
    return {
        "org_id": org_id,
        "checkout": {
            "razorpay_order_id": order["id"], "key_id": RAZORPAY_KEY_ID,
            "amount": amount, "currency": "INR",
        },
    }


async def get_org_by_owner(owner_id: str) -> Optional[dict]:
    if organizations_collection is None:
        return None
    return await organizations_collection.find_one({"owner_user_id": owner_id})


async def create_invite(owner_id: str, member_role: str, email: Optional[str]) -> dict:
    if org_invites_collection is None:
        raise HTTPException(503, "Database connection not available")
    org = await get_org_by_owner(owner_id)
    if not org:
        raise HTTPException(404, "No organization found for your account")
    code = _gen_code()
    await org_invites_collection.insert_one({
        "org_id": org["org_id"], "code": code, "member_role": member_role,
        "email": email, "created_at": datetime.now(timezone.utc),
        "expires_at": None, "used_by_user_id": None,
    })
    return {"code": code}


async def enroll(user_id: str, code: str) -> dict:
    if any(c is None for c in (org_invites_collection, organizations_collection, users_collection)):
        raise HTTPException(503, "Database connection not available")
    inv = await org_invites_collection.find_one({"code": code.upper(), "used_by_user_id": None})
    if not inv:
        raise HTTPException(404, "Invalid or already-used invite code")
    org = await organizations_collection.find_one({"org_id": inv["org_id"]})
    if not org or org.get("status") != "active":
        raise HTTPException(403, "Organization is not active")
    expires_at = org.get("expires_at")
    if expires_at and expires_at < datetime.now(timezone.utc):
        raise HTTPException(403, "Organization license has expired")
    if org.get("seats_used", 0) >= org.get("seats_total", 0):
        raise HTTPException(402, detail={
            "resource": "org_seat", "used": org.get("seats_used", 0),
            "limit": org.get("seats_total", 0), "plan": org.get("tier"),
            "upgrade_url": "/billing",
        })
    now = datetime.now(timezone.utc)
    await organizations_collection.update_one(
        {"org_id": org["org_id"]}, {"$inc": {"seats_used": 1}, "$set": {"updated_at": now}})
    await org_invites_collection.update_one(
        {"code": code.upper()}, {"$set": {"used_by_user_id": user_id}})
    await users_collection.update_one(
        {"email": user_id},
        {"$set": {"org_id": org["org_id"], "member_role": inv["member_role"],
                  "org_joined_at": now}})
    return {"org_id": org["org_id"], "member_role": inv["member_role"]}


async def list_members(owner_id: str) -> list:
    org = await get_org_by_owner(owner_id)
    if not org or users_collection is None:
        return []
    users = await users_collection.find({"org_id": org["org_id"]})
    return [{"email": u.get("email"), "name": u.get("name"),
             "member_role": u.get("member_role")} for u in users]


async def remove_member(owner_id: str, member_email: str) -> dict:
    if users_collection is None or organizations_collection is None:
        raise HTTPException(503, "Database connection not available")
    org = await get_org_by_owner(owner_id)
    if not org:
        raise HTTPException(404, "No organization")
    member = await users_collection.find_one({"email": member_email, "org_id": org["org_id"]})
    if not member:
        raise HTTPException(404, "Member not found in your org")
    await users_collection.update_one(
        {"email": member_email},
        {"$set": {"org_id": None, "member_role": None, "org_joined_at": None}})
    await organizations_collection.update_one(
        {"org_id": org["org_id"]}, {"$inc": {"seats_used": -1}})
    return {"removed": member_email}


async def add_seats(owner_id: str, add_seats: int) -> dict:
    org = await get_org_by_owner(owner_id)
    if not org:
        raise HTTPException(404, "No organization")
    amount = PLAN_PRICES.get((org["tier"], org.get("billing_cycle", "monthly")), 0) * add_seats
    order = billing_service.get_client().create_order(amount, "INR", owner_id)
    return {
        "checkout": {
            "razorpay_order_id": order["id"], "key_id": RAZORPAY_KEY_ID,
            "amount": amount, "currency": "INR",
        }
    }