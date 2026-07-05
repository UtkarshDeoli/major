from datetime import datetime, timezone, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.security import require_role
from src.core.data_store import (
    users_collection, organizations_collection, subscriptions_collection, payments_collection,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


class RoleUpdate(BaseModel):
    role: Literal["student", "teacher", "subadmin", "admin"]


class StatusUpdate(BaseModel):
    active: bool


class OrgUpdate(BaseModel):
    status: Optional[Literal["active", "suspended", "expired"]] = None
    seats_total: Optional[int] = None
    expires_at: Optional[datetime] = None


class ManualActivate(BaseModel):
    plan: Literal["pro", "premium"]
    days: int = 30


@router.get("/users")
async def list_users(role: Optional[str] = Query(None), org_id: Optional[str] = Query(None),
                     limit: int = 50, skip: int = 0, _=Depends(require_role("admin"))):
    if users_collection is None:
        raise HTTPException(503, "Database connection not available")
    q = {}
    if role:
        q["role"] = role
    if org_id:
        q["org_id"] = org_id
    docs = await users_collection.find(q)
    users = [{"email": d.get("email"), "name": d.get("name"), "role": d.get("role", "student"),
              "org_id": d.get("org_id"), "active": d.get("active", True)} for d in docs]
    return {"users": users[skip:skip + limit], "total": len(users)}


@router.patch("/users/{email}/role")
async def update_role(email: str, req: RoleUpdate, _=Depends(require_role("admin"))):
    if users_collection is None:
        raise HTTPException(503, "Database connection not available")
    await users_collection.update_one({"email": email}, {"$set": {"role": req.role}})
    return {"email": email, "role": req.role}


@router.patch("/users/{email}/status")
async def update_status(email: str, req: StatusUpdate, _=Depends(require_role("admin"))):
    if users_collection is None:
        raise HTTPException(503, "Database connection not available")
    await users_collection.update_one({"email": email}, {"$set": {"active": req.active}})
    return {"email": email, "active": req.active}


@router.get("/orgs")
async def list_orgs(_=Depends(require_role("admin"))):
    if organizations_collection is None:
        return {"orgs": []}
    docs = await organizations_collection.find({})
    return {"orgs": [{"org_id": d.get("org_id"), "name": d.get("name"), "tier": d.get("tier"),
                      "seats_used": d.get("seats_used", 0), "seats_total": d.get("seats_total", 0),
                      "status": d.get("status")} for d in docs]}


@router.patch("/orgs/{org_id}")
async def update_org(org_id: str, req: OrgUpdate, _=Depends(require_role("admin"))):
    if organizations_collection is None:
        raise HTTPException(503, "Database connection not available")
    set_fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not set_fields:
        raise HTTPException(400, "No fields to update")
    set_fields["updated_at"] = datetime.now(timezone.utc)
    await organizations_collection.update_one({"org_id": org_id}, {"$set": set_fields})
    return {"org_id": org_id, "updated": set_fields}


@router.get("/subscriptions")
async def list_subs(_=Depends(require_role("admin"))):
    subs = await subscriptions_collection.find({}) if subscriptions_collection else []
    pays = await payments_collection.find({}) if payments_collection else []
    return {"subscriptions": list(subs), "payments": list(pays)}


@router.post("/subscriptions/{user_id}/activate")
async def manual_activate(user_id: str, req: ManualActivate, _=Depends(require_role("admin"))):
    if subscriptions_collection is None:
        raise HTTPException(503, "Database connection not available")
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=req.days)
    await subscriptions_collection.update_one(
        {"user_id": user_id, "source": "self"},
        {"$set": {"status": "active", "plan": req.plan, "source": "self",
                  "current_period_start": now, "current_period_end": end,
                  "updated_at": now}},
        upsert=True,
    )
    if users_collection is not None:
        await users_collection.update_one(
            {"email": user_id},
            {"$set": {"subscription": {"plan": req.plan, "status": "active"}}},
        )
    return {"user_id": user_id, "plan": req.plan, "status": "active", "expires_at": end.isoformat()}


@router.get("/analytics")
async def analytics(_=Depends(require_role("admin"))):
    users = await users_collection.find({}) if users_collection else []
    by_role = {}
    for u in users:
        r = u.get("role", "student")
        by_role[r] = by_role.get(r, 0) + 1
    subs = await subscriptions_collection.find({}) if subscriptions_collection else []
    active_subs = sum(1 for s in subs if s.get("status") == "active")
    pays = await payments_collection.find({}) if payments_collection else []
    mrr = sum(int(p.get("amount", 0)) for p in pays
              if p.get("status") == "captured" and p.get("amount"))
    orgs = await organizations_collection.find({}) if organizations_collection else []
    org_count = sum(1 for o in orgs if o.get("status") == "active")
    total_users = len(users)
    conversion = round(active_subs / total_users, 4) if total_users else 0.0
    return {"totals": {"users_by_role": by_role, "active_subscriptions": active_subs,
                       "org_count": org_count, "mrr_paise": mrr},
            "conversion": conversion}