import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Path, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.core.security import require_role, get_current_user_with_role
from src.services import org_service as svc

router = APIRouter(prefix="/orgs", tags=["Organizations"])


class OrgCreateRequest(BaseModel):
    name: str
    brand_name: Optional[str] = None
    tagline: Optional[str] = None
    tier: Literal["pro", "premium"]
    seats_total: int = 1
    billing_cycle: Literal["monthly", "yearly"] = "monthly"


class OrgUpdateRequest(BaseModel):
    brand_name: Optional[str] = None
    tagline: Optional[str] = None


class InviteRequest(BaseModel):
    member_role: Literal["teacher", "student"]
    email: Optional[str] = None


class SeatsRequest(BaseModel):
    add_seats: int


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_org(req: OrgCreateRequest, user=Depends(require_role("subadmin", "admin"))):
    return await svc.create_org(user["email"], req.name, req.brand_name,
                                req.tier, req.seats_total, req.billing_cycle, req.tagline)


@router.patch("/")
async def update_org(req: OrgUpdateRequest, user=Depends(require_role("subadmin"))):
    return await svc.update_org(user["email"], req.brand_name, req.tagline)


@router.get("/{org_id}/branding")
async def get_branding(org_id: str = Path(...), user=Depends(get_current_user_with_role)):
    org = await svc.get_org_by_org_id(org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    return svc.public_branding(org)


@router.post("/logo")
async def upload_logo(file: UploadFile = File(...), user=Depends(require_role("subadmin"))):
    return await svc.upload_logo(user["email"], file)


@router.get("/{org_id}/logo")
async def get_logo(org_id: str = Path(...)):
    path = await svc.get_logo_path(org_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No logo on file")
    return FileResponse(path, media_type="image/*")


@router.get("/me")
async def my_org(user=Depends(require_role("subadmin"))):
    org = await svc.get_org_by_owner(user["email"])
    if not org:
        raise HTTPException(404, "No organization found")
    members = await svc.list_members(user["email"])
    return {
        "org": org,
        "members": members,
        "seats": {"used": org.get("seats_used", 0), "total": org.get("seats_total", 0)},
    }


@router.post("/invite")
async def invite(req: InviteRequest, user=Depends(require_role("subadmin"))):
    return await svc.create_invite(user["email"], req.member_role, req.email)


@router.post("/enroll/{code}")
async def enroll(code: str = Path(...), user=Depends(get_current_user_with_role)):
    return await svc.enroll(user["email"], code)


@router.get("/members")
async def members(user=Depends(require_role("subadmin"))):
    return {"members": await svc.list_members(user["email"])}


@router.delete("/members/{member_email}")
async def remove_member(member_email: str = Path(...), user=Depends(require_role("subadmin"))):
    return await svc.remove_member(user["email"], member_email)


@router.post("/seats")
async def add_seats(req: SeatsRequest, user=Depends(require_role("subadmin"))):
    return await svc.add_seats(user["email"], req.add_seats)