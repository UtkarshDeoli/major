from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.core.security import get_current_user_with_role
from src.core.plans import PLAN_LIMITS, PLAN_PRICES, ALL_PLANS, STARTER
from src.services import subscription_service as svc

router = APIRouter(prefix="/subscriptions", tags=["Subscriptions"])


class PlanPublic(BaseModel):
    plan: str
    monthly_price: int
    yearly_price: int
    limits: dict


class PlansResponse(BaseModel):
    plans: list[PlanPublic]


class CheckoutRequest(BaseModel):
    plan: Literal["pro", "premium"]
    billing_cycle: Literal["monthly", "yearly"] = "monthly"


class VerifyRequest(BaseModel):
    razorpay_payment_id: str
    razorpay_subscription_id: Optional[str] = None
    razorpay_signature: str


@router.get("/plans", response_model=PlansResponse)
async def list_plans():
    out = []
    for p in ALL_PLANS:
        out.append(PlanPublic(
            plan=p,
            monthly_price=PLAN_PRICES.get((p, "monthly"), 0),
            yearly_price=PLAN_PRICES.get((p, "yearly"), 0),
            limits=PLAN_LIMITS[p],
        ))
    return PlansResponse(plans=out)


@router.get("/me")
async def get_me(user=Depends(get_current_user_with_role)):
    email = user["email"]
    info = await svc.get_effective_plan_with_usage(email)
    invoices = await svc.list_invoices(email)
    return {
        "plan": info["plan"],
        "status": "active" if info["plan"] != STARTER else "free",
        "source": info["source"],
        "usage": info["usage"],
        "invoices": invoices,
    }


@router.post("/checkout")
async def checkout(req: CheckoutRequest, user=Depends(get_current_user_with_role)):
    return await svc.create_checkout(user["email"], req.plan, req.billing_cycle)


@router.post("/verify")
async def verify(req: VerifyRequest, user=Depends(get_current_user_with_role)):
    return await svc.verify_and_activate(
        user["email"], req.razorpay_payment_id, req.razorpay_subscription_id, req.razorpay_signature)


@router.post("/cancel")
async def cancel(user=Depends(get_current_user_with_role)):
    return await svc.cancel(user["email"])


@router.get("/invoices")
async def invoices(user=Depends(get_current_user_with_role)):
    return {"invoices": await svc.list_invoices(user["email"])}