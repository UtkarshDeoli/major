"""DB + plan logic for subscriptions and payments."""
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from src.core.config import RAZORPAY_KEY_ID
from src.core.plans import PLAN_PRICES, limit_for, ALL_RESOURCES, STARTER
from src.core.data_store import (
    users_collection, subscriptions_collection, payments_collection, organizations_collection,
)
from src.services import billing_service


async def get_effective_plan_with_usage(user_id: str) -> dict:
    from src.core.plan_enforcement import get_effective_plan, get_usage
    plan, source, org_id = await get_effective_plan(user_id)
    usage = {}
    for r in ALL_RESOURCES:
        usage[r] = {"used": await get_usage(user_id, r), "limit": limit_for(plan, r)}
    return {"plan": plan, "source": source, "org_id": org_id, "usage": usage}


async def create_checkout(user_id: str, plan: str, billing_cycle: str) -> dict:
    if plan == STARTER:
        raise HTTPException(status_code=400, detail="Starter is free; no checkout needed")
    amount = PLAN_PRICES.get((plan, billing_cycle))
    if amount is None:
        raise HTTPException(status_code=400, detail="Invalid plan/billing_cycle")
    client = billing_service.get_client()
    if billing_cycle == "monthly":
        pid = billing_service.plan_id_for(plan, "monthly")
        sub = client.create_subscription(pid, user_id, amount)
        return {
            "razorpay_subscription_id": sub["id"], "razorpay_order_id": None,
            "key_id": RAZORPAY_KEY_ID, "amount": amount, "currency": "INR",
        }
    # yearly -> one-time order
    order = client.create_order(amount, "INR", user_id)
    return {
        "razorpay_subscription_id": None, "razorpay_order_id": order["id"],
        "key_id": RAZORPAY_KEY_ID, "amount": amount, "currency": "INR",
    }


async def verify_and_activate(user_id: str, payment_id: str,
                              subscription_id: Optional[str], signature: str) -> dict:
    client = billing_service.get_client()
    # Idempotency: a payment with this id is already recorded?
    if payments_collection is not None:
        existing = await payments_collection.find_one({"razorpay_payment_id": payment_id})
        if existing:
            return {"status": "active", "already_verified": True}
    ok = client.verify_payment_signature(payment_id, subscription_id or "", signature)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid payment signature")
    if payments_collection is not None:
        await payments_collection.insert_one({
            "user_id": user_id, "razorpay_payment_id": payment_id,
            "razorpay_subscription_id": subscription_id, "amount": 0, "currency": "INR",
            "status": "captured", "created_at": datetime.now(timezone.utc),
        })
    plan, cycle = "pro", "monthly"  # defaults; enriched by webhook in practice
    if subscriptions_collection is not None:
        sub = await subscriptions_collection.find_one(
            {"user_id": user_id, "razorpay_subscription_id": subscription_id})
        if sub:
            plan, cycle = sub.get("plan", plan), sub.get("billing_cycle", cycle)
        await subscriptions_collection.update_one(
            {"user_id": user_id, "razorpay_subscription_id": subscription_id},
            {"$set": {"status": "active", "plan": plan, "billing_cycle": cycle,
                      "source": "self", "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        if users_collection is not None:
            await users_collection.update_one(
                {"email": user_id},
                {"$set": {"subscription": {"plan": plan, "status": "active"}}},
            )
    return {"status": "active"}


async def cancel(user_id: str) -> dict:
    if subscriptions_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    sub = await subscriptions_collection.find_one({"user_id": user_id, "status": "active"})
    if not sub:
        raise HTTPException(status_code=404, detail="No active subscription")
    rzp_id = sub.get("razorpay_subscription_id")
    if rzp_id:
        try:
            billing_service.get_client().cancel_subscription(rzp_id)
        except Exception:
            pass  # webhook will reconcile
    await subscriptions_collection.update_one(
        {"user_id": user_id, "status": "active"},
        {"$set": {"cancel_at_period_end": True, "updated_at": datetime.now(timezone.utc)}},
    )
    return {"status": "cancel_at_period_end"}


async def list_invoices(user_id: str) -> list:
    if payments_collection is None:
        return []
    docs = await payments_collection.find({"user_id": user_id})
    return [{
        "payment_id": d.get("razorpay_payment_id"), "amount": d.get("amount"),
        "currency": d.get("currency"), "status": d.get("status"),
        "created_at": d.get("created_at"),
    } for d in docs]