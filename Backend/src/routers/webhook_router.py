"""Razorpay webhook receiver — source of truth for subscription state.

Raw-body HMAC verification; idempotent on X-Razorpay-Event-Id. No auth dep —
authenticity is established by the signature.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException

from src.core.config import RAZORPAY_WEBHOOK_SECRET
from src.services.billing_service import verify_webhook_signature
from src.core.data_store import (
    subscriptions_collection, payments_collection, users_collection,
)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

_seen_events: set = set()  # in-process idempotency cache (v1)


@router.post("/razorpay")
async def razorpay_webhook(request: Request):
    raw = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    if not verify_webhook_signature(raw, signature, RAZORPAY_WEBHOOK_SECRET):
        raise HTTPException(status_code=400, detail="invalid signature")

    import json
    try:
        payload = json.loads(raw or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    event_id = request.headers.get("X-Razorpay-Event-Id")
    if event_id:
        if event_id in _seen_events:
            return {"status": "duplicate", "event_id": event_id}
        _seen_events.add(event_id)

    event = payload.get("event")
    entity = (((payload.get("payload") or {}).get("subscription") or {}).get("entity") or {})
    sub_id = entity.get("id")
    now = datetime.now(timezone.utc)

    if event in ("subscription.activated", "subscription.charged"):
        if subscriptions_collection is not None and sub_id:
            sub = await subscriptions_collection.find_one({"razorpay_subscription_id": sub_id})
            await subscriptions_collection.update_one(
                {"razorpay_subscription_id": sub_id},
                {"$set": {"status": "active", "updated_at": now}},
            )
            if sub and users_collection is not None:
                await users_collection.update_one(
                    {"email": sub.get("user_id")},
                    {"$set": {"subscription": {"plan": sub.get("plan"), "status": "active"}}},
                )
    elif event in ("subscription.cancelled", "subscription.expired"):
        if subscriptions_collection is not None and sub_id:
            sub = await subscriptions_collection.find_one({"razorpay_subscription_id": sub_id})
            await subscriptions_collection.update_one(
                {"razorpay_subscription_id": sub_id},
                {"$set": {"status": "cancelled" if event == "subscription.cancelled" else "expired",
                          "updated_at": now}},
            )
            if sub and users_collection is not None:
                await users_collection.update_one(
                    {"email": sub.get("user_id")},
                    {"$set": {"subscription": {"plan": sub.get("plan"), "status": "cancelled"}}},
                )
    elif event == "payment.failed":
        if payments_collection is not None:
            pay = (((payload.get("payload") or {}).get("payment") or {}).get("entity") or {})
            if pay.get("id"):
                await payments_collection.update_one(
                    {"razorpay_payment_id": pay["id"]},
                    {"$set": {"status": "failed", "updated_at": now}}, upsert=True,
                )

    return {"status": "ok", "event": event}