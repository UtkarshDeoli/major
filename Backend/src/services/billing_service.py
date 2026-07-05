"""Razorpay SDK wrapper with a test fake.

The real client is constructed lazily from env keys. Tests substitute
FakeRazorpayClient so no network calls occur.
"""
import hashlib
import hmac
from typing import Optional

from src.core.config import (
    RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET,
    RAZORPAY_PLAN_PRO_MONTHLY, RAZORPAY_PLAN_PREMIUM_MONTHLY,
    RAZORPAY_PLAN_PRO_YEARLY, RAZORPAY_PLAN_PREMIUM_YEARLY,
)


def plan_id_for(plan: str, cycle: str) -> str:
    """Map (plan, cycle) to the env-configured Razorpay plan id.

    Resolved at call time from the current module globals so tests that
    monkeypatch the RAZORPAY_PLAN_* attributes take effect.
    """
    mapping = {
        ("pro", "monthly"): RAZORPAY_PLAN_PRO_MONTHLY,
        ("premium", "monthly"): RAZORPAY_PLAN_PREMIUM_MONTHLY,
        ("pro", "yearly"): RAZORPAY_PLAN_PRO_YEARLY,
        ("premium", "yearly"): RAZORPAY_PLAN_PREMIUM_YEARLY,
    }
    return mapping.get((plan, cycle), "")


def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class RazorpayClient:
    """Thin wrapper over the `razorpay` SDK."""
    def __init__(self):
        import razorpay  # imported lazily so tests without the SDK still load this module
        self._client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

    def create_subscription(self, plan_id: str, customer_email: str, amount_paise: int) -> dict:
        return self._client.subscription.create({
            "plan_id": plan_id,
            "customer_notify": 1,
            "total_count": 12,
            "quantity": 1,
            "notes": {"email": customer_email},
        })

    def create_order(self, amount_paise: int, currency: str, customer_email: str) -> dict:
        return self._client.order.create({
            "amount": amount_paise, "currency": currency,
            "notes": {"email": customer_email},
        })

    def verify_payment_signature(self, payment_id: str, subscription_id: str, signature: str) -> bool:
        body = f"{payment_id}|{subscription_id}"
        expected = hmac.new(RAZORPAY_KEY_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def cancel_subscription(self, subscription_id: str) -> dict:
        return self._client.subscription.cancel(subscription_id)


class FakeRazorpayClient:
    """Deterministic in-memory stand-in for tests."""
    def __init__(self, secret: str = "keysecret"):
        self._secret = secret
        self.subscriptions = {}
        self.orders = {}
        self._n = 0

    def _next_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def create_subscription(self, plan_id: str, customer_email: str, amount_paise: int) -> dict:
        sid = self._next_id("sub")
        self.subscriptions[sid] = {"id": sid, "status": "created", "plan_id": plan_id}
        return dict(self.subscriptions[sid])

    def create_order(self, amount_paise: int, currency: str, customer_email: str) -> dict:
        oid = self._next_id("order")
        self.orders[oid] = {"id": oid, "amount": amount_paise, "currency": currency, "status": "created"}
        return dict(self.orders[oid])

    def verify_payment_signature(self, payment_id: str, subscription_id: str, signature: str) -> bool:
        body = f"{payment_id}|{subscription_id}"
        expected = hmac.new(self._secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def cancel_subscription(self, subscription_id: str) -> dict:
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id]["status"] = "cancelled"
            return dict(self.subscriptions[subscription_id])
        return {"id": subscription_id, "status": "cancelled"}


_client: Optional[RazorpayClient] = None


def get_client():
    """Return the real client (tests monkeypatch this)."""
    global _client
    if _client is None:
        _client = RazorpayClient()
    return _client