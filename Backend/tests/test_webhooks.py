import hashlib
import hmac
import json
import sys

import pytest

import src.core.data_store as ds
from src.main import app
from fastapi.testclient import TestClient

wh = sys.modules["src.routers.webhook_router"]


class _FakeColl:
    def __init__(self):
        self.docs = {}
        self._i = 0
        self.updates = []

    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                return dict(d)
        return None

    async def insert_one(self, doc):
        self._i += 1
        self.docs[str(self._i)] = dict(doc)

        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op, upsert=False):
        self.updates.append((dict(q), dict(op)))
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return
        if upsert:
            self._i += 1
            doc = {k: v for k, v in q.items()}
            doc.update(op.get("$set", {}))
            self.docs[str(self._i)] = doc


@pytest.fixture
def setup(monkeypatch):
    subs = _FakeColl(); pays = _FakeColl(); users = _FakeColl()
    monkeypatch.setattr(ds, "subscriptions_collection", subs)
    monkeypatch.setattr(ds, "payments_collection", pays)
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(wh, "subscriptions_collection", subs)
    monkeypatch.setattr(wh, "payments_collection", pays)
    monkeypatch.setattr(wh, "users_collection", users)
    monkeypatch.setattr(wh, "RAZORPAY_WEBHOOK_SECRET", "whsec")
    subs.docs["1"] = {"user_id": "u@x.com", "razorpay_subscription_id": "sub_1", "plan": "pro"}
    # clear in-process idempotency cache between tests
    wh._seen_events.clear()
    return dict(client=TestClient(app), subs=subs, pays=pays, secret="whsec")


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_bad_signature_rejected(setup):
    r = setup["client"].post("/webhooks/razorpay", content=b"{}",
                             headers={"X-Razorpay-Signature": "bad"})
    assert r.status_code == 400


def test_subscription_activated_sets_active(setup):
    body = json.dumps({
        "event": "subscription.activated",
        "payload": {"subscription": {"entity": {"id": "sub_1"}}},
    }).encode()
    r = setup["client"].post("/webhooks/razorpay", content=body,
                             headers={"X-Razorpay-Signature": _sign(body, setup["secret"])})
    assert r.status_code == 200
    assert any("status" in op.get("$set", {}) for _, op in setup["subs"].updates)
    assert setup["subs"].docs["1"]["status"] == "active"


def test_subscription_cancelled_sets_cancelled(setup):
    body = json.dumps({
        "event": "subscription.cancelled",
        "payload": {"subscription": {"entity": {"id": "sub_1"}}},
    }).encode()
    r = setup["client"].post("/webhooks/razorpay", content=body,
                             headers={"X-Razorpay-Signature": _sign(body, setup["secret"])})
    assert r.status_code == 200
    assert setup["subs"].docs["1"]["status"] == "cancelled"


def test_payment_failed_records(setup):
    body = json.dumps({
        "event": "payment.failed",
        "payload": {"payment": {"entity": {"id": "pay_fail_1"}}},
    }).encode()
    r = setup["client"].post("/webhooks/razorpay", content=body,
                             headers={"X-Razorpay-Signature": _sign(body, setup["secret"])})
    assert r.status_code == 200
    assert any(d.get("razorpay_payment_id") == "pay_fail_1" for d in setup["pays"].docs.values())


def test_event_idempotent(setup):
    body = json.dumps({
        "event": "subscription.activated",
        "payload": {"subscription": {"entity": {"id": "sub_1"}}},
    }).encode()
    sig = _sign(body, setup["secret"])
    h = {"X-Razorpay-Signature": sig, "X-Razorpay-Event-Id": "evt_1"}
    r1 = setup["client"].post("/webhooks/razorpay", content=body, headers=h)
    r2 = setup["client"].post("/webhooks/razorpay", content=body, headers=h)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r2.json().get("status") == "duplicate"