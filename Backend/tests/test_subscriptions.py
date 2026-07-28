import hashlib
import hmac

import pytest

import src.services.billing_service as bs
import src.services.subscription_service as ss
import src.core.plan_enforcement as pe
import src.core.data_store as ds
from src.core.security import get_current_user_with_role
from src.services.billing_service import FakeRazorpayClient
from src.main import app
from fastapi.testclient import TestClient


class _FakeColl:
    def __init__(self):
        self.docs = {}
        self._i = 0

    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                return dict(d)
        return None

    async def find(self, q=None):
        q = q or {}
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]

    async def insert_one(self, doc):
        self._i += 1
        self.docs[str(self._i)] = dict(doc)

        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return
        if upsert:
            self._i += 1
            doc = {k: v for k, v in q.items() if k != "_id"}
            doc.update(op.get("$set", {}))
            self.docs[str(self._i)] = doc


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); subs = _FakeColl(); pays = _FakeColl(); orgs = _FakeColl()
    mt = _FakeColl(); decks = _FakeColl(); fc = _FakeColl(); aim = _FakeColl()
    pdfs = _FakeColl(); classes = _FakeColl(); usage = _FakeColl()
    study = _FakeColl()
    for name, coll in [
        ("users_collection", users), ("subscriptions_collection", subs),
        ("payments_collection", pays), ("organizations_collection", orgs),
        ("mock_tests_collection", mt), ("flashcard_decks_collection", decks),
        ("flashcards_collection", fc), ("ai_materials_collection", aim),
        ("pdfs_collection", pdfs), ("classes_collection", classes),
        ("usage_events_collection", usage), ("study_plans_collection", study),
    ]:
        monkeypatch.setattr(ds, name, coll)
        monkeypatch.setattr(ss, name, coll, raising=False)
        monkeypatch.setattr(pe, name, coll, raising=False)
    fake = FakeRazorpayClient(secret="keysecret")
    monkeypatch.setattr(bs, "get_client", lambda: fake)
    monkeypatch.setattr(ss, "RAZORPAY_KEY_ID", "rzp_test_XXX")
    users.docs["1"] = {"email": "u@x.com", "role": "student"}

    def _auth():
        return {"email": "u@x.com", "user": {"email": "u@x.com", "role": "student"}}

    app.dependency_overrides[get_current_user_with_role] = _auth
    client = TestClient(app)

    yield dict(client=client, users=users, subs=subs, pays=pays, fake=fake, secret="keysecret")

    app.dependency_overrides.pop(get_current_user_with_role, None)


def test_plans_endpoint_public(setup):
    r = setup["client"].get("/subscriptions/plans")
    assert r.status_code == 200
    plans = r.json()["plans"]
    ids = {p["plan"] for p in plans}
    assert {"starter", "pro", "premium"} <= ids


def test_checkout_creates_subscription(setup):
    r = setup["client"].post("/subscriptions/checkout", json={"plan": "pro", "billing_cycle": "monthly"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key_id"] == "rzp_test_XXX"
    assert body["amount"] == 29900
    assert body["razorpay_subscription_id"].startswith("sub_")


def test_checkout_yearly_creates_order(setup):
    r = setup["client"].post("/subscriptions/checkout", json={"plan": "premium", "billing_cycle": "yearly"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["razorpay_order_id"].startswith("order_")
    assert body["amount"] == 599000


def test_checkout_rejects_starter(setup):
    r = setup["client"].post("/subscriptions/checkout", json={"plan": "starter", "billing_cycle": "monthly"})
    # Pydantic Literal["pro","premium"] rejects "starter" at the API layer
    assert r.status_code == 422


def test_verify_activates_subscription_idempotent(setup):
    c = setup["client"]; secret = setup["secret"]
    co = c.post("/subscriptions/checkout", json={"plan": "pro", "billing_cycle": "monthly"}).json()
    sid = co["razorpay_subscription_id"]; pid = "pay_1"
    sig = hmac.new(secret.encode(), f"{pid}|{sid}".encode(), hashlib.sha256).hexdigest()
    r1 = c.post("/subscriptions/verify", json={
        "razorpay_payment_id": pid, "razorpay_subscription_id": sid, "razorpay_signature": sig})
    assert r1.status_code == 200 and r1.json()["status"] == "active"
    r2 = c.post("/subscriptions/verify", json={
        "razorpay_payment_id": pid, "razorpay_subscription_id": sid, "razorpay_signature": sig})
    assert r2.status_code == 200
    assert len([d for d in setup["pays"].docs.values() if d.get("razorpay_payment_id") == pid]) == 1


def test_verify_rejects_bad_signature(setup):
    c = setup["client"]
    co = c.post("/subscriptions/checkout", json={"plan": "pro", "billing_cycle": "monthly"}).json()
    r = c.post("/subscriptions/verify", json={
        "razorpay_payment_id": "pay_2", "razorpay_subscription_id": co["razorpay_subscription_id"],
        "razorpay_signature": "deadbeef"})
    assert r.status_code == 400


def test_me_returns_usage(setup):
    r = setup["client"].get("/subscriptions/me")
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "starter" and body["source"] == "free"
    assert "usage" in body and "mock_test" in body["usage"]