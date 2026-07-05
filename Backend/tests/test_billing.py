import hashlib
import hmac

import src.services.billing_service as bs


def test_webhook_signature_valid():
    secret = "whsec"
    body = b'{"event":"subscription.activated"}'
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert bs.verify_webhook_signature(body, sig, secret) is True


def test_webhook_signature_bad_secret():
    body = b'{"event":"x"}'
    sig = hmac.new(b"wrong", body, hashlib.sha256).hexdigest()
    assert bs.verify_webhook_signature(body, sig, "correct") is False


def test_webhook_signature_empty_rejected():
    assert bs.verify_webhook_signature(b"{}", "", "whsec") is False
    assert bs.verify_webhook_signature(b"{}", "somesig", "") is False


def test_payment_signature_uses_concat():
    """Razorpay payment signature = HMAC over f"{payment_id}|{subscription_id}"."""
    secret = "keysecret"
    pid, sid = "pay_123", "sub_abc"
    sig = hmac.new(secret.encode(), f"{pid}|{sid}".encode(), hashlib.sha256).hexdigest()
    fake = bs.FakeRazorpayClient(secret=secret)
    assert fake.verify_payment_signature(pid, sid, sig) is True
    assert fake.verify_payment_signature(pid, sid, "deadbeef") is False


def test_fake_client_creates_subscription_and_order():
    fake = bs.FakeRazorpayClient(secret="k")
    sub = fake.create_subscription("plan_x", "u@x.com", 29900)
    assert sub["id"].startswith("sub_") and sub["status"] == "created"
    order = fake.create_order(29900, "INR", "u@x.com")
    assert order["id"].startswith("order_") and order["amount"] == 29900
    cancelled = fake.cancel_subscription(sub["id"])
    assert cancelled["status"] == "cancelled"


def test_plan_id_for_maps_env(monkeypatch):
    monkeypatch.setattr(bs, "RAZORPAY_PLAN_PRO_MONTHLY", "plan_pro_m")
    monkeypatch.setattr(bs, "RAZORPAY_PLAN_PREMIUM_YEARLY", "plan_prem_y")
    assert bs.plan_id_for("pro", "monthly") == "plan_pro_m"
    assert bs.plan_id_for("premium", "yearly") == "plan_prem_y"
    assert bs.plan_id_for("starter", "monthly") == ""  # free has no plan id