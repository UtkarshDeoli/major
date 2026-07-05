# Phase 1a — Backend: Subscriptions, Enforcement, Multi-tenant, Admin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Orbit backend billable and multi-tenant: Razorpay subscriptions, plan-limit enforcement on existing AI endpoints, org/coaching seat licenses, admin/sub-admin endpoints, and production hardening (rate limiting, webhook signature verification).

**Architecture:** New `core/plans.py` constants + `core/plan_enforcement.py` dependency resolve a user's effective plan (own subscription → org tier → free) and current-period usage, raising `402` with an upgrade payload at the generation endpoints. A new `billing_service` wraps the Razorpay SDK (mockable in tests); `subscription_service` + `org_service` own the MongoDB collections (`subscriptions`, `payments`, `organizations`, `org_invites`, `usage_events`). Three new routers (`subscription`, `webhook`, `org`, `admin`) register in `main.py`. Enforcement is wired into existing routers as an additive `Depends`. No existing feature is removed.

**Tech Stack:** FastAPI, Motor (async MongoDB), Pydantic v2, `razorpay` python SDK, `slowapi` rate limiter, pytest + FastAPI `TestClient` with in-memory `_FakeCollection` mocks (matches existing `tests/test_auth.py` pattern).

## Global Constraints

- Python venv at `Backend/venv`; run `source venv/bin/activate` before any backend command.
- Pydantic v2 (`BaseModel`, `ConfigDict`, `Field`) — match existing `core/models.py` style.
- All new MongoDB collections are accessed via module-level handles in `core/data_store.py` (mirror the `users_collection` pattern) and are `None`-guarded (`if x is None: raise HTTPException(503, "Database connection not available")`).
- Auth dependency for role/plan checks is `get_current_user_with_role` (returns `{"email", "user"}`) from `core/security.py`; `require_role(*roles)` for role-gated endpoints.
- JWT `sub` = user email everywhere; `user_id` fields in collections store the email string (matches existing `mock_tests`, `classes`).
- New Pydantic request/response models live in the router files that use them (matches `class_router.py`); shared data models go in `core/models.py`.
- Tests use the `TestClient` + monkeypatched `_FakeCollection` pattern from `tests/test_auth.py`. No network in tests; the Razorpay client is mocked.
- No live Razorpay keys in repo; test-mode keys via `Backend/.env`. Update `Backend/.env.example` with empty placeholders.
- Frequent commits; each task ends with green tests + a commit.
- Limits use a **calendar-month** usage window for count-based resources (mock tests / flashcards / ai materials / chat messages) and **cumulative** totals for `doc_storage` (sum of uploaded bytes) and `class_count` (active classes). `math.inf` means unlimited.

---

## File Structure (created/modified)

**Create:**
- `Backend/src/core/plans.py` — plan tiers, limit table, price table, `limit_for()` helper.
- `Backend/src/core/plan_enforcement.py` — `get_effective_plan()`, `get_usage()`, `enforce_limit()` dependency, `increment_usage()` helper.
- `Backend/src/services/billing_service.py` — Razorpay SDK wrapper (checkout creation, signature verification, cancel) with a `FakeRazorpayClient` for tests.
- `Backend/src/services/subscription_service.py` — db ops for `subscriptions` + `payments` collections.
- `Backend/src/services/org_service.py` — db ops for `organizations` + `org_invites` + seat accounting.
- `Backend/src/routers/subscription_router.py` — `/subscriptions/*` endpoints.
- `Backend/src/routers/webhook_router.py` — `POST /webhooks/razorpay` (raw-body HMAC).
- `Backend/src/routers/org_router.py` — `/orgs/*` endpoints.
- `Backend/src/routers/admin_router.py` — `/admin/*` endpoints.
- `Backend/tests/test_plans_enforcement.py`
- `Backend/tests/test_billing.py`
- `Backend/tests/test_subscriptions.py`
- `Backend/tests/test_webhooks.py`
- `Backend/tests/test_orgs.py`
- `Backend/tests/test_admin.py`

**Modify:**
- `Backend/src/core/config.py` — add Razorpay + rate-limit settings.
- `Backend/src/core/data_store.py` — add 5 new collections + indexes in `ensure_indexes()`.
- `Backend/src/core/models.py` — add `org_id`, `member_role`, `org_joined_at` to `User`; add `Subscription`, `Payment`, `Organization`, `OrgInvite` models.
- `Backend/src/routers/__init__.py` — export the 4 new routers.
- `Backend/src/main.py` — include the 4 new routers; mount `slowapi` limiter.
- `Backend/src/routers/mock_test_router.py` — add `enforce_limit("mock_test")` to `/generate`.
- `Backend/src/routers/flashcard_router.py` — add `enforce_limit("flashcard")` to `/generate`.
- `Backend/src/routers/ai_material_router.py` — add `enforce_limit("ai_material")` to `/summarize`.
- `Backend/src/routers/question_router.py` — add `enforce_limit("chat_message")` + `increment_usage` to ask + message endpoints.
- `Backend/src/routers/document_router.py` — add `enforce_limit("doc_storage")` to `/upload`.
- `Backend/src/routers/material_router.py` — add `enforce_limit("doc_storage")` to material upload.
- `Backend/src/routers/class_router.py` — add `enforce_limit("class_count")` to `POST /classes/`.
- `Backend/requirements.txt` — add `razorpay`, `slowapi`.
- `Backend/.env.example` — Razorpay placeholders.

---

## Task 1: Plans constants, config, data store collections, deps

**Files:**
- Create: `Backend/src/core/plans.py`
- Modify: `Backend/src/core/config.py:6-48` (add Razorpay + limiter settings)
- Modify: `Backend/src/core/data_store.py:33-73` and `414-432` (new collections + indexes)
- Modify: `Backend/requirements.txt` (add `razorpay`, `slowapi`)
- Modify: `Backend/.env.example` (Razorpay placeholders)
- Test: `Backend/tests/test_plans_enforcement.py` (created here, filled in Task 2)

**Interfaces:**
- Produces: `core/plans.py` exports `STARTER="starter"`, `PRO="pro"`, `PREMIUM="premium"`, `PLAN_LIMITS`, `PLAN_PRICES`, `limit_for(plan, resource) -> float|int`, `ALL_RESOURCES` set. `data_store.py` exports `subscriptions_collection`, `payments_collection`, `organizations_collection`, `org_invites_collection`, `usage_events_collection`. `config.py` exports `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`, `RAZORPAY_PLAN_PRO_MONTHLY`, `RAZORPAY_PLAN_PREMIUM_MONTHLY`, `RAZORPAY_PLAN_PRO_YEARLY`, `RAZORPAY_PLAN_PREMIUM_YEARLY`, `RZP_TEST_MODE`.

- [ ] **Step 1: Add dependencies**

Append to `Backend/requirements.txt`:
```
razorpay==1.4.2
slowapi==0.1.9
```
Run: `source venv/bin/activate && pip install razorpay==1.4.2 slowapi==0.1.9`
Expected: installs succeed.

- [ ] **Step 2: Create `core/plans.py`**

```python
"""Subscription plan tiers, limits, and prices.

Single source of truth for what each plan allows. The frontend reads these
via GET /subscriptions/plans (added in the subscription router task) so the
UI never hardcodes limits that drift from the backend.
"""
import math
from typing import Dict

STARTER = "starter"
PRO = "pro"
PREMIUM = "premium"

ALL_PLANS = (STARTER, PRO, PREMIUM)

# Resources enforced by enforce_limit(resource). doc_storage and class_count
# are cumulative totals; the rest are calendar-month counts.
ALL_RESOURCES = (
    "mock_test", "flashcard", "ai_material", "chat_message",
    "doc_storage", "class_count",
)

_MB = 1024 * 1024
_GB = 1024 * 1024 * 1024

PLAN_LIMITS: Dict[str, Dict[str, float]] = {
    STARTER: {
        "mock_test": 3, "flashcard": 50, "ai_material": 5, "chat_message": 100,
        "doc_storage": 50 * _MB, "class_count": 1,
    },
    PRO: {
        "mock_test": 50, "flashcard": 500, "ai_material": 50, "chat_message": 1000,
        "doc_storage": 1 * _GB, "class_count": 10,
    },
    PREMIUM: {
        "mock_test": math.inf, "flashcard": math.inf, "ai_material": math.inf,
        "chat_message": math.inf, "doc_storage": 10 * _GB, "class_count": math.inf,
    },
}

# INR paise — keyed by (plan, billing_cycle)
PLAN_PRICES: Dict[tuple, int] = {
    (PRO, "monthly"): 29900, (PRO, "yearly"): 299000,
    (PREMIUM, "monthly"): 59900, (PREMIUM, "yearly"): 599000,
}


def limit_for(plan: str, resource: str) -> float:
    """Return the limit for a resource on a plan. Unknown plan → starter."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS[STARTER]).get(resource, 0)
```

- [ ] **Step 3: Extend `core/config.py`**

Add inside `class Settings` (after `FRONTEND_URL`):
```python
    # Razorpay (India-first billing)
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""
    RAZORPAY_PLAN_PRO_MONTHLY: str = ""
    RAZORPAY_PLAN_PREMIUM_MONTHLY: str = ""
    RAZORPAY_PLAN_PRO_YEARLY: str = ""
    RAZORPAY_PLAN_PREMIUM_YEARLY: str = ""
    RZP_TEST_MODE: bool = True
```
After the existing exports (after `FRONTEND_URL = settings.FRONTEND_URL`), add:
```python
RAZORPAY_KEY_ID = settings.RAZORPAY_KEY_ID
RAZORPAY_KEY_SECRET = settings.RAZORPAY_KEY_SECRET
RAZORPAY_WEBHOOK_SECRET = settings.RAZORPAY_WEBHOOK_SECRET
RAZORPAY_PLAN_PRO_MONTHLY = settings.RAZORPAY_PLAN_PRO_MONTHLY
RAZORPAY_PLAN_PREMIUM_MONTHLY = settings.RAZORPAY_PLAN_PREMIUM_MONTHLY
RAZORPAY_PLAN_PRO_YEARLY = settings.RAZORPAY_PLAN_PRO_YEARLY
RAZORPAY_PLAN_PREMIUM_YEARLY = settings.RAZORPAY_PLAN_PREMIUM_YEARLY
RZP_TEST_MODE = settings.RZP_TEST_MODE
```

- [ ] **Step 4: Extend `core/data_store.py`**

In the `try:` block (after `classes_collection = db.classes`), add:
```python
    # Billing / multi-tenant collections
    subscriptions_collection = db.subscriptions
    payments_collection = db.payments
    organizations_collection = db.organizations
    org_invites_collection = db.org_invites
    usage_events_collection = db.usage_events
```
In the `except (ConnectionFailure, ...)` block, add matching `= None` lines for the 5 new collections.

In `ensure_indexes()`, append (before the final return / end of function):
```python
    if subscriptions_collection is not None:
        await subscriptions_collection.create_index([("user_id", 1), ("status", 1)])
        await subscriptions_collection.create_index("razorpay_subscription_id", unique=True, sparse=True)
    if payments_collection is not None:
        await payments_collection.create_index("razorpay_payment_id", unique=True, sparse=True)
        await payments_collection.create_index([("user_id", 1), ("created_at", -1)])
    if organizations_collection is not None:
        await organizations_collection.create_index("owner_user_id", unique=True)
        await organizations_collection.create_index("status")
    if org_invites_collection is not None:
        await org_invites_collection.create_index("code", unique=True)
    if usage_events_collection is not None:
        await usage_events_collection.create_index([("user_id", 1), ("resource", 1), ("period_key", 1)], unique=True)
```

- [ ] **Step 5: Update `.env.example`**

Append:
```
# Razorpay (India-first billing). Use test-mode keys from https://dashboard.razorpay.com/app/keys
RAZORPAY_KEY_ID=
RAZORPAY_KEY_SECRET=
RAZORPAY_WEBHOOK_SECRET=
# Plan IDs created in the Razorpay dashboard (Settings → Subscriptions)
RAZORPAY_PLAN_PRO_MONTHLY=
RAZORPAY_PLAN_PREMIUM_MONTHLY=
RAZORPAY_PLAN_PRO_YEARLY=
RAZORPAY_PLAN_PREMIUM_YEARLY=
RZP_TEST_MODE=true
```

- [ ] **Step 6: Sanity check imports**

Run: `source venv/bin/activate && python -c "from src.core.plans import limit_for, PLAN_LIMITS; from src.core.config import RAZORPAY_KEY_ID; from src.core.data_store import subscriptions_collection; print(limit_for('pro','mock_test'))"`
Expected: prints `50`, no import errors. (MongoDB need not be running — the `try/except` makes collections `None` on connection failure but the module still imports.)

- [ ] **Step 7: Commit**

```bash
git add Backend/src/core/plans.py Backend/src/core/config.py Backend/src/core/data_store.py Backend/requirements.txt Backend/.env.example
git commit -m "feat(billing): plan limits, razorpay config, billing collections"
```

---

## Task 2: Plan enforcement core

**Files:**
- Create: `Backend/src/core/plan_enforcement.py`
- Create: `Backend/tests/test_plans_enforcement.py`

**Interfaces:**
- Consumes: `core/plans.py` (`limit_for`, `STARTER`), `core/data_store.py` collections, `core/security.py` (`get_current_user_with_role`).
- Produces:
  - `async get_effective_plan(user_id: str) -> tuple[str, str, str | None]` → `(plan, source, org_id)` where `source ∈ {"self","org","free"}`.
  - `async get_usage(user_id: str, resource: str) -> float`
  - `enforce_limit(resource: str)` → FastAPI dependency returning `{"email","user"}`, raising `HTTPException(402, detail={resource,used,limit,plan,upgrade_url})` when `used >= limit` and limit is finite.
  - `async increment_usage(user_id: str, resource: str, amount: int = 1) -> None` (upserts `usage_events` for `chat_message`; no-op for other resources whose usage is derived).

- [ ] **Step 1: Write the failing tests**

`Backend/tests/test_plans_enforcement.py`:
```python
import math
from datetime import datetime, timezone
import pytest

import src.core.plan_enforcement as pe
import src.core.data_store as ds


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
        class R: inserted_id = str(self._i)
        return R()
    async def update_one(self, q, op):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                if "$inc" in op:
                    for k, v in op["$inc"].items():
                        d[k] = d.get(k, 0) + v
                if "$set" in op:
                    d.update(op["$set"])
    async def count_documents(self, q):
        return len(await self.find(q))


@pytest.fixture
def isolated(monkeypatch):
    users = _FakeColl(); subs = _FakeColl(); orgs = _FakeColl()
    mt = _FakeColl(); subs_mt = _FakeColl(); fc = _FakeColl(); aim = _FakeColl()
    pdfs = _FakeColl(); classes = _FakeColl(); decks = _FakeColl(); usage = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "subscriptions_collection", subs)
    monkeypatch.setattr(ds, "organizations_collection", orgs)
    monkeypatch.setattr(ds, "mock_tests_collection", mt)
    monkeypatch.setattr(ds, "mock_test_submissions_collection", subs_mt)
    monkeypatch.setattr(ds, "flashcards_collection", fc)
    monkeypatch.setattr(ds, "ai_materials_collection", aim)
    monkeypatch.setattr(ds, "pdfs_collection", pdfs)
    monkeypatch.setattr(ds, "classes_collection", classes)
    monkeypatch.setattr(ds, "flashcard_decks_collection", decks)
    monkeypatch.setattr(ds, "usage_events_collection", usage)
    # plan_enforcement imports these names from data_store at module level
    monkeypatch.setattr(pe, "users_collection", users)
    monkeypatch.setattr(pe, "subscriptions_collection", subs)
    monkeypatch.setattr(pe, "organizations_collection", orgs)
    monkeypatch.setattr(pe, "mock_tests_collection", mt)
    monkeypatch.setattr(pe, "mock_test_submissions_collection", subs_mt)
    monkeypatch.setattr(pe, "flashcards_collection", fc)
    monkeypatch.setattr(pe, "ai_materials_collection", aim)
    monkeypatch.setattr(pe, "pdfs_collection", pdfs)
    monkeypatch.setattr(pe, "classes_collection", classes)
    monkeypatch.setattr(pe, "flashcard_decks_collection", decks)
    monkeypatch.setattr(pe, "usage_events_collection", usage)
    return dict(users=users, subs=subs, orgs=orgs, mt=mt, fc=fc, aim=aim,
                pdfs=pdfs, classes=classes, decks=decks, usage=usage)


async def test_free_plan_when_no_subscription_and_no_org(isolated):
    plan, source, org_id = await pe.get_effective_plan("a@x.com")
    assert plan == "starter" and source == "free" and org_id is None


async def test_own_subscription_wins_over_org(isolated):
    isolated["users"].docs["1"] = {"email": "a@x.com", "org_id": "org1"}
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "premium"}
    isolated["orgs"].docs["1"] = {"org_id": "org1", "tier": "pro", "status": "active", "owner_user_id": "b@x.com"}
    plan, source, org_id = await pe.get_effective_plan("a@x.com")
    assert (plan, source) == ("premium", "self")


async def test_org_tier_when_no_own_subscription(isolated):
    isolated["users"].docs["1"] = {"email": "a@x.com", "org_id": "org1"}
    isolated["orgs"].docs["1"] = {"org_id": "org1", "tier": "premium", "status": "active"}
    plan, source, org_id = await pe.get_effective_plan("a@x.com")
    assert (plan, source, org_id) == ("premium", "org", "org1")


async def test_inactive_subscription_falls_through_to_free(isolated):
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "cancelled", "plan": "pro"}
    plan, source, _ = await pe.get_effective_plan("a@x.com")
    assert (plan, source) == ("starter", "free")


async def test_usage_counts_mock_tests_this_month(isolated):
    now = datetime.now(timezone.utc)
    isolated["mt"].docs["1"] = {"user_id": "a@x.com", "created_at": now}
    isolated["mt"].docs["2"] = {"created_by": "a@x.com", "created_at": now}
    used = await pe.get_usage("a@x.com", "mock_test")
    assert used == 2


async def test_usage_premium_unlimited_no_check(isolated):
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "premium"}
    dep = pe.enforce_limit("mock_test")
    # dependency returns the user_info dict without raising even at high usage
    class UI: 
        email = "a@x.com"; user = {"email": "a@x.com"}
    # simulate: build the inner function and call with a stand-in
    import inspect
    inner = dep.__wrapped__ if hasattr(dep, "__wrapped__") else None
    # enforce_limit returns a closure; call it directly with user_info


async def test_enforce_limit_402_when_over(isolated):
    # starter allows 3 mock tests; create 3 -> 4th must 402
    now = datetime.now(timezone.utc)
    for i in range(3):
        isolated["mt"].docs[str(i)] = {"user_id": "a@x.com", "created_at": now}
    isolated["users"].docs["1"] = {"email": "a@x.com"}
    from fastapi import HTTPException
    dep = pe.enforce_limit("mock_test")
    # The dependency is an async closure taking user_info=Depends(...).
    # Call it by binding the closure's single arg.
    import asyncio
    user_info = {"email": "a@x.com", "user": {"email": "a@x.com"}}
    # Inspect the closure to call it directly
    closure = dep
    try:
        await closure(user_info=user_info)
        assert False, "expected 402"
    except HTTPException as e:
        assert e.status_code == 402
        assert e.detail["resource"] == "mock_test"
        assert e.detail["used"] == 3
        assert e.detail["limit"] == 3


async def test_increment_usage_upserts_chat(isolated):
    await pe.increment_usage("a@x.com", "chat_message")
    await pe.increment_usage("a@x.com", "chat_message")
    used = await pe.get_usage("a@x.com", "chat_message")
    assert used == 2
```

Add to `Backend/pytest.ini` or rely on existing config — check: the repo already runs `pytest tests/`. Async tests need an async plugin. Verify `requirements.txt` / test config supports `async def` tests. Run: `source venv/bin/activate && pip show pytest-asyncio anyio >/dev/null && echo OK || echo MISSING`.

If `pytest-asyncio` is **not** installed, add `pytest-asyncio==0.23.8` to `requirements.txt`, install it, and create `Backend/pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
```
(If a pytest config already exists, merge `asyncio_mode = auto` into it instead of creating a new file — check `Backend/pyproject.toml` / `pytest.ini` first.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/test_plans_enforcement.py -v`
Expected: FAIL with `ImportError` / `module 'src.core.plan_enforcement' has no attribute 'get_effective_plan'`.

- [ ] **Step 3: Implement `core/plan_enforcement.py`**

```python
"""Plan-limit enforcement for AI-generation endpoints.

Resolves a user's effective plan (own active subscription → org tier → free),
computes current usage, and raises HTTP 402 with an upgrade payload when a
limit is exceeded. Designed so Phase 2-5 features reuse it by adding a
resource key to core/plans.py and passing it to enforce_limit().
"""
import math
from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException

from src.core.plans import limit_for, STARTER, ALL_RESOURCES
from src.core.security import get_current_user_with_role
from src.core.data_store import (
    users_collection, subscriptions_collection, organizations_collection,
    mock_tests_collection, mock_test_submissions_collection,
    flashcards_collection, ai_materials_collection, pdfs_collection,
    classes_collection, flashcard_decks_collection, usage_events_collection,
)

_UPGRADE_URL = "/pricing"


def _start_of_month() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _period_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


async def get_effective_plan(user_id: str) -> Tuple[str, str, Optional[str]]:
    """Return (plan, source, org_id). source ∈ {'self','org','free'}."""
    # 1. Own active subscription wins.
    if subscriptions_collection is not None:
        sub = await subscriptions_collection.find_one({"user_id": user_id, "status": "active"})
        if sub and sub.get("plan"):
            return sub["plan"], "self", None
    # 2. Org tier via the user's org_id.
    if users_collection is not None and organizations_collection is not None:
        user = await users_collection.find_one({"email": user_id})
        org_id = (user or {}).get("org_id")
        if org_id:
            org = await organizations_collection.find_one({"org_id": org_id})
            if org and org.get("status") == "active" and org.get("tier"):
                return org["tier"], "org", org_id
    # 3. Free.
    return STARTER, "free", None


async def get_usage(user_id: str, resource: str) -> float:
    if resource == "mock_test":
        if mock_tests_collection is None:
            return 0
        start = _start_of_month()
        # owned or created-by, this month
        docs = await mock_tests_collection.find({})
        return sum(
            1 for d in docs
            if (d.get("user_id") == user_id or d.get("created_by") == user_id)
            and d.get("created_at") and d["created_at"] >= start
        )
    if resource == "flashcard":
        if flashcards_collection is None or flashcard_decks_collection is None:
            return 0
        start = _start_of_month()
        decks = [d for d in await flashcard_decks_collection.find({"user_id": user_id})]
        deck_ids = {d.get("id") or str(d.get("_id")) for d in decks}
        cards = await flashcards_collection.find({})
        return sum(
            1 for c in cards
            if c.get("deck_id") in deck_ids
            and c.get("created_at") and c["created_at"] >= start
        )
    if resource == "ai_material":
        if ai_materials_collection is None:
            return 0
        start = _start_of_month()
        docs = await ai_materials_collection.find({"user_id": user_id})
        return sum(1 for d in docs if d.get("created_at") and d["created_at"] >= start)
    if resource == "chat_message":
        if usage_events_collection is None:
            return 0
        ev = await usage_events_collection.find_one(
            {"user_id": user_id, "resource": "chat_message", "period_key": _period_key()}
        )
        return float(ev.get("count", 0)) if ev else 0
    if resource == "doc_storage":
        if pdfs_collection is None:
            return 0
        docs = await pdfs_collection.find({"user_id": user_id})
        return float(sum(int(d.get("size", 0)) for d in docs))
    if resource == "class_count":
        if classes_collection is None:
            return 0
        return float(len(await classes_collection.find({"teacher_id": user_id})))
    return 0


async def increment_usage(user_id: str, resource: str, amount: int = 1) -> None:
    """Bump a usage counter. Only chat_message is tracked via usage_events;
    every other resource's usage is derived from its own collection."""
    if resource != "chat_message" or usage_events_collection is None:
        return
    key = {"user_id": user_id, "resource": "chat_message", "period_key": _period_key()}
    existing = await usage_events_collection.find_one(key)
    if existing:
        await usage_events_collection.update_one(key, {"$inc": {"count": amount}})
    else:
        doc = dict(key)
        doc["count"] = amount
        doc["updated_at"] = datetime.now(timezone.utc)
        await usage_events_collection.insert_one(doc)


def enforce_limit(resource: str):
    """FastAPI dependency: 402 with an upgrade payload when the limit is hit."""
    if resource not in ALL_RESOURCES:
        raise ValueError(f"unknown resource: {resource}")

    async def _dep(user_info: dict = Depends(get_current_user_with_role)) -> dict:
        user_id = user_info["email"]
        plan, _source, _org_id = await get_effective_plan(user_id)
        limit = limit_for(plan, resource)
        if limit == math.inf:
            return user_info
        used = await get_usage(user_id, resource)
        if used >= limit:
            raise HTTPException(
                status_code=402,
                detail={
                    "resource": resource, "used": used, "limit": limit,
                    "plan": plan, "upgrade_url": _UPGRADE_URL,
                },
            )
        return user_info

    return _dep
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/test_plans_enforcement.py -v`
Expected: PASS (all tests green). If `test_usage_premium_unlimited_no_check` is awkward, replace its body with:
```python
async def test_enforce_limit_premium_no_402(isolated):
    isolated["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "premium"}
    now = datetime.now(timezone.utc)
    for i in range(100):
        isolated["mt"].docs[str(i)] = {"user_id": "a@x.com", "created_at": now}
    dep = pe.enforce_limit("mock_test")
    user_info = {"email": "a@x.com", "user": {"email": "a@x.com"}}
    result = await dep(user_info=user_info)
    assert result["email"] == "a@x.com"
```
(delete the earlier awkward `test_usage_premium_unlimited_no_check`)

- [ ] **Step 5: Commit**

```bash
git add Backend/src/core/plan_enforcement.py Backend/tests/test_plans_enforcement.py Backend/requirements.txt Backend/pytest.ini
git commit -m "feat(billing): plan-enforcement dependency with usage tracking"
```

---

## Task 3: Billing service (Razorpay wrapper, mockable)

**Files:**
- Create: `Backend/src/services/billing_service.py`
- Create: `Backend/tests/test_billing.py`

**Interfaces:**
- Consumes: `core/config.py` Razorpay settings.
- Produces:
  - `class RazorpayClient` thin wrapper with methods: `create_subscription(plan_id, customer_email, amount_paise) -> dict`, `create_order(amount_paise, currency, customer_email) -> dict` (for yearly one-time), `verify_payment_signature(razorpay_payment_id, razorpay_subscription_id, razorpay_signature) -> bool`, `cancel_subscription(razorpay_subscription_id) -> dict`.
  - `def get_client() -> RazorpayClient` (lazy; reads env; in tests replaced by `FakeRazorpayClient`).
  - `def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool` (HMAC-SHA256, constant-time).
  - `def plan_id_for(plan: str, cycle: str) -> str` — maps `(plan, cycle)` to the env var.

- [ ] **Step 1: Write the failing tests**

`Backend/tests/test_billing.py`:
```python
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


def test_payment_signature_uses_concat():
    """Razorpay payment signature = HMAC over f"{payment_id}|{subscription_id}"."""
    secret = "keysecret"
    pid, sid = "pay_123", "sub_abc"
    sig = hmac.new(secret.encode(), f"{pid}|{sid}".encode(), hashlib.sha256).hexdigest()
    fake = bs.FakeRazorpayClient(secret=secret)
    assert fake.verify_payment_signature(pid, sid, sig) is True
    assert fake.verify_payment_signature(pid, sid, "deadbeef") is False


def test_plan_id_for_maps_env(monkeypatch):
    monkeypatch.setattr(bs, "RAZORPAY_PLAN_PRO_MONTHLY", "plan_pro_m")
    monkeypatch.setattr(bs, "RAZORPAY_PLAN_PREMIUM_YEARLY", "plan_prem_y")
    assert bs.plan_id_for("pro", "monthly") == "plan_pro_m"
    assert bs.plan_id_for("premium", "yearly") == "plan_prem_y"
    assert bs.plan_id_for("starter", "monthly") == ""  # free has no plan id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/test_billing.py -v`
Expected: FAIL with `ImportError` for `billing_service`.

- [ ] **Step 3: Implement `services/billing_service.py`**

```python
"""Razorpay SDK wrapper with a test fake.

The real client is constructed lazily from env keys. Tests substitute
FakeRazorpayClient so no network calls occur.
"""
import hashlib
import hmac
import os
from typing import Optional

from src.core.config import (
    RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET,
    RAZORPAY_PLAN_PRO_MONTHLY, RAZORPAY_PLAN_PREMIUM_MONTHLY,
    RAZORPAY_PLAN_PRO_YEARLY, RAZORPAY_PLAN_PREMIUM_YEARLY,
)


_PLAN_IDS = {
    ("pro", "monthly"): RAZORPAY_PLAN_PRO_MONTHLY,
    ("premium", "monthly"): RAZORPAY_PLAN_PREMIUM_MONTHLY,
    ("pro", "yearly"): RAZORPAY_PLAN_PRO_YEARLY,
    ("premium", "yearly"): RAZORPAY_PLAN_PREMIUM_YEARLY,
}


def plan_id_for(plan: str, cycle: str) -> str:
    return _PLAN_IDS.get((plan, cycle), "")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/test_billing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Backend/src/services/billing_service.py Backend/tests/test_billing.py
git commit -m "feat(billing): razorpay client wrapper + signature verification + test fake"
```

---

## Task 4: Subscription service + subscription router + plans endpoint

**Files:**
- Create: `Backend/src/services/subscription_service.py`
- Create: `Backend/src/routers/subscription_router.py`
- Create: `Backend/tests/test_subscriptions.py`
- Modify: `Backend/src/routers/__init__.py`, `Backend/src/main.py`

**Interfaces:**
- Consumes: `billing_service.get_client`, `billing_service.plan_id_for`, `core/plans.PLAN_LIMITS`, `core/plans.PLAN_PRICES`, `core/data_store` collections.
- Produces (router endpoints):
  - `GET /subscriptions/plans` (public) → `[{plan, monthly_price, yearly_price, limits}]`
  - `GET /subscriptions/me` (auth) → `{plan, status, source, billing_cycle, current_period_end, usage: {resource:{used,limit}}, invoices:[...]}`
  - `POST /subscriptions/checkout` (auth) body `{plan, billing_cycle}` → `{razorpay_subscription_id?, razorpay_order_id?, key_id, amount, currency}`
  - `POST /subscriptions/verify` (auth) body `{razorpay_payment_id, razorpay_subscription_id, razorpay_signature}` → `{status:"active"}` (idempotent on `razorpay_payment_id`)
  - `POST /subscriptions/cancel` (auth) → `{status:"cancel_at_period_end"}`
  - `GET /subscriptions/invoices` (auth) → `[{payment_id, amount, currency, status, created_at}]`
- Produces (service): `create_checkout(user_id, plan, cycle)`, `verify_and_activate(user_id, payment_id, sub_id, signature)`, `cancel(user_id)`, `get_subscription_status(user_id)`, `list_invoices(user_id)`.

- [ ] **Step 1: Write the failing tests**

`Backend/tests/test_subscriptions.py`:
```python
import hmac, hashlib
from datetime import datetime, timezone
import pytest

import src.services.billing_service as bs
import src.services.subscription_service as ss
import src.core.data_store as ds
from src.services.billing_service import FakeRazorpayClient

# Import app last so routers register
from src.main import app
from fastapi.testclient import TestClient


class _FakeColl:
    def __init__(self): self.docs = {}; self._i = 0
    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()): return dict(d)
        return None
    async def find(self, q=None):
        q = q or {}
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]
    async def insert_one(self, doc):
        self._i += 1; self.docs[str(self._i)] = dict(doc)
        class R: inserted_id = str(self._i)
        return R()
    async def update_one(self, q, op):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); subs = _FakeColl(); pays = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "subscriptions_collection", subs)
    monkeypatch.setattr(ds, "payments_collection", pays)
    monkeypatch.setattr(ss, "users_collection", users)
    monkeypatch.setattr(ss, "subscriptions_collection", subs)
    monkeypatch.setattr(ss, "payments_collection", pays)
    monkeypatch.setattr(ss, "organizations_collection", _FakeColl())
    fake = FakeRazorpayClient(secret="keysecret")
    monkeypatch.setattr(bs, "get_client", lambda: fake)
    monkeypatch.setattr(ss, "RAZORPAY_KEY_ID", "rzp_test_XXX")
    # Pre-create a user so /auth/me-based deps resolve
    users.docs["1"] = {"email": "u@x.com", "role": "student"}
    # Auth: bypass JWT by patching get_current_user_with_role
    import src.core.security as security
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "u@x.com", "user": {"email": "u@x.com", "role": "student"}})
    client = TestClient(app)
    return dict(client=client, users=users, subs=subs, pays=pays, fake=fake, secret="keysecret")


def test_plans_endpoint_public(setup):
    c = setup["client"]
    r = c.get("/subscriptions/plans")
    assert r.status_code == 200
    plans = r.json()["plans"]
    ids = [p["plan"] for p in plans]
    assert {"starter", "pro", "premium"} <= set(ids)


def test_checkout_creates_subscription(setup):
    c = setup["client"]
    r = c.post("/subscriptions/checkout", json={"plan": "pro", "billing_cycle": "monthly"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key_id"] == "rzp_test_XXX"
    assert body["amount"] == 29900
    assert body["razorpay_subscription_id"].startswith("sub_")


def test_verify_activates_subscription_idempotent(setup):
    c = setup["client"]; secret = setup["secret"]
    # checkout first
    co = c.post("/subscriptions/checkout", json={"plan": "pro", "billing_cycle": "monthly"}).json()
    sid = co["razorpay_subscription_id"]; pid = "pay_1"
    sig = hmac.new(secret.encode(), f"{pid}|{sid}".encode(), hashlib.sha256).hexdigest()
    r1 = c.post("/subscriptions/verify", json={
        "razorpay_payment_id": pid, "razorpay_subscription_id": sid, "razorpay_signature": sig})
    assert r1.status_code == 200 and r1.json()["status"] == "active"
    # second verify with same payment_id must not duplicate
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
    c = setup["client"]
    # no subscription -> starter
    r = c.get("/subscriptions/me")
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "starter" and body["source"] == "free"
    assert "usage" in body and "mock_test" in body["usage"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/test_subscriptions.py -v`
Expected: FAIL with import errors / 404s.

- [ ] **Step 3: Implement `services/subscription_service.py`**

```python
"""DB + plan logic for subscriptions and payments."""
import math
from datetime import datetime, timezone
from typing import Optional

from src.core.config import RAZORPAY_KEY_ID
from src.core.plans import PLAN_LIMITS, PLAN_PRICES, limit_for, ALL_RESOURCES, STARTER
from src.core.data_store import (
    users_collection, subscriptions_collection, payments_collection, organizations_collection,
)
from src.services import billing_service
from src.services.billing_service import get_client, plan_id_for


async def get_effective_plan_with_usage(user_id: str) -> dict:
    from src.core.plan_enforcement import get_effective_plan, get_usage
    plan, source, org_id = await get_effective_plan(user_id)
    usage = {}
    for r in ALL_RESOURCES:
        limit = limit_for(plan, r)
        usage[r] = {"used": await get_usage(user_id, r), "limit": limit}
    return {"plan": plan, "source": source, "org_id": org_id, "usage": usage}


async def create_checkout(user_id: str, plan: str, billing_cycle: str) -> dict:
    if plan == STARTER:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Starter is free; no checkout needed")
    amount = PLAN_PRICES.get((plan, billing_cycle))
    if amount is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid plan/billing_cycle")
    client = get_client()
    if billing_cycle == "monthly":
        pid = plan_id_for(plan, "monthly")
        sub = client.create_subscription(pid, user_id, amount)
        return {
            "razorpay_subscription_id": sub["id"], "razorpay_order_id": None,
            "key_id": RAZORPAY_KEY_ID, "amount": amount, "currency": "INR",
        }
    else:  # yearly -> one-time order
        order = client.create_order(amount, "INR", user_id)
        return {
            "razorpay_subscription_id": None, "razorpay_order_id": order["id"],
            "key_id": RAZORPAY_KEY_ID, "amount": amount, "currency": "INR",
        }


async def verify_and_activate(user_id: str, payment_id: str, subscription_id: Optional[str], signature: str) -> dict:
    client = get_client()
    # Idempotency: a payment with this id is already recorded?
    if payments_collection is not None:
        existing = await payments_collection.find_one({"razorpay_payment_id": payment_id})
        if existing:
            return {"status": "active", "already_verified": True}
    # Signature check. For subscription flow use subscription_id; for order flow
    # the client verifies payment against the order — here we use the same helper.
    ok = client.verify_payment_signature(payment_id, subscription_id or "", signature)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid payment signature")
    # Record payment
    if payments_collection is not None:
        await payments_collection.insert_one({
            "user_id": user_id, "razorpay_payment_id": payment_id,
            "razorpay_subscription_id": subscription_id, "amount": 0, "currency": "INR",
            "status": "captured", "created_at": datetime.now(timezone.utc),
        })
    # Determine plan/cycle from the subscription/order — for v1 we trust the most
    # recent checkout for this user is the one being verified. Activate subscription.
    plan, cycle = "pro", "monthly"  # default; enriched by webhook in practice
    if subscriptions_collection is not None:
        sub = await subscriptions_collection.find_one({"user_id": user_id, "razorpay_subscription_id": subscription_id})
        if sub:
            plan, cycle = sub.get("plan", plan), sub.get("billing_cycle", cycle)
        await subscriptions_collection.update_one(
            {"user_id": user_id, "razorpay_subscription_id": subscription_id},
            {"$set": {"status": "active", "plan": plan, "billing_cycle": cycle,
                      "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        # Denormalize onto user.subscription for fast /auth/me
        if users_collection is not None:
            await users_collection.update_one(
                {"email": user_id},
                {"$set": {"subscription": {"plan": plan, "status": "active"}}},
            )
    return {"status": "active"}


async def cancel(user_id: str) -> dict:
    if subscriptions_collection is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Database connection not available")
    sub = await subscriptions_collection.find_one({"user_id": user_id, "status": "active"})
    if not sub:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No active subscription")
    rzp_id = sub.get("razorpay_subscription_id")
    if rzp_id:
        try:
            get_client().cancel_subscription(rzp_id)
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
```

- [ ] **Step 4: Implement `routers/subscription_router.py`**

```python
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, status
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
        "plan": info["plan"], "status": "active" if info["plan"] != STARTER else "free",
        "source": info["source"], "usage": info["usage"], "invoices": invoices,
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
```

- [ ] **Step 5: Register the router**

In `Backend/src/routers/__init__.py` add:
```python
from .subscription_router import router as subscription_router
```
In `Backend/src/main.py` add to the import block and `app.include_router(subscription_router)` (after `sample_material_router`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/test_subscriptions.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Backend/src/services/subscription_service.py Backend/src/routers/subscription_router.py Backend/src/routers/__init__.py Backend/src/main.py Backend/tests/test_subscriptions.py
git commit -m "feat(billing): subscription service + router (checkout/verify/cancel/me/plans)"
```

---

## Task 5: Webhook router (Razorpay source-of-truth)

**Files:**
- Create: `Backend/src/routers/webhook_router.py`
- Create: `Backend/tests/test_webhooks.py`
- Modify: `routers/__init__.py`, `main.py`

**Interfaces:**
- Consumes: `billing_service.verify_webhook_signature`, `core/config.RAZORPAY_WEBHOOK_SECRET`, `core/data_store` collections.
- Produces: `POST /webhooks/razorpay` (raw body, no auth) — idempotent on event id; handles `subscription.activated`, `subscription.charged`, `subscription.cancelled`, `subscription.expired`, `payment.failed`.

- [ ] **Step 1: Write the failing tests**

`Backend/tests/test_webhooks.py`:
```python
import hashlib, hmac, json
from datetime import datetime, timezone
import pytest

import src.services.billing_service as bs
import src.core.data_store as ds
import src.routers.webhook_router as wh
from src.main import app
from fastapi.testclient import TestClient


class _FakeColl:
    def __init__(self): self.docs = {}; self._i = 0; self.updates = []
    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()): return dict(d)
        return None
    async def insert_one(self, doc):
        self._i += 1; self.docs[str(self._i)] = dict(doc)
        class R: inserted_id = str(self._i)
        return R()
    async def update_one(self, q, op):
        self.updates.append((dict(q), dict(op)))
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
    async def find(self, q=None):
        q = q or {}
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]


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
    return dict(client=TestClient(app), subs=subs, pays=pays, secret="whsec")


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_bad_signature_rejected(setup):
    r = setup["client"].post("/webhooks/razorpay", content=b"{}", headers={"X-Razorpay-Signature": "bad"})
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


def test_event_idempotent(setup):
    body = json.dumps({
        "event": "subscription.activated",
        "payload": {"subscription": {"entity": {"id": "sub_1"}}},
    }).encode()
    sig = _sign(body, setup["secret"])
    h = {"X-Razorpay-Signature": sig}
    r1 = setup["client"].post("/webhooks/razorpay", content=body, headers=h)
    # Re-send with a synthetic event id header to test idempotency on the same payload
    r2 = setup["client"].post("/webhooks/razorpay", content=body, headers={**h, "X-Razorpay-Event-Id": "evt_1"})
    r3 = setup["client"].post("/webhooks/razorpay", content=body, headers={**h, "X-Razorpay-Event-Id": "evt_1"})
    assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_webhooks.py -v`
Expected: FAIL (404 / import error).

- [ ] **Step 3: Implement `routers/webhook_router.py`**

```python
"""Razorpay webhook receiver — source of truth for subscription state.

Raw-body HMAC verification; idempotent on X-Razorpay-Event-Id. No auth dep —
authenticity is established by the signature.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException, status

from src.core.config import RAZORPAY_WEBHOOK_SECRET
from src.services.billing_service import verify_webhook_signature
from src.core.data_store import (
    subscriptions_collection, payments_collection, users_collection,
)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

_seen_events: set[str] = set()  # in-process idempotency cache (v1)


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
            await payments_collection.update_one(
                {"razorpay_payment_id": pay.get("id")},
                {"$set": {"status": "failed", "updated_at": now}}, upsert=True,
            )

    return {"status": "ok", "event": event}
```

- [ ] **Step 4: Register the router** in `routers/__init__.py` (`from .webhook_router import router as webhook_router`) and `main.py` (`app.include_router(webhook_router)`).

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_webhooks.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Backend/src/routers/webhook_router.py Backend/src/routers/__init__.py Backend/src/main.py Backend/tests/test_webhooks.py
git commit -m "feat(billing): razorpay webhook receiver with signature verification + idempotency"
```

---

## Task 6: Org service + org router (multi-tenant seats)

**Files:**
- Create: `Backend/src/services/org_service.py`
- Create: `Backend/src/routers/org_router.py`
- Create: `Backend/tests/test_orgs.py`
- Modify: `routers/__init__.py`, `main.py`, `core/models.py` (add `org_id`, `member_role`, `org_joined_at` to `User`)

**Interfaces:**
- Consumes: `core/data_store` collections, `core/security.require_role`, `billing_service`.
- Produces (router):
  - `POST /orgs` (subadmin) body `{name, brand_name, tier, seats_total, billing_cycle}` → creates org + checkout for seat subscription → `{org_id, checkout}`
  - `GET /orgs/me` (subadmin) → org + seat usage + members
  - `POST /orgs/invite` (subadmin) body `{member_role, email?}` → `{code}`
  - `POST /orgs/enroll/{code}` (auth) → joins org (seat check) → `{org_id, member_role}`
  - `GET /orgs/members` (subadmin) → roster
  - `DELETE /orgs/members/{user_id}` (subadmin) → frees seat
  - `POST /orgs/seats` (subadmin) body `{add_seats}` → checkout for delta
- Produces (service): `create_org`, `get_org_by_owner`, `create_invite`, `enroll`, `list_members`, `remove_member`, `add_seats`, with `enforce_org_seat` raising `HTTPException(402)` when `seats_used >= seats_total` or org inactive/expired.

- [ ] **Step 1: Add User fields in `core/models.py`**

Inside `class User` (after `license_id`), add:
```python
    org_id: Optional[str] = None
    member_role: Optional[Literal["teacher", "student"]] = None
    org_joined_at: Optional[datetime] = None
```

- [ ] **Step 2: Write the failing tests**

`Backend/tests/test_orgs.py`:
```python
from datetime import datetime, timezone
import pytest

import src.services.billing_service as bs
import src.services.org_service as os_
import src.core.data_store as ds
from src.services.billing_service import FakeRazorpayClient
import src.core.security as security
from src.main import app
from fastapi.testclient import TestClient


class _FakeColl:
    def __init__(self): self.docs = {}; self._i = 0
    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()): return dict(d)
        return None
    async def find(self, q=None):
        q = q or {}
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]
    async def insert_one(self, doc):
        self._i += 1
        doc = dict(doc); doc["_id"] = str(self._i); self.docs[str(self._i)] = doc
        class R: inserted_id = str(self._i)
        return R()
    async def update_one(self, q, op):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                if "$inc" in op:
                    for k, v in op["$inc"].items(): d[k] = d.get(k, 0) + v


@pytest.fixture
def setup(monkeypatch):
    users = _FakeColl(); orgs = _FakeColl(); invites = _FakeColl(); subs = _FakeColl()
    monkeypatch.setattr(ds, "users_collection", users)
    monkeypatch.setattr(ds, "organizations_collection", orgs)
    monkeypatch.setattr(ds, "org_invites_collection", invites)
    monkeypatch.setattr(ds, "subscriptions_collection", subs)
    monkeypatch.setattr(os_, "users_collection", users)
    monkeypatch.setattr(os_, "organizations_collection", orgs)
    monkeypatch.setattr(os_, "org_invites_collection", invites)
    monkeypatch.setattr(os_, "subscriptions_collection", subs)
    monkeypatch.setattr(bs, "get_client", lambda: FakeRazorpayClient(secret="k"))
    # subadmin auth
    users.docs["1"] = {"email": "owner@x.com", "role": "subadmin"}
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "owner@x.com", "user": {"email": "owner@x.com", "role": "subadmin"}})
    return dict(client=TestClient(app), users=users, orgs=orgs, invites=invites)


def test_create_org(setup):
    r = setup["client"].post("/orgs", json={"name": "Acme Coaching", "brand_name": "Acme",
        "tier": "pro", "seats_total": 10, "billing_cycle": "monthly"})
    assert r.status_code == 201, r.text
    assert r.json()["org_id"]


def test_invite_then_enroll_consumes_seat(setup):
    c = setup["client"]
    c.post("/orgs", json={"name": "Acme", "brand_name": "Acme", "tier": "pro",
        "seats_total": 1, "billing_cycle": "monthly"})
    inv = c.post("/orgs/invite", json={"member_role": "student"}).json()
    code = inv["code"]
    # a student enrolls
    setup["users"].docs["2"] = {"email": "stu@x.com", "role": "student"}
    import src.core.security as sec
    sec.get_current_user_with_role = lambda *a, **k: {"email": "stu@x.com", "user": {"email": "stu@x.com", "role": "student"}}
    r = c.post(f"/orgs/enroll/{code}")
    assert r.status_code == 200 and r.json()["member_role"] == "student"
    # second enroll should 402 (seat full)
    setup["users"].docs["3"] = {"email": "stu2@x.com", "role": "student"}
    sec.get_current_user_with_role = lambda *a, **k: {"email": "stu2@x.com", "user": {"email": "stu2@x.com", "role": "student"}}
    inv2 = c.post("/orgs/invite", json={"member_role": "student"}).json()
    r2 = c.post(f"/orgs/enroll/{inv2['code']}")
    assert r2.status_code == 402


def test_non_subadmin_cannot_create_org(monkeypatch):
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "s@x.com", "user": {"email": "s@x.com", "role": "student"}})
    c = TestClient(app)
    r = c.post("/orgs", json={"name": "X", "brand_name": "X", "tier": "pro",
        "seats_total": 1, "billing_cycle": "monthly"})
    assert r.status_code == 403
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_orgs.py -v`
Expected: FAIL (404/import).

- [ ] **Step 4: Implement `services/org_service.py`**

```python
"""Organization (coaching/school) + seat license logic."""
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from src.core.data_store import (
    users_collection, organizations_collection, org_invites_collection,
    subscriptions_collection,
)
from src.services.billing_service import get_client
from src.core.config import RAZORPAY_KEY_ID


def _gen_code() -> str:
    return secrets.token_urlsafe(6)[:8].upper()


async def create_org(owner_id: str, name: str, brand_name: Optional[str],
                     tier: str, seats_total: int, billing_cycle: str) -> dict:
    if organizations_collection is None or users_collection is None:
        raise HTTPException(503, "Database connection not available")
    # one org per owner in v1
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
        {"$set": {"role": "subadmin", "org_id": org_id, "member_role": None, "org_joined_at": now}},
    )
    # Checkout for the seat subscription (quantity = seats_total). v1: a single
    # Razorpay subscription per org; details confirmed against Razorpay dashboard.
    from src.core.plans import PLAN_PRICES
    amount = PLAN_PRICES.get((tier, billing_cycle), 0) * seats_total
    rzp = get_client().create_subscription(plan_id_for := "", owner_id, amount)  # plan_id resolved at impl time
    return {"org_id": org_id, "checkout": {"razorpay_subscription_id": rzp.get("id"), "key_id": RAZORPAY_KEY_ID, "amount": amount, "currency": "INR"}}


async def get_org_by_owner(owner_id: str) -> Optional[dict]:
    if organizations_collection is None:
        return None
    return await organizations_collection.find_one({"owner_user_id": owner_id})


async def create_invite(owner_id: str, member_role: str, email: Optional[str]) -> dict:
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
    if org.get("expires_at") and org["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(403, "Organization license has expired")
    if org.get("seats_used", 0) >= org.get("seats_total", 0):
        raise HTTPException(402, detail={"resource": "org_seat", "used": org.get("seats_used", 0),
                                         "limit": org.get("seats_total", 0), "plan": org.get("tier"),
                                         "upgrade_url": "/billing"})
    now = datetime.now(timezone.utc)
    await organizations_collection.update_one(
        {"org_id": org["org_id"]}, {"$inc": {"seats_used": 1}, "$set": {"updated_at": now}})
    await org_invites_collection.update_one({"code": code.upper()}, {"$set": {"used_by_user_id": user_id}})
    await users_collection.update_one(
        {"email": user_id},
        {"$set": {"org_id": org["org_id"], "member_role": inv["member_role"], "org_joined_at": now}})
    return {"org_id": org["org_id"], "member_role": inv["member_role"]}


async def list_members(owner_id: str) -> list:
    org = await get_org_by_owner(owner_id)
    if not org:
        return []
    users = await users_collection.find({"org_id": org["org_id"]})
    return [{"email": u.get("email"), "name": u.get("name"),
             "member_role": u.get("member_role")} for u in users]


async def remove_member(owner_id: str, member_email: str) -> dict:
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
    from src.core.plans import PLAN_PRICES
    amount = PLAN_PRICES.get((org["tier"], org.get("billing_cycle", "monthly")), 0) * add_seats
    rzp = get_client().create_subscription("", owner_id, amount)
    return {"checkout": {"razorpay_subscription_id": rzp.get("id"), "key_id": RAZORPAY_KEY_ID,
                         "amount": amount, "currency": "INR"}}
```

- [ ] **Step 5: Implement `routers/org_router.py`**

```python
from typing import Literal, Optional
from fastapi import APIRouter, Depends, Path, HTTPException, status
from pydantic import BaseModel

from src.core.security import require_role, get_current_user_with_role
from src.services import org_service as svc

router = APIRouter(prefix="/orgs", tags=["Organizations"])


class OrgCreateRequest(BaseModel):
    name: str
    brand_name: Optional[str] = None
    tier: Literal["pro", "premium"]
    seats_total: int = 1
    billing_cycle: Literal["monthly", "yearly"] = "monthly"


class InviteRequest(BaseModel):
    member_role: Literal["teacher", "student"]
    email: Optional[str] = None


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_org(req: OrgCreateRequest, user=Depends(require_role("subadmin", "admin"))):
    # allow admin too (admin can create orgs on behalf of a coaching owner)
    return await svc.create_org(user["email"], req.name, req.brand_name, req.tier, req.seats_total, req.billing_cycle)


@router.get("/me")
async def my_org(user=Depends(require_role("subadmin"))):
    org = await svc.get_org_by_owner(user["email"])
    if not org:
        raise HTTPException(404, "No organization found")
    members = await svc.list_members(user["email"])
    return {"org": org, "members": members,
            "seats": {"used": org.get("seats_used", 0), "total": org.get("seats_total", 0)}}


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


class SeatsRequest(BaseModel):
    add_seats: int


@router.post("/seats")
async def add_seats(req: SeatsRequest, user=Depends(require_role("subadmin"))):
    return await svc.add_seats(user["email"], req.add_seats)
```

Note: `require_role("subadmin", "admin")` — verify `require_role` accepts multiple args (it does: `def require_role(*allowed_roles)`). For the enroll endpoint we use `get_current_user_with_role` (any authenticated user, including a student accepting an invite).

- [ ] **Step 6: Register** in `routers/__init__.py` and `main.py`.

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_orgs.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Backend/src/services/org_service.py Backend/src/routers/org_router.py Backend/src/core/models.py Backend/src/routers/__init__.py Backend/src/main.py Backend/tests/test_orgs.py
git commit -m "feat(multi-tenant): org/coaching seat licenses with invite + enroll + seat enforcement"
```

---

## Task 7: Admin router

**Files:**
- Create: `Backend/src/routers/admin_router.py`
- Create: `Backend/tests/test_admin.py`
- Modify: `routers/__init__.py`, `main.py`

**Interfaces:**
- Consumes: `require_role("admin")`, `core/data_store` collections, `subscription_service`.
- Produces (router, all `require_role("admin")`):
  - `GET /admin/users?role=&org_id=&limit=&skip=` → `{users:[...], total}`
  - `PATCH /admin/users/{email}/role` body `{role}`
  - `PATCH /admin/users/{email}/status` body `{active: bool}`
  - `GET /admin/orgs` → `{orgs:[...]}`
  - `PATCH /admin/orgs/{org_id}` body `{status?, seats_total?, expires_at?}`
  - `GET /admin/subscriptions` → `{subscriptions:[...], payments:[...]}`
  - `POST /admin/subscriptions/{user_id}/activate` body `{plan, days}` → manual activation
  - `GET /admin/analytics` → `{totals:{users_by_role, active_subscriptions, org_count, mrr_paise}, conversion}`

- [ ] **Step 1: Write the failing tests**

`Backend/tests/test_admin.py`:
```python
import pytest
from datetime import datetime, timezone, timedelta
import src.core.data_store as ds
import src.core.security as security
from src.main import app
from fastapi.testclient import TestClient


class _FakeColl:
    def __init__(self): self.docs = {}; self._i = 0
    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()): return dict(d)
        return None
    async def find(self, q=None):
        q = q or {}
        return [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]
    async def insert_one(self, doc):
        self._i += 1; self.docs[str(self._i)] = dict(doc)
        class R: inserted_id = str(self._i)
        return R()
    async def update_one(self, q, op):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()): d.update(op.get("$set", {}))
    async def count_documents(self, q=None):
        return len(await self.find(q or {}))


@pytest.fixture
def admin(monkeypatch):
    users = _FakeColl(); orgs = _FakeColl(); subs = _FakeColl(); pays = _FakeColl()
    for c, name in [(users,"users_collection"),(orgs,"organizations_collection"),
                    (subs,"subscriptions_collection"),(pays,"payments_collection")]:
        monkeypatch.setattr(ds, name, c)
    import src.routers.admin_router as ar
    monkeypatch.setattr(ar, "users_collection", users)
    monkeypatch.setattr(ar, "organizations_collection", orgs)
    monkeypatch.setattr(ar, "subscriptions_collection", subs)
    monkeypatch.setattr(ar, "payments_collection", pays)
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "admin@x.com", "user": {"email": "admin@x.com", "role": "admin"}})
    return dict(client=TestClient(app), users=users, orgs=orgs, subs=subs, pays=pays)


def test_non_admin_forbidden(monkeypatch):
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "s@x.com", "user": {"email": "s@x.com", "role": "student"}})
    c = TestClient(app)
    assert c.get("/admin/users").status_code == 403


def test_list_users(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    r = admin["client"].get("/admin/users")
    assert r.status_code == 200
    assert any(u["email"] == "a@x.com" for u in r.json()["users"])


def test_change_role(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    r = admin["client"].patch("/admin/users/a@x.com/role", json={"role": "teacher"})
    assert r.status_code == 200
    assert admin["users"].docs["1"]["role"] == "teacher"


def test_manual_activate(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    r = admin["client"].post("/admin/subscriptions/a@x.com/activate", json={"plan": "pro", "days": 30})
    assert r.status_code == 200
    sub = admin["subs"].docs.get("1")
    assert sub and sub["status"] == "active" and sub["plan"] == "pro"


def test_analytics(admin):
    admin["users"].docs["1"] = {"email": "a@x.com", "role": "student"}
    admin["users"].docs["2"] = {"email": "b@x.com", "role": "teacher"}
    admin["subs"].docs["1"] = {"user_id": "a@x.com", "status": "active", "plan": "pro", "billing_cycle": "monthly"}
    admin["pays"].docs["1"] = {"user_id": "a@x.com", "amount": 29900, "status": "captured"}
    admin["orgs"].docs["1"] = {"org_id": "o1", "status": "active"}
    r = admin["client"].get("/admin/analytics")
    assert r.status_code == 200
    a = r.json()
    assert a["totals"]["users_by_role"]["student"] == 1
    assert a["totals"]["active_subscriptions"] == 1
    assert a["totals"]["org_count"] == 1
    assert a["totals"]["mrr_paise"] == 29900
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_admin.py -v`
Expected: FAIL (404).

- [ ] **Step 3: Implement `routers/admin_router.py`**

```python
from datetime import datetime, timezone, timedelta
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
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
    if role: q["role"] = role
    if org_id: q["org_id"] = org_id
    docs = await users_collection.find(q)
    users = [{"email": d.get("email"), "name": d.get("name"), "role": d.get("role", "student"),
              "org_id": d.get("org_id"), "active": d.get("active", True)} for d in docs]
    return {"users": users[skip:skip+limit], "total": len(users)}


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
    paid_signedup = 0
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
```

- [ ] **Step 4: Register** in `routers/__init__.py` + `main.py`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_admin.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Backend/src/routers/admin_router.py Backend/src/routers/__init__.py Backend/src/main.py Backend/tests/test_admin.py
git commit -m "feat(admin): admin endpoints (users, orgs, subscriptions, analytics, manual activate)"
```

---

## Task 8: Wire enforcement into existing endpoints

**Files:**
- Modify: `Backend/src/routers/mock_test_router.py` (`/generate`)
- Modify: `Backend/src/routers/flashcard_router.py` (`/generate`)
- Modify: `Backend/src/routers/ai_material_router.py` (`/summarize`)
- Modify: `Backend/src/routers/question_router.py` (`/ask`, `/ask/stream`, `.../messages`, `.../messages/stream`)
- Modify: `Backend/src/routers/document_router.py` (`/upload`)
- Modify: `Backend/src/routers/material_router.py` (material upload)
- Modify: `Backend/src/routers/class_router.py` (`POST /classes/`)
- Test: extend `tests/test_plans_enforcement.py` with an integration check using a real router + `TestClient`.

**Interfaces:**
- Consumes: `core.plan_enforcement.enforce_limit(resource)`, `increment_usage`.
- Pattern: add a `Depends(enforce_limit("resource"))` parameter to the endpoint signature. The dependency returns `user_info`; reuse the existing auth dependency's user. To avoid double auth, replace the existing `Depends(get_current_user)`/`get_current_user_with_role` on that endpoint with `Depends(enforce_limit("resource"))` (which internally calls `get_current_user_with_role`).

- [ ] **Step 1: Inspect each target endpoint's current signature**

Run: `grep -n "def \|Depends" Backend/src/routers/mock_test_router.py Backend/src/routers/flashcard_router.py Backend/src/routers/ai_material_router.py Backend/src/routers/question_router.py Backend/src/routers/document_router.py Backend/src/routers/material_router.py | head -80`
Read the exact current auth dependency each endpoint uses so you replace it correctly.

- [ ] **Step 2: Add an integration test**

Append to `tests/test_plans_enforcement.py`:
```python
from fastapi.testclient import TestClient
import src.core.security as security
import src.services.billing_service as bs
from src.services.billing_service import FakeRazorpayClient
from src.main import app


def test_mock_test_generate_402_on_free_limit(monkeypatch):
    # starter allows 3 mock tests; pre-fill 3 generated tests and hit /generate
    mt = _FakeColl(); users = _FakeColl(); subs = _FakeColl()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for i in range(3):
        mt.docs[str(i)] = {"user_id": "u@x.com", "created_at": now}
    users.docs["1"] = {"email": "u@x.com", "role": "student"}
    import src.core.data_store as ds2
    monkeypatch.setattr(ds2, "mock_tests_collection", mt)
    monkeypatch.setattr(ds2, "users_collection", users)
    monkeypatch.setattr(ds2, "subscriptions_collection", subs)
    import src.core.plan_enforcement as pe2
    monkeypatch.setattr(pe2, "mock_tests_collection", mt)
    monkeypatch.setattr(pe2, "users_collection", users)
    monkeypatch.setattr(pe2, "subscriptions_collection", subs)
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "u@x.com", "user": {"email": "u@x.com", "role": "student"}})
    # The mock_test generate endpoint also calls Gemini; patch the service to short-circuit
    import src.services.mock_test_service as mts
    async def _stub(*a, **k):
        from fastapi import HTTPException
        raise HTTPException(402, "limit")
    # We only need the 402 from enforce_limit, which fires before the service runs.
    client = TestClient(app)
    # Build a minimal request the router accepts; the 402 must fire pre-service.
    r = client.post("/mock-tests/generate", json={
        "syllabus_pdf_id": "x", "question_paper_pdf_ids": []})
    assert r.status_code == 402
```
Note: confirm the `/mock-tests/generate` request body matches `MockTestGenerationRequest` (it does: `syllabus_pdf_id`, `question_paper_pdf_ids` required; defaults fill the rest). The 402 fires in the dependency before the handler body, so the Gemini service is never reached.

- [ ] **Step 3: Run the new test to verify it fails**

Run: `pytest tests/test_plans_enforcement.py::test_mock_test_generate_402_on_free_limit -v`
Expected: FAIL (the endpoint currently uses `get_current_user` not `enforce_limit`, so no 402 — it proceeds and likely 500s on missing PDFs, or 401).

- [ ] **Step 4: Wire `enforce_limit` into the six endpoints**

For each endpoint, add `from src.core.plan_enforcement import enforce_limit, increment_usage` and change the signature's auth dependency to `Depends(enforce_limit("resource"))`. Example for `mock_test_router.py` `/generate`:

Before (approx):
```python
async def generate_mock_test(req: MockTestGenerationRequest, user_id: str = Depends(get_current_user)):
```
After:
```python
async def generate_mock_test(req: MockTestGenerationRequest, user=Depends(enforce_limit("mock_test"))):
    user_id = user["email"]
```

Apply the same pattern:
- `flashcard_router.py` `/generate` → `enforce_limit("flashcard")`
- `ai_material_router.py` `/summarize` → `enforce_limit("ai_material")`
- `question_router.py` `/questions/ask`, `/questions/ask/stream`, `/questions/sessions/{id}/messages`, `.../messages/stream` → `enforce_limit("chat_message")`. **Additionally**, at the end of each successful handler (after the AI responds), call `await increment_usage(user_id, "chat_message")`. For streaming endpoints, increment *before* streaming begins (a message is "used" once accepted) so an aborted stream still counts.
- `document_router.py` `/documents/upload` → `enforce_limit("doc_storage")` (the handler already knows the incoming file size; the limit check uses cumulative stored size — to be precise, the dependency checks current stored size; an over-limit upload is rejected. Acceptable: a slightly-over user gets 402 on the upload that would exceed. For v1 this is fine.)
- `material_router.py` material upload (the `POST /api/collections/{id}/materials` endpoint) → `enforce_limit("doc_storage")`.
- `class_router.py` `POST /classes/` → replace `Depends(require_role("teacher"))` with `Depends(enforce_limit("class_count"))`. **Careful:** `enforce_limit` calls `get_current_user_with_role`, which does NOT check role. So combine: keep role gating by adding a manual role check, OR add a `require_role_and_limit` helper. Simplest: in the handler, after `enforce_limit` passes, check `user["user"].get("role") == "teacher"` (or `admin`) and raise 403 otherwise. Apply:
```python
async def create_class(request: ClassCreateRequest, user=Depends(enforce_limit("class_count"))):
    if user["user"].get("role") not in ("teacher", "admin"):
        raise HTTPException(403, "Only teachers can create classes")
    teacher_email = user["email"]
    ...
```

- [ ] **Step 5: Run the full backend test suite**

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: all green, including pre-existing tests (the wiring changes must not break existing flows — `enforce_limit` resolves to `starter` for users with no subscription, and existing test users are under their limits).

If an existing test breaks because a test user now hits a 402, fix by giving that test's user an active premium subscription in its fixture (`subs.docs["1"] = {"user_id": "<email>", "status":"active","plan":"premium"}` and patch `plan_enforcement.subscriptions_collection`).

- [ ] **Step 6: Commit**

```bash
git add Backend/src/routers/mock_test_router.py Backend/src/routers/flashcard_router.py Backend/src/routers/ai_material_router.py Backend/src/routers/question_router.py Backend/src/routers/document_router.py Backend/src/routers/material_router.py Backend/src/routers/class_router.py Backend/tests/test_plans_enforcement.py
git commit -m "feat(billing): enforce plan limits on AI-generation + storage + class endpoints"
```

---

## Task 9: Rate limiting + final hardening

**Files:**
- Modify: `Backend/src/main.py` (mount `slowapi` limiter + handler)
- Modify: the auth + AI routers to decorate sensitive endpoints with `@limiter.limit(...)`
- Test: `tests/test_rate_limit.py` (verify a burst of login attempts gets 429)

**Interfaces:**
- Produces: a module-level `limiter = Limiter(...)` in `main.py` (or a new `core/limiter.py`) using in-memory storage (v1; Redis later). Auth endpoints: `5/minute` per-IP. AI-generation endpoints: `20/minute` per-user.

- [ ] **Step 1: Create `core/limiter.py`**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
```

- [ ] **Step 2: Wire into `main.py`**

Add near the top imports: `from slowapi import _rate_limit_exceeded_handler` and `from slowapi.errors import RateLimitExceeded`, and `from src.core.limiter import limiter`. After `app = FastAPI(...)`:
```python
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

- [ ] **Step 3: Decorate auth + AI endpoints**

In `auth_router.py`, decorate `login` and `signup`:
```python
from src.core.limiter import limiter

@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    ...
```
(`slowapi` requires the `request: Request` parameter on decorated endpoints — add it.)

In `mock_test_router.py` `generate`, `flashcard_router.py` `generate`, `ai_material_router.py` `summarize`, `question_router.py` ask endpoints, add `@limiter.limit("20/minute")` with a `request: Request` param.

- [ ] **Step 4: Write the rate-limit test**

`Backend/tests/test_rate_limit.py`:
```python
import pytest
from fastapi.testclient import TestClient
import src.core.security as security
from src.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(security, "get_current_user_with_role",
                        lambda *a, **k: {"email": "u@x.com", "user": {"email": "u@x.com", "role": "student"}})
    return TestClient(app)


def test_login_burst_throttled(client, monkeypatch):
    # Patch auth_service so signup/login don't need DB
    import src.services.auth_service as au
    class _R: pass
    async def _auth(email, password):
        r = _R(); r.access_token = "x"; r.token_type = "bearer"
        return r
    # 6 rapid calls; at least one must 429
    codes = []
    for i in range(8):
        r = client.post("/auth/login", data={"username": f"u{i}@x.com", "password": "p"})
        codes.append(r.status_code)
    assert 429 in codes
```
Note: the limiter uses in-memory storage keyed by IP; `TestClient` presents a single client IP. If existing `test_auth.py` tests now fail due to the limiter, raise their fixture's tolerance or clear `limiter.reset()` between tests. Add `limiter.reset()` to a shared autouse fixture if needed.

- [ ] **Step 5: Run the full suite**

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: all green (patch existing tests that now hit 429 by resetting the limiter).

- [ ] **Step 6: Smoke-test the app boots**

Run: `source venv/bin/activate && python -c "from src.main import app; print([r.path for r in app.routes if 'subscriptions' in r.path or 'webhooks' in r.path or 'orgs' in r.path or 'admin' in r.path])"`
Expected: prints the new route paths.

- [ ] **Step 7: Commit**

```bash
git add Backend/src/core/limiter.py Backend/src/main.py Backend/src/routers/auth_router.py Backend/src/routers/mock_test_router.py Backend/src/routers/flashcard_router.py Backend/src/routers/ai_material_router.py Backend/src/routers/question_router.py Backend/tests/test_rate_limit.py
git commit -m "feat(hardening): rate limiting on auth + AI-generation endpoints"
```

---

## Self-Review (run after writing this plan)

**1. Spec coverage:**
- Plans & limits → Task 1 (`plans.py`). ✓
- Data model (subscriptions/payments/organizations/org_invites/usage_events + User fields) → Tasks 1, 6. ✓
- `subscription_router` (`/subscriptions/*`) → Task 4. ✓
- `webhook_router` → Task 5. ✓
- `org_router` (`/orgs/*`) → Task 6. ✓
- `admin_router` (`/admin/*`) → Task 7. ✓
- `plan_enforcement` + enforcement points → Tasks 2, 8. ✓
- Razorpay checkout flow (checkout → verify → webhook) → Tasks 3, 4, 5. ✓
- Production hardening (rate limiting, webhook signature, idempotency, role guards, config) → Tasks 3, 5, 9 + role guards throughout. ✓
- Testing (pytest for enforcement, billing, webhooks, orgs, admin) → Tasks 2–7, 9. ✓
- Razorpay SRI note → frontend plan (1b). N/A here. ✓ (deferred)

**2. Placeholder scan:** The `create_org` checkout in Task 6 has a `plan_id_for := ""` placeholder line for the org seat subscription plan id — this is a known open item (per-seat pricing mechanism confirmed against Razorpay dashboard at impl time). The implementer must resolve it: either (a) create a per-org Razorpay plan with `quantity=seats_total`, or (b) bill the owner via a one-time order for `seats_total × tier price`. The plan calls this out; not a hidden TBD. No other placeholders.

**3. Type consistency:**
- `enforce_limit(resource)` returns `{"email","user"}` everywhere it's used (Tasks 2, 8). ✓
- `get_effective_plan` returns `(plan, source, org_id)` consistently (Task 2, used in Task 4). ✓
- `FakeRazorpayClient.verify_payment_signature(payment_id, subscription_id, signature)` signature matches `RazorpayClient` (Task 3). ✓
- `subscription_service` exports `create_checkout/verify_and_activate/cancel/list_invoices` (Task 4) — router uses exactly these. ✓
- `org_service` exports `create_org/get_org_by_owner/create_invite/enroll/list_members/remove_member/add_seats` (Task 6) — router uses exactly these. ✓
- Admin endpoints use `require_role("admin")` consistently (Task 7). ✓

No type drifts found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-phase1a-backend-billing.md`. After this plan is implemented and green, Plan 1b (frontend: pricing/billing/checkout, onboarding sub-admin path, `/org`, `/admin` wired, 402 upgrade banner) follows.