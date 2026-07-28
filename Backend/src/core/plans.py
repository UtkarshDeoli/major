"""Subscription plan tiers, limits, and prices.

Single source of truth for what each plan allows. The frontend reads these
via GET /subscriptions/plans so the UI never hardcodes limits that drift
from the backend.
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
    "doc_storage", "class_count", "study_plan",
)

_MB = 1024 * 1024
_GB = 1024 * 1024 * 1024

PLAN_LIMITS: Dict[str, Dict[str, float]] = {
    STARTER: {
        "mock_test": 3, "flashcard": 50, "ai_material": 5, "chat_message": 100,
        "doc_storage": 50 * _MB, "class_count": 1, "study_plan": 5,
    },
    PRO: {
        "mock_test": 50, "flashcard": 500, "ai_material": 50, "chat_message": 1000,
        "doc_storage": 1 * _GB, "class_count": 10, "study_plan": 50,
    },
    PREMIUM: {
        "mock_test": math.inf, "flashcard": math.inf, "ai_material": math.inf,
        "chat_message": math.inf, "doc_storage": 10 * _GB, "class_count": math.inf,
        "study_plan": math.inf,
    },
}

# INR paise — keyed by (plan, billing_cycle)
PLAN_PRICES: Dict[tuple, int] = {
    (PRO, "monthly"): 29900, (PRO, "yearly"): 299000,
    (PREMIUM, "monthly"): 59900, (PREMIUM, "yearly"): 599000,
}


def limit_for(plan: str, resource: str) -> float:
    """Return the limit for a resource on a plan. Unknown plan -> starter."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS[STARTER]).get(resource, 0)