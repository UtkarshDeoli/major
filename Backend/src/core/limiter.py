"""Slowapi rate limiter (in-memory for v1; Redis for production later)."""
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.security import _decode_token


def _rate_limit_key(request: Request) -> str:
    """Per-user rate limit key when a Bearer token is present, else IP address."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            return _decode_token(auth[7:])
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key, storage_uri="memory://")


# Common limit strings. The key function above means these are per-user by default.
GENERATION_LIMIT = "10/hour"  # mock tests, flashcards, ai materials, analysis
CHAT_LIMIT = "60/hour"        # chat messages (beyond plan enforcement)
UPLOAD_LIMIT = "30/hour"      # PDF/document uploads