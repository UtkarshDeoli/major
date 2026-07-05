"""Slowapi rate limiter (in-memory for v1; Redis for production later)."""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")