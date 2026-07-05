import asyncio

import pytest

# Ensure an event loop exists on the main thread before pytest imports modules
# that perform async checks at import time (e.g. MongoDB connection probes).
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


@pytest.fixture(scope="session")
def event_loop():
    """Provide the module-scoped event loop for async tests."""
    yield _loop
    _loop.close()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear slowapi's in-memory counters before each test so rate-limit state
    never leaks between tests (TestClient presents a single client IP)."""
    from src.core.limiter import limiter
    limiter.reset()
    yield
    limiter.reset()
