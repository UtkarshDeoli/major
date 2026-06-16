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
