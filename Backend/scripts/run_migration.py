"""Run coaching-platform migrations against the real database.

Usage:  source venv/bin/activate && python scripts/run_migration.py
"""
import asyncio

from src.core.data_store import classes_collection, users_collection
from src.core.migrations import run_coaching_p1_migration


async def main():
    if classes_collection is None:
        raise SystemExit("MongoDB not connected; cannot run migration.")
    res = await run_coaching_p1_migration(classes_collection, users_collection)
    print("Migration result:", res)


if __name__ == "__main__":
    asyncio.run(main())
