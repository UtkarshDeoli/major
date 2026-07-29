"""One-off data migrations for the coaching-platform reshape.

Each function is idempotent: it only sets fields that are missing, so it is
safe to run repeatedly. Run via `python scripts/run_migration.py`.
"""
from typing import Optional


async def run_coaching_p1_migration(classes_coll, users_coll=None) -> dict:
    """Backfill existing classes with Phase 1 fields.

    - teacher_ids: [teacher_id] if missing
    - subject_ids: [] if missing
    - org_id: derived from the teacher's user doc if missing
    """
    cursor = classes_coll.find({})
    count = 0
    async for cls in cursor:
        update = {}
        if not cls.get("teacher_ids"):
            tid = cls.get("teacher_id")
            update["teacher_ids"] = [tid] if tid else []
        if "subject_ids" not in cls:
            update["subject_ids"] = []
        if not cls.get("org_id") and users_coll is not None:
            teacher = await users_coll.find_one({"email": cls.get("teacher_id")})
            update["org_id"] = teacher.get("org_id") if teacher else None
        if update:
            await classes_coll.update_one({"_id": cls["_id"]}, {"$set": update})
            count += 1
    return {"classes_backfilled": count}
