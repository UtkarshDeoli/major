import pytest
from src.core.migrations import run_coaching_p1_migration


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._docs):
            raise StopAsyncIteration
        d = dict(self._docs[self._i])
        self._i += 1
        return d


class _FakeColl:
    def __init__(self, docs):
        self.docs = docs

    def find(self, q=None):
        return _AsyncCursor(self.docs.values())

    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                return dict(d)
        return None

    async def update_one(self, q, op, upsert=False):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                d.update(op.get("$set", {}))
                return


@pytest.mark.asyncio
async def test_backfill_adds_teacher_ids_subject_ids_org_id():
    classes = _FakeColl({
        "1": {"_id": "1", "teacher_id": "t@x.com", "enroll_code": "X1"},
    })
    users = _FakeColl({
        "t": {"email": "t@x.com", "org_id": "org-9"},
    })
    res = await run_coaching_p1_migration(classes, users)
    assert res["classes_backfilled"] == 1
    assert classes.docs["1"]["teacher_ids"] == ["t@x.com"]
    assert classes.docs["1"]["subject_ids"] == []
    assert classes.docs["1"]["org_id"] == "org-9"


@pytest.mark.asyncio
async def test_backfill_is_idempotent():
    classes = _FakeColl({
        "1": {"_id": "1", "teacher_id": "t@x.com", "teacher_ids": ["t@x.com"],
              "subject_ids": [], "org_id": "org-9"},
    })
    res = await run_coaching_p1_migration(classes, _FakeColl({}))
    assert res["classes_backfilled"] == 0
