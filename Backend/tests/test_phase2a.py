"""Tests for Phase 2a AI tutoring features."""
from datetime import datetime, timezone, timedelta

import pytest

import src.core.data_store as ds
from src.services.student_mastery_service import (
    update_mastery_from_submission,
    get_mastery_scores,
    get_weak_topics,
    recommended_difficulty,
    build_adaptive_bias,
)


class _FakeCursor:
    """Async cursor-like wrapper over a list for Motor find().to_list()."""

    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        return self._docs[:length] if length is not None else list(self._docs)


class _FakeColl:
    """Minimal async Mongo collection mock backed by an in-memory dict."""

    def __init__(self):
        self.docs = {}
        self._i = 0

    async def find_one(self, q):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                return dict(d)
        return None

    def find(self, q=None):
        q = q or {}
        results = [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in q.items())]
        return _FakeCursor(results)

    async def insert_one(self, doc):
        self._i += 1
        doc = dict(doc)
        doc["_id"] = str(self._i)
        self.docs[str(self._i)] = doc

        class R:
            inserted_id = str(self._i)
        return R()

    async def update_one(self, q, op):
        for d in self.docs.values():
            if all(d.get(k) == v for k, v in q.items()):
                if "$set" in op:
                    d.update(op["$set"])


@pytest.fixture
def mastery_isolated(monkeypatch):
    mastery = _FakeColl()
    monkeypatch.setattr(ds, "student_mastery_collection", mastery)
    return mastery


async def test_update_mastery_creates_new_entry(mastery_isolated):
    await update_mastery_from_submission(
        "student@x.com",
        [
            {"topic": "Algebra", "difficulty": "medium", "marks_awarded": 2, "max_marks": 5},
        ],
    )
    scores = await get_mastery_scores("student@x.com")
    assert "Algebra" in scores


async def test_update_mastery_moves_score_based_on_accuracy(mastery_isolated):
    # Wrong answer on hard topic should decrease mastery
    await update_mastery_from_submission(
        "student@x.com",
        [
            {"topic": "Algebra", "difficulty": "hard", "marks_awarded": 0, "max_marks": 5},
        ],
    )
    score_after_wrong = (await get_mastery_scores("student@x.com"))["Algebra"]
    assert score_after_wrong < 50

    # Correct answer on easy topic should increase mastery
    await update_mastery_from_submission(
        "student@x.com",
        [
            {"topic": "Algebra", "difficulty": "easy", "marks_awarded": 5, "max_marks": 5},
        ],
    )
    score_after_correct = (await get_mastery_scores("student@x.com"))["Algebra"]
    assert score_after_correct > score_after_wrong


async def test_weak_topics_returns_lowest_scoring(mastery_isolated):
    await update_mastery_from_submission(
        "student@x.com",
        [
            {"topic": "Algebra", "difficulty": "easy", "marks_awarded": 5, "max_marks": 5},
            {"topic": "Calculus", "difficulty": "hard", "marks_awarded": 0, "max_marks": 5},
        ],
    )
    scores = await get_mastery_scores("student@x.com")
    weak = await get_weak_topics("student@x.com", threshold=50)
    # Calculus should be weaker than Algebra and appear earlier in the list
    assert "Calculus" in weak
    assert scores["Calculus"] < scores["Algebra"]


async def test_recommended_difficulty_mapping():
    assert recommended_difficulty(80) == "hard"
    assert recommended_difficulty(60) == "medium"
    assert recommended_difficulty(30) == "easy"
    assert recommended_difficulty(None) == "mixed"


async def test_build_adaptive_bias_ignores_non_adaptive_request(mastery_isolated):
    bias = await build_adaptive_bias("student@x.com", "medium")
    assert bias["difficulty"] == "medium"
    assert bias["focus_topics"] == []


async def test_build_adaptive_bias_uses_mastery(mastery_isolated):
    await update_mastery_from_submission(
        "student@x.com",
        [
            {"topic": "Algebra", "difficulty": "hard", "marks_awarded": 0, "max_marks": 5},
        ],
    )
    bias = await build_adaptive_bias("student@x.com", "adaptive")
    assert bias["difficulty"] == "easy"
    assert "Algebra" in bias["focus_topics"]
    assert "Algebra" in bias["weak_topics"]


async def test_answer_feedback_model_accepts_rubric():
    from src.core.models import AnswerFeedback

    fb = AnswerFeedback(
        question_id="1",
        question="Q",
        user_answer="A",
        feedback="Good",
        marks_awarded=3,
        max_marks=5,
        rubric_scores={"understanding": 2, "accuracy": 1},
        rubric_max={"understanding": 2, "accuracy": 2},
    )
    assert fb.rubric_scores["understanding"] == 2
