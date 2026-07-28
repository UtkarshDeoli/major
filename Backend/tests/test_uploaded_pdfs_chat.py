"""Tests for uploaded-PDF chat integration: subject grouping, RAG indexing, and quick mock tests."""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from httpx import ASGITransport
import pytest
from bson import ObjectId

from src.core.data_store import (
    users_collection,
    pdfs_collection,
    document_chunks_collection,
)
from src.main import app


pytestmark = pytest.mark.skipif(
    users_collection is None,
    reason="MongoDB connection unavailable",
)

_loop = asyncio.get_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


async def _ensure_user(client: httpx.AsyncClient, email: str, password: str):
    from src.services.auth_service import get_password_hash, get_user_by_email

    existing = await get_user_by_email(email)
    if existing:
        if "password_hash" not in existing:
            await users_collection.update_one(
                {"email": email},
                {"$set": {"password_hash": get_password_hash(password)}},
            )
        return

    signup_resp = await client.post(
        "/auth/signup",
        json={"email": email, "password": password},
    )
    assert signup_resp.status_code in (200, 201), signup_resp.text


async def _token_for(client: httpx.AsyncClient, email: str, password: str):
    await _ensure_user(client, email, password)
    login_resp = await client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    assert login_resp.status_code == 200, login_resp.text
    return login_resp.json()["access_token"]


def _auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def _unique(prefix: str) -> str:
    return f"{prefix}.{uuid.uuid4().hex[:8]}@example.com"


def _minimal_pdf_bytes() -> bytes:
    """Return a tiny valid PDF with extractable text."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 68 >>
stream
BT
/F1 12 Tf
100 700 Td
(Photoelectric effect and electron emission from metals.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000303 00000 n
0000000421 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
499
%%EOF"""


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


def _fake_practice_test_json(num_mcq: int = 2, num_text: int = 1) -> str:
    import json

    questions = []
    for i in range(num_mcq):
        questions.append(
            {
                "id": str(i + 1),
                "type": "mcq",
                "question": f"MCQ question {i + 1}",
                "options": ["A) Opt 1", "B) Opt 2", "C) Opt 3", "D) Opt 4"],
                "correctAnswer": "A) Opt 1",
                "marks": 2,
                "unit": "Unit 1",
                "topic": "Photoelectric effect",
                "difficulty": "medium",
            }
        )
    for i in range(num_text):
        questions.append(
            {
                "id": str(num_mcq + i + 1),
                "type": "text",
                "question": f"Text question {i + 1}",
                "marks": 5,
                "unit": "Unit 1",
                "topic": "Photoelectric effect",
                "difficulty": "medium",
            }
        )
    return json.dumps({"questions": questions})


def test_pdfs_upload_indexes_for_rag_and_groups_by_subject(monkeypatch):
    async def _test():
        from src.services import gemini_service as gs

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("uploads")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            # Upload a PDF with a subject
            pdf_bytes = _minimal_pdf_bytes()
            upload_resp = await client.post(
                "/pdfs/upload",
                files={"file": ("physics_notes.pdf", pdf_bytes, "application/pdf")},
                data={"subject": "Physics", "title": "Physics Notes"},
                headers=_auth_headers(token),
            )
            assert upload_resp.status_code == 200, upload_resp.text
            pdf_id = upload_resp.json()["id"]

            # It should be marked processed
            assert upload_resp.json()["processed"] is True

            # It should have document_chunks entries (RAG-ready)
            if document_chunks_collection is not None:
                chunks = await document_chunks_collection.find({"doc_id": pdf_id}).to_list(length=None)
                assert len(chunks) > 0, "Uploaded PDF should produce RAG chunks"

            # Upload another PDF without a subject
            upload_resp2 = await client.post(
                "/pdfs/upload",
                files={"file": ("random_notes.pdf", pdf_bytes, "application/pdf")},
                headers=_auth_headers(token),
            )
            assert upload_resp2.status_code == 200, upload_resp2.text

            # Grouping endpoint
            by_subject_resp = await client.get(
                "/documents/by-subject",
                headers=_auth_headers(token),
            )
            assert by_subject_resp.status_code == 200, by_subject_resp.text
            data = by_subject_resp.json()

            subject_names = {g["name"] for g in data["subjects"]}
            assert "Physics" in subject_names
            physics_group = next(g for g in data["subjects"] if g["name"] == "Physics")
            assert any(d["id"] == pdf_id for d in physics_group["documents"])

            assert data["others"]["name"] == "Others"
            assert len(data["others"]["documents"]) >= 1

    _run(_test())


def test_generate_practice_mock_test_from_doc(monkeypatch):
    async def _test():
        from src.services import gemini_service as gs
        from src.services import mock_test_service as mts

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("practice")
            password = "testpassword123"
            token = await _token_for(client, email, password)

            pdf_bytes = _minimal_pdf_bytes()
            upload_resp = await client.post(
                "/pdfs/upload",
                files={"file": ("practice_material.pdf", pdf_bytes, "application/pdf")},
                data={"subject": "Physics"},
                headers=_auth_headers(token),
            )
            assert upload_resp.status_code == 200, upload_resp.text
            doc_id = upload_resp.json()["id"]

            fake_text = _fake_practice_test_json(num_mcq=2, num_text=1)
            monkeypatch.setattr(gs.gemini_service.model, "generate_content", lambda prompt: _FakeGeminiResponse(fake_text))

            gen_resp = await client.post(
                "/mock-tests/generate-from-doc",
                json={
                    "doc_ids": [doc_id],
                    "subject": "Physics",
                    "num_mcq": 2,
                    "num_text": 1,
                    "total_marks": 10,
                    "difficulty_level": "medium",
                },
                headers=_auth_headers(token),
            )
            assert gen_resp.status_code == 200, gen_resp.text
            test = gen_resp.json()
            assert test["total_marks"] == 10
            assert len(test["questions"]) == 3
            assert all(q["id"] for q in test["questions"])
            assert test["subject"] == "Physics"

            # It should be retrievable as a regular mock test
            get_resp = await client.get(
                f"/mock-tests/{test['test_id']}",
                headers=_auth_headers(token),
            )
            assert get_resp.status_code == 200, get_resp.text
            assert get_resp.json()["test_id"] == test["test_id"]

    _run(_test())
