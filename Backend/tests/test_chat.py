"""Tests for the chat /questions endpoints including streaming."""

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from httpx import ASGITransport
import pytest

from src.core.data_store import (
    users_collection,
    chat_sessions_collection,
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


def test_chat_session_lifecycle_and_message(monkeypatch):
    async def _test():
        from src.services import gemini_service as gs

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("chat")
            password = "testpassword123"
            token = await _token_for(client, email, password)
            headers = _auth_headers(token)

            # Upload a PDF so the RAG pipeline has chunks to retrieve.
            pdf_bytes = _minimal_pdf_bytes()
            upload_resp = await client.post(
                "/pdfs/upload",
                files={"file": ("physics_notes.pdf", pdf_bytes, "application/pdf")},
                data={"subject": "Physics", "title": "Physics Notes"},
                headers=headers,
            )
            assert upload_resp.status_code == 200, upload_resp.text
            doc_id = upload_resp.json()["id"]

            # Create a chat session scoped to the uploaded document.
            session_resp = await client.post(
                "/questions/sessions",
                json={"title": "Physics chat", "pdf_id": doc_id, "doc_ids": [doc_id]},
                headers=headers,
            )
            assert session_resp.status_code == 200, session_resp.text
            session = session_resp.json()
            assert session["doc_ids"] == [doc_id]
            session_id = session["id"]

            # Mock Gemini response.
            monkeypatch.setattr(
                gs.gemini_service.model,
                "generate_content",
                lambda prompt, stream=False: _FakeGeminiResponse("The photoelectric effect is the emission of electrons from a metal surface when light shines on it."),
            )

            # Send a message in the session.
            msg_resp = await client.post(
                f"/questions/sessions/{session_id}/messages",
                json={"content": "What is the photoelectric effect?"},
                headers=headers,
            )
            assert msg_resp.status_code == 200, msg_resp.text
            body = msg_resp.json()
            assert "photoelectric" in body["answer"].lower()

            # Verify the session now has both user and assistant messages.
            get_resp = await client.get(
                f"/questions/sessions/{session_id}",
                headers=headers,
            )
            assert get_resp.status_code == 200, get_resp.text
            messages = get_resp.json()["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"

    _run(_test())


def test_chat_message_stream_returns_ndjson(monkeypatch):
    async def _test():
        from src.services import gemini_service as gs

        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            email = _unique("chat_stream")
            password = "testpassword123"
            token = await _token_for(client, email, password)
            headers = _auth_headers(token)

            pdf_bytes = _minimal_pdf_bytes()
            upload_resp = await client.post(
                "/pdfs/upload",
                files={"file": ("physics_notes.pdf", pdf_bytes, "application/pdf")},
                data={"subject": "Physics"},
                headers=headers,
            )
            assert upload_resp.status_code == 200, upload_resp.text
            doc_id = upload_resp.json()["id"]

            session_resp = await client.post(
                "/questions/sessions",
                json={"title": "Stream chat", "doc_ids": [doc_id]},
                headers=headers,
            )
            assert session_resp.status_code == 200, session_resp.text
            session_id = session_resp.json()["id"]

            class _FakeStreamGeminiResponse:
                def __init__(self, chunks):
                    self._chunks = chunks

                def __iter__(self):
                    for text in self._chunks:
                        yield _FakeGeminiResponse(text)

            monkeypatch.setattr(
                gs.gemini_service.model,
                "generate_content",
                lambda prompt, stream=False: _FakeStreamGeminiResponse(["The ", "photoelectric ", "effect ", "is..."]),
            )

            stream_resp = await client.post(
                f"/questions/sessions/{session_id}/messages/stream",
                json={"content": "Explain photoelectric effect"},
                headers=headers,
            )
            assert stream_resp.status_code == 200, stream_resp.text

            # Response should be parseable NDJSON.
            lines = [line for line in stream_resp.text.split("\n") if line.strip()]
            assert len(lines) > 0
            import json
            for line in lines:
                parsed = json.loads(line)
                # Each line should be a dict with at least a response or done marker.
                assert isinstance(parsed, dict)

    _run(_test())
