"""Flashcard endpoints — AI-generated decks grounded on uploaded material.

A student (or their teacher) can generate a deck from one or more materials /
documents. Cards are stored with a light spaced-repetition schedule that the
review endpoint updates.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel

from src.core.models import GenerateFlashcardsRequest
from src.core.security import get_current_user
from src.core.plan_enforcement import enforce_limit
from src.core.data_store import (
    materials_collection,
    pdfs_collection,
    get_pdf_metadata,
    store_flashcard_deck,
    get_user_flashcard_decks,
    get_flashcard_deck,
    store_flashcards,
    get_flashcards_for_deck,
    update_flashcard,
    delete_flashcard_deck,
)
from src.services.gemini_service import gemini_service, generate_flashcards

router = APIRouter(prefix="/flashcards", tags=["Flashcards"])


class FlashcardDeckSummary(BaseModel):
    id: str
    user_id: str
    title: str
    subject: Optional[str] = None
    source_type: Optional[str] = None
    created_by: Optional[str] = None
    card_count: int = 0
    created_at: datetime


class FlashcardDeckListResponse(BaseModel):
    decks: List[FlashcardDeckSummary]


class FlashcardOut(BaseModel):
    id: str
    deck_id: str
    front: str
    back: str
    ease: int = 2
    interval_days: int = 0
    reps: int = 0
    due_at: datetime
    created_at: datetime


class FlashcardDeckDetail(BaseModel):
    id: str
    user_id: str
    title: str
    subject: Optional[str] = None
    source_type: Optional[str] = None
    created_by: Optional[str] = None
    card_count: int = 0
    created_at: datetime
    cards: List[FlashcardOut] = []


class GenerateDeckResponse(BaseModel):
    deck_id: str
    card_count: int


class ReviewRequest(BaseModel):
    grade: str  # "again" | "hard" | "good" | "easy"


async def _gather_content(user_id: str, material_ids: List[str], doc_ids: List[str]) -> str:
    """Collect text from the given materials/documents for grounding generation."""
    from bson import ObjectId

    parts: List[str] = []
    resolved_doc_ids: List[str] = list(doc_ids or [])

    # Resolve material_ids -> doc_ids (the pdfs metadata id used as RAG doc scope)
    if material_ids and materials_collection is not None:
        try:
            cursor = materials_collection.find({"_id": {"$in": [ObjectId(m) for m in material_ids]}})
            async for mat in cursor:
                did = mat.get("doc_id")
                if did:
                    resolved_doc_ids.append(did)
        except Exception:
            pass

    for did in resolved_doc_ids:
        pdf = await get_pdf_metadata(did)
        if not pdf or pdf.get("user_id") != user_id:
            continue
        file_path = pdf.get("file_path")
        if not file_path:
            continue
        try:
            if (file_path or "").lower().endswith(".pdf"):
                text = await gemini_service.extract_text_from_pdf(file_path)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if text:
                parts.append(text)
        except Exception as e:
            print(f"Skipping doc {did} for flashcards: {e}")

    return "\n\n".join(parts)[:8000]


@router.post("/generate", response_model=GenerateDeckResponse, status_code=status.HTTP_201_CREATED)
async def generate_deck(
    request: GenerateFlashcardsRequest,
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("flashcard")),
):
    """Generate a flashcard deck from materials/documents."""
    content = await _gather_content(user_id, request.material_ids, request.doc_ids)
    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable material found to generate flashcards from")

    cards_data = await generate_flashcards(content, num_cards=request.num_cards, subject=request.subject)
    if not cards_data:
        raise HTTPException(status_code=502, detail="Failed to generate flashcards from the material")

    deck_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    title = request.title or "Flashcard Deck"
    deck_doc = {
        "id": deck_id,
        "user_id": user_id,
        "title": title,
        "subject": request.subject,
        "source_material_ids": request.material_ids,
        "source_type": "ai",
        "created_by": None,
        "card_count": len(cards_data),
        "created_at": now,
        "updated_at": now,
    }
    await store_flashcard_deck(deck_doc)

    card_docs = [{
        "id": str(uuid.uuid4()),
        "deck_id": deck_id,
        "front": c["front"],
        "back": c["back"],
        "ease": 2,
        "interval_days": 0,
        "reps": 0,
        "due_at": now,
        "created_at": now,
    } for c in cards_data]
    await store_flashcards(card_docs)

    return GenerateDeckResponse(deck_id=deck_id, card_count=len(card_docs))


@router.get("/decks", response_model=FlashcardDeckListResponse)
async def list_decks(user_id: str = Depends(get_current_user)):
    decks = await get_user_flashcard_decks(user_id)
    return FlashcardDeckListResponse(decks=[
        FlashcardDeckSummary(
            id=d["id"], user_id=d["user_id"], title=d.get("title", "Deck"),
            subject=d.get("subject"), source_type=d.get("source_type"),
            created_by=d.get("created_by"), card_count=d.get("card_count", 0),
            created_at=d["created_at"],
        ) for d in decks
    ])


@router.get("/decks/{deck_id}", response_model=FlashcardDeckDetail)
async def get_deck(deck_id: str = Path(...), user_id: str = Depends(get_current_user)):
    deck = await get_flashcard_deck(deck_id)
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")
    # Teacher-created decks: allow the creator too
    allowed = {deck.get("user_id"), deck.get("created_by")}
    if user_id not in allowed:
        raise HTTPException(status_code=403, detail="Not authorized to view this deck")
    cards = await get_flashcards_for_deck(deck_id)
    return FlashcardDeckDetail(
        id=deck["id"], user_id=deck["user_id"], title=deck.get("title", "Deck"),
        subject=deck.get("subject"), source_type=deck.get("source_type"),
        created_by=deck.get("created_by"), card_count=deck.get("card_count", len(cards)),
        created_at=deck["created_at"],
        cards=[FlashcardOut(
            id=c["id"], deck_id=c["deck_id"], front=c["front"], back=c["back"],
            ease=c.get("ease", 2), interval_days=c.get("interval_days", 0),
            reps=c.get("reps", 0), due_at=c.get("due_at"), created_at=c["created_at"],
        ) for c in cards],
    )


@router.post("/cards/{card_id}/review")
async def review_card(
    card_id: str = Path(...),
    review: ReviewRequest = ...,
    user_id: str = Depends(get_current_user),
):
    """Update a card's spaced-repetition schedule based on a review grade."""
    from src.core.data_store import flashcards_collection
    if flashcards_collection is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    from bson import ObjectId
    card = await flashcards_collection.find_one({"_id": ObjectId(card_id)})
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    grade = review.grade
    ease = max(1.3, float(card.get("ease", 2)))
    reps = int(card.get("reps", 0)) + 1
    if grade == "again":
        ease = max(1.3, ease - 0.2)
        interval = 0
        reps = 0
    elif grade == "hard":
        ease = max(1.3, ease - 0.15)
        interval = max(1, int(card.get("interval_days", 0)) * 1.2)
    elif grade == "easy":
        ease = ease + 0.15
        interval = max(4, int(card.get("interval_days", 0)) * ease * 2.5)
    else:  # "good"
        interval = max(2, int(card.get("interval_days", 0)) * ease) if card.get("interval_days", 0) else 3

    due_at = datetime.now(timezone.utc) + timedelta(days=int(interval))
    await update_flashcard(card_id, {
        "ease": round(ease, 2), "interval_days": int(interval),
        "reps": reps, "due_at": due_at,
    })
    return {"card_id": card_id, "interval_days": int(interval), "due_at": due_at.isoformat(), "reps": reps}


@router.delete("/decks/{deck_id}", status_code=status.HTTP_200_OK)
async def remove_deck(deck_id: str = Path(...), user_id: str = Depends(get_current_user)):
    deck = await get_flashcard_deck(deck_id)
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")
    if deck.get("user_id") != user_id and deck.get("created_by") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this deck")
    await delete_flashcard_deck(deck_id)
    return {"deck_id": deck_id, "deleted": True}