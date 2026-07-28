from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.core.security import get_current_user
from src.core.plan_enforcement import enforce_limit
from src.core.limiter import limiter, SOCRATIC_LIMIT
from src.services.socratic_service import explain_socratically, socratic_feedback_for_answer
from src.services.query_engine import QueryEngine
from src.services.vector_store import VectorStore
from src.services.bm25_index import BM25IndexService

router = APIRouter(prefix="/socratic", tags=["Socratic Tutor"])


class SocraticExplainRequest(BaseModel):
    question: str
    concept: Optional[str] = None
    doc_ids: Optional[List[str]] = None


class SocraticFeedbackRequest(BaseModel):
    question: str
    user_answer: str
    correct_answer: Optional[str] = None


# Reuse query engine for document context
_query_engine = QueryEngine(VectorStore(), BM25IndexService())


@router.post("/explain")
@limiter.limit(SOCRATIC_LIMIT)
async def socratic_explain(
    request: Request,
    req: SocraticExplainRequest,
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("chat_message")),
):
    """Get a Socratic, step-by-step explanation that guides without giving the answer."""
    try:
        doc_context = ""
        if req.doc_ids:
            context, _sources, _chunks = await _query_engine.query(
                user_id=user_id,
                question=req.question,
                doc_ids=req.doc_ids,
                top_k=3,
            )
            doc_context = context

        result = await explain_socratically(
            question=req.question,
            concept=req.concept,
            doc_context=doc_context,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Socratic explanation failed: {str(e)}")


@router.post("/feedback")
@limiter.limit(SOCRATIC_LIMIT)
async def socratic_feedback(
    request: Request,
    req: SocraticFeedbackRequest,
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("chat_message")),
):
    """Get Socratic feedback on a specific answer (right or wrong)."""
    try:
        result = await socratic_feedback_for_answer(
            question=req.question,
            user_answer=req.user_answer,
            correct_answer=req.correct_answer,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Socratic feedback failed: {str(e)}")
