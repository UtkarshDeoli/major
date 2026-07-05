import json
from fastapi import APIRouter, Depends, HTTPException, Body, Path
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

from src.core.models import (
    QuestionRequest,
    QuestionResponse,
    ChatSession,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatMessageRequest,
    ChatMessageResponse
)
from src.core.security import get_current_user
from src.core.plan_enforcement import enforce_limit, increment_usage
from src.services.llm_service import ask_question
from src.core.data_store import (
    create_chat_session,
    get_user_chat_sessions,
    get_chat_session,
    add_message_to_chat
)

router = APIRouter(prefix="/questions", tags=["Questions"])

@router.post(
    "/ask", 
    response_model=QuestionResponse,
    summary="Ask a question",
    description="Ask a question with optional PDF context.",
)
async def ask(
    question_data: QuestionRequest,
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("chat_message")),
):
    """Ask a question with optional document context."""
    try:
        response = await ask_question(
            question=question_data.question,
            pdf_id=question_data.pdf_id,      # backward compat
            doc_ids=question_data.doc_ids,    # new multi-doc
            subject=question_data.subject,
            tags=question_data.tags,
            user_id=user_id,
            stream=False
        )

        await increment_usage(user_id, "chat_message")
        return QuestionResponse(
            answer=response["answer"],
            sources=response.get("sources"),  # NEW
            context=response.get("context")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error asking question: {str(e)}"
        )

@router.post(
    "/ask/stream", 
    summary="Ask a question with streaming response",
    description="Ask a question with optional PDF context and get a streaming response.",
)
async def ask_stream(
    question_data: QuestionRequest,
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("chat_message")),
):
    """Ask a question with streaming response."""
    try:
        stream_generator = await ask_question(
            question=question_data.question,
            pdf_id=question_data.pdf_id,
            doc_ids=question_data.doc_ids,
            subject=question_data.subject,
            tags=question_data.tags,
            user_id=user_id,
            stream=True
        )

        await increment_usage(user_id, "chat_message")
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error streaming: {str(e)}"
        )

# Chat session endpoints
@router.post(
    "/sessions", 
    response_model=ChatSession,
    summary="Create a new chat session",
    description="Create a new chat session with optional PDF context.",
)
async def create_session(
    title: str = Body(..., embed=True),
    pdf_id: Optional[str] = Body(None, embed=True),  # deprecated
    doc_ids: Optional[List[str]] = Body(None, embed=True),  # new
    user_id: str = Depends(get_current_user)
):
    """Create a new chat session."""
    try:
        # Normalize
        if pdf_id and not doc_ids:
            doc_ids = [pdf_id]
        
        session = await create_chat_session(
            user_id=user_id,
            title=title,
            pdf_id=pdf_id,
            doc_ids=doc_ids
        )
        
        return ChatSession(
            id=session["id"],
            user_id=session["user_id"],
            pdf_id=session.get("pdf_id"),
            doc_ids=session.get("doc_ids"),
            title=session["title"],
            messages=[],
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error creating session: {str(e)}"
        )

@router.get(
    "/sessions", 
    response_model=ChatSessionListResponse,
    summary="List all chat sessions",
    description="List all chat sessions for the current user.",
)
async def list_sessions(user_id: str = Depends(get_current_user)):
    """
    List all chat sessions for the current user.
    """
    try:
        # Get all chat sessions for the user
        sessions = await get_user_chat_sessions(user_id)
        
        return ChatSessionListResponse(
            sessions=[
                ChatSessionResponse(
                    id=session["id"],
                    title=session["title"],
                    pdf_id=session.get("pdf_id"),
                    doc_ids=session.get("doc_ids"),
                    created_at=session["created_at"],
                    updated_at=session["updated_at"],
                    message_count=session.get("message_count", 0)
                )
                for session in sessions
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error listing chat sessions: {str(e)}"
        )

@router.get(
    "/sessions/{session_id}", 
    response_model=ChatSession,
    summary="Get a chat session",
    description="Get a specific chat session with all messages.",
)
async def get_session(
    session_id: str = Path(..., description="The ID of the chat session"),
    user_id: str = Depends(get_current_user)
):
    """
    Get a specific chat session with all messages.
    """
    try:
        # Get the chat session
        session = await get_chat_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404, 
                detail=f"Chat session with ID {session_id} not found"
            )
        
        # Check if the chat session belongs to the user
        if session["user_id"] != user_id:
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to access this chat session"
            )
        
        return ChatSession(
            id=session["id"],
            user_id=session["user_id"],
            pdf_id=session.get("pdf_id"),
            title=session["title"],
            messages=session.get("messages", []),
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting chat session: {str(e)}"
        )

@router.post(
    "/sessions/{session_id}/messages", 
    response_model=QuestionResponse,
    summary="Add a message to a chat session",
    description="Add a user message to a chat session and get an AI response.",
)
async def add_message(
    session_id: str = Path(..., description="The ID of the chat session"),
    message: ChatMessageRequest = Body(...),
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("chat_message")),
):
    """Add a message to a chat session."""
    try:
        session = await get_chat_session(session_id)
        if not session or session["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        # Resolve document scope
        doc_ids = session.get("doc_ids")
        if not doc_ids and session.get("pdf_id"):
            doc_ids = [session["pdf_id"]]

        await add_message_to_chat(session_id, "user", message.content)

        response = await ask_question(
            question=message.content,
            doc_ids=doc_ids,
            user_id=user_id,
            stream=False
        )

        await add_message_to_chat(session_id, "assistant", response["answer"])
        await increment_usage(user_id, "chat_message")
        return QuestionResponse(
            answer=response["answer"],
            sources=response.get("sources"),
            context=response.get("context")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )

@router.post(
    "/sessions/{session_id}/messages/stream", 
    summary="Add a message to a chat session with streaming response",
    description="Add a user message to a chat session and get a streaming AI response.",
)
async def add_message_stream(
    session_id: str = Path(..., description="The ID of the chat session"),
    message: ChatMessageRequest = Body(...),
    user_id: str = Depends(get_current_user),
    _plan: dict = Depends(enforce_limit("chat_message")),
):
    """Add a message to a chat session with streaming response."""
    try:
        session = await get_chat_session(session_id)
        if not session or session["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        # Resolve document scope
        doc_ids = session.get("doc_ids")
        if not doc_ids and session.get("pdf_id"):
            doc_ids = [session["pdf_id"]]

        await add_message_to_chat(session_id, "user", message.content)
        await increment_usage(user_id, "chat_message")

        async def stream_with_save():
            stream_generator = await ask_question(
                question=message.content,
                doc_ids=doc_ids,
                user_id=user_id,
                stream=True
            )
            
            full_response = ""
            context = None
            
            async for chunk in stream_generator():
                yield chunk
                
                try:
                    data = json.loads(chunk)
                    if "context" in data:
                        context = data["context"]
                    elif "response" in data:
                        full_response += data["response"]
                except:
                    pass
            
            await add_message_to_chat(session_id, "assistant", full_response)
        
        return StreamingResponse(
            stream_with_save(),
            media_type="application/x-ndjson"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )
