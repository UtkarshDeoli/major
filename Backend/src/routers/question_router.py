from fastapi import APIRouter, Depends, HTTPException, Body, Path, Query
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional
import jwt
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
from src.core.config import SECRET_KEY, ALGORITHM
from src.services.llm_service import ask_question
from src.core.data_store import (
    create_chat_session,
    get_user_chat_sessions,
    get_chat_session,
    add_message_to_chat
)

router = APIRouter(prefix="/questions", tags=["Questions"])

# Helper function to get the current user from JWT token
async def get_current_user(token: str = Depends(lambda authorization: authorization)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token.replace("Bearer ", "")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authentication credentials"
        )

@router.post(
    "/ask", 
    response_model=QuestionResponse,
    summary="Ask a question",
    description="Ask a question with optional PDF context.",
)
async def ask(
    question_data: QuestionRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Ask a question with optional PDF context.
    """
    try:
        # Ask the question
        response = await ask_question(
            question=question_data.question,
            pdf_id=question_data.pdf_id,
            stream=False
        )
        
        return QuestionResponse(
            answer=response["answer"],
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
    user_id: str = Depends(get_current_user)
):
    """
    Ask a question with streaming response.
    """
    try:
        # Ask the question with streaming
        stream_generator = await ask_question(
            question=question_data.question,
            pdf_id=question_data.pdf_id,
            stream=True
        )
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error asking question with streaming: {str(e)}"
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
    pdf_id: Optional[str] = Body(None, embed=True),
    user_id: str = Depends(get_current_user)
):
    """
    Create a new chat session.
    """
    try:
        # Create a new chat session
        session = await create_chat_session(
            user_id=user_id,
            title=title,
            pdf_id=pdf_id
        )
        
        return ChatSession(
            id=session["id"],
            user_id=session["user_id"],
            pdf_id=session.get("pdf_id"),
            title=session["title"],
            messages=[],
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error creating chat session: {str(e)}"
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
    user_id: str = Depends(get_current_user)
):
    """
    Add a user message to a chat session and get an AI response.
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
        
        # Add the user message to the chat session
        await add_message_to_chat(
            session_id=session_id,
            role="user",
            content=message.content
        )
        
        # Ask the question using the PDF context if available
        response = await ask_question(
            question=message.content,
            pdf_id=session.get("pdf_id"),
            stream=False
        )
        
        # Add the AI response to the chat session
        await add_message_to_chat(
            session_id=session_id,
            role="assistant",
            content=response["answer"]
        )
        
        return QuestionResponse(
            answer=response["answer"],
            context=response.get("context")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error adding message to chat session: {str(e)}"
        )

@router.post(
    "/sessions/{session_id}/messages/stream", 
    summary="Add a message to a chat session with streaming response",
    description="Add a user message to a chat session and get a streaming AI response.",
)
async def add_message_stream(
    session_id: str = Path(..., description="The ID of the chat session"),
    message: ChatMessageRequest = Body(...),
    user_id: str = Depends(get_current_user)
):
    """
    Add a user message to a chat session and get a streaming AI response.
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
        
        # Add the user message to the chat session
        await add_message_to_chat(
            session_id=session_id,
            role="user",
            content=message.content
        )
        
        # Prepare streaming response
        async def stream_with_save():
            # Get streaming response
            stream_generator = await ask_question(
                question=message.content,
                pdf_id=session.get("pdf_id"),
                stream=True
            )
            
            # Create variables to collect full response
            full_response = ""
            context = None
            
            # Stream the response
            async for chunk in stream_generator():
                yield chunk
                
                # Try to parse the chunk to build full response
                try:
                    data = json.loads(chunk)
                    if "context" in data:
                        context = data["context"]
                    elif "response" in data:
                        full_response += data["response"]
                except:
                    pass
            
            # After streaming completes, save the full response to chat history
            await add_message_to_chat(
                session_id=session_id,
                role="assistant",
                content=full_response
            )
        
        return StreamingResponse(
            stream_with_save(),
            media_type="application/x-ndjson"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error adding message with streaming: {str(e)}"
        )
