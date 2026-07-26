import json
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from fastapi import HTTPException
from src.services.gemini_service import gemini_service
from src.services.vector_store import VectorStore
from src.services.bm25_index import BM25IndexService
from src.services.query_engine import QueryEngine

# Initialize new services
vector_store = VectorStore()
bm25_service = BM25IndexService()
query_engine = QueryEngine(vector_store, bm25_service)


# ISO-639-1 / common language codes mapped to friendly names
_LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese",
}


def _build_language_instruction(language: Optional[str]) -> str:
    """Return a prompt appendix that tells the model which language to respond in."""
    if not language or language.lower() in {"en", "english", ""}:
        return ""
    name = _LANGUAGE_NAMES.get(language.lower()) or language
    return f"IMPORTANT: Respond entirely in {name} ({language})."

async def ask_question(question: str,
                       pdf_id: Optional[str] = None,      # DEPRECATED
                       doc_ids: Optional[List[str]] = None,  # NEW
                       subject: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       user_id: str = None,
                       stream: bool = False,
                       language: Optional[str] = None,
                       image_data_url: Optional[str] = None):
    """Ask a question using the multi-document RAG system.

    Backward compatibility: if pdf_id is provided, treat as doc_ids=[pdf_id].
    """
    # Normalize deprecated pdf_id
    if pdf_id and not doc_ids:
        doc_ids = [pdf_id]

    context = ""
    sources = []

    # Get context from documents if user_id is provided
    if user_id:
        context, sources, chunks = await query_engine.query(
            user_id=user_id,
            question=question,
            doc_ids=doc_ids,
            subject=subject,
            tags=tags,
            top_k=5
        )

    language_instruction = _build_language_instruction(language)

    # Build prompt
    if context:
        prompt = query_engine.build_prompt(question, context, sources)
        if language_instruction:
            prompt += f"\n\n{language_instruction}"
    else:
        prompt = f"""You are an AI tutor.

- Provide a clear, concise, and well-structured answer.
- Focus on key points that are important for exams.
- Avoid unnecessary introductions—start directly with the answer.
- If necessary, break down complex ideas into simpler explanations.

**Question:** {question}

**Exam-Focused Answer:**
"""
        if language_instruction:
            prompt += f"\n\n{language_instruction}"

    image_part = _build_image_part(image_data_url)

    if stream:
        return await stream_llm_response(prompt, context, sources, image_part)
    else:
        return await get_llm_response(prompt, context, sources, image_part)


def _build_image_part(image_data_url: Optional[str]):
    """Build a Gemini image part from a base64 data URL.

    Supports 'data:image/...;base64,...' URLs.
    """
    if not image_data_url or not image_data_url.startswith("data:image"):
        return None
    try:
        header, encoded = image_data_url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        import base64
        image_bytes = base64.b64decode(encoded)
        return {"mime_type": mime, "data": image_bytes}
    except Exception:
        return None


async def get_llm_response(prompt: str, context: str = "", sources: List[Dict] = None, image_part: Optional[Dict] = None):
    """Get a non-streaming response from Gemini LLM."""
    if not gemini_service:
        raise HTTPException(
            status_code=503,
            detail="Gemini service is not available. Please check GEMINI_API_KEY configuration."
        )

    try:
        contents = [prompt]
        if image_part:
            contents.append(image_part)
        response = gemini_service.model.generate_content(contents)

        if not response or not response.text:
            raise HTTPException(
                status_code=500,
                detail="Empty response from Gemini API"
            )

        return {
            "answer": response.text.strip(),
            "sources": sources or [],
            "context": context
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response from Gemini: {str(e)}"
        )

async def stream_llm_response(prompt: str, context: str = "", sources: List[Dict] = None, image_part: Optional[Dict] = None):
    """Get a streaming response from Gemini LLM."""
    if not gemini_service:
        async def error_response():
            yield json.dumps({"error": "Gemini service is not available."}) + "\n"
        return error_response
    
    async def stream_response():
        try:
            # Add context and sources as first chunk
            if context:
                context_data = {"context": context}
                if sources:
                    context_data["sources"] = sources
                yield json.dumps(context_data) + "\n"
            
            contents = [prompt]
            if image_part:
                contents.append(image_part)
            response = gemini_service.model.generate_content(contents, stream=True)

            for chunk in response:
                if chunk.text:
                    data = {
                        "response": chunk.text,
                        "done": False
                    }
                    yield json.dumps(data) + "\n"
            
            yield json.dumps({"response": "", "done": True}) + "\n"
            
        except Exception as e:
            error_data = {"error": f"Error generating streaming response: {str(e)}"}
            yield json.dumps(error_data) + "\n"
    
    return stream_response
