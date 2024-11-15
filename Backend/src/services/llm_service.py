import json
import torch
import aiohttp
import requests
from typing import Dict, Any, AsyncGenerator
from sentence_transformers import util
from fastapi import HTTPException
from ..core.config import settings
from ..core.data_store import data_store

async def get_answer(question: str) -> Dict[str, Any]:
    """
    Get answer for a question from the LLM
    
    Args:
        question: User question
        
    Returns:
        Dictionary containing answer and context
    """
    try:
        # Get context for the question
        context = get_relevant_context(question)
        
        # Create prompt
        prompt = create_prompt(question, context)
        
        # Send request to LLM
        response = requests.post(
            settings.LLM_URL,
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return {"answer": result.get("response", "No response from LLM"), "context": context}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error from LLM server: {response.text}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

async def stream_answer(question: str) -> AsyncGenerator[str, None]:
    """
    Stream answer for a question from the LLM
    
    Args:
        question: User question
        
    Yields:
        JSON strings for each chunk of the answer
    """
    try:
        # Get context for the question
        context = get_relevant_context(question)
        
        # Create prompt
        prompt = create_prompt(question, context)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.LLM_URL,
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "max_tokens": settings.MAX_TOKENS,
                    "stream": True
                }
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            yield json.dumps(data) + "\n"
                        except json.JSONDecodeError:
                            yield '{"error": "Invalid JSON received from LLM"}\n'
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

def get_relevant_context(question: str) -> str:
    """
    Get relevant context for a question
    
    Args:
        question: User question
        
    Returns:
        Relevant context from PDF paragraphs
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)
    pdf_data = data_store.get_pdf_data()
    
    # Find most relevant paragraphs
    similarities = util.pytorch_cos_sim(
        question_embedding,
        torch.tensor(pdf_data["embeddings"])
    )[0]
    
    # Get top k most relevant paragraphs
    top_indices = similarities.argsort(descending=True)[:settings.CONTEXT_SIZE]
    context = "\n\n".join([pdf_data["paragraphs"][idx] for idx in top_indices.tolist()])
    
    return context

def create_prompt(question: str, context: str) -> str:
    """
    Create a prompt for the LLM
    
    Args:
        question: User question
        context: Relevant context
        
    Returns:
        Formatted prompt
    """
    return f"""You are an AI tutor. Answer the question strictly based on the provided textbook excerpts.  

    - Provide a clear, concise, and well-structured answer.  
    - Focus on key points that are important for exams.  
    - Avoid unnecessary introductions—start directly with the answer.  
    - If necessary, break down complex ideas into simpler explanations.  
    - If the context does not contain relevant information, state that the question is not covered in the given context.  

    **Context:** {context}  

    **Question:** {question}  

    **Exam-Focused Answer:**  
    """
