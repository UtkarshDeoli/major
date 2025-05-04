import json
import torch
import aiohttp
import requests
from typing import Dict, Any, Optional, Tuple
from fastapi import HTTPException
from src.services.pdf_service import get_relevant_context

async def ask_question(question: str, pdf_id: Optional[str] = None, stream: bool = False):
    """
    Ask a question to the LLM with or without PDF context
    """
    context = ""
    
    # Get context from PDF if pdf_id is provided
    if pdf_id:
        try:
            context, _ = await get_relevant_context(pdf_id, question, top_k=3)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error getting context from PDF: {str(e)}"
            )
    
    # Generate the prompt with or without context
    if context:
        prompt = f"""You are an AI tutor. Answer the question strictly based on the provided textbook excerpts.  

        - Provide a clear, concise, and well-structured answer.  
        - Focus on key points that are important for exams.  
        - Avoid unnecessary introductions—start directly with the answer.  
        - If necessary, break down complex ideas into simpler explanations.  
        - If the context does not contain relevant information, state that the question is not covered in the given context.  

        **Context:** {context}  

        **Question:** {question}  

        **Exam-Focused Answer:**  

        """
    else:
        prompt = f"""You are an AI tutor. 

        - Provide a clear, concise, and well-structured answer.  
        - Focus on key points that are important for exams.  
        - Avoid unnecessary introductions—start directly with the answer.  
        - If necessary, break down complex ideas into simpler explanations.  

        **Question:** {question}  

        **Exam-Focused Answer:**  

        """
    
    # Handle streaming or non-streaming response
    if stream:
        return await stream_llm_response(prompt, context)
    else:
        return await get_llm_response(prompt, context)

async def get_llm_response(prompt: str, context: str = ""):
    """
    Get a non-streaming response from the LLM
    """
    try:
        # Send request to local LLM with stream=False
        response = requests.post(
            "http://ollama.utkarshdeoli.in/api/generate", 
            json={
                "model": "llama3.2:1b",
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
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating response: {str(e)}"
        )

async def stream_llm_response(prompt: str, context: str = ""):
    """
    Get a streaming response from the LLM
    """
    async def stream_response():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://ollama.utkarshdeoli.in/api/generate", 
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "max_tokens": 500,
                    "stream": True
                }
            ) as response:
                # Add context as first chunk if available
                if context:
                    context_data = {"context": context}
                    yield json.dumps(context_data) + "\n"
                
                # Read the streaming response line by line
                async for line in response.content:
                    if line:
                        # Parse the JSON response from each line
                        try:
                            data = json.loads(line)
                            # Yield the token with proper JSON formatting
                            yield json.dumps(data) + "\n"
                        except json.JSONDecodeError:
                            yield '{"error": "Invalid JSON received from LLM"}\n'
    
    return stream_response
