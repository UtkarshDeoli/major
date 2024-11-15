from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import requests
import io
import re
import json
from typing import Dict
import torch
import aiohttp
import nltk

app = FastAPI()

# Allow CORS for all origins (you can restrict this to specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store processed PDF data
pdf_data = {
    "text": "",
    "paragraphs": [],
    "embeddings": []
}

# Initialize Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Route 1: Accept PDF file and process with Sentence-BERT
@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into paragraphs
        paragraphs = [para.strip() for para in re.split(r'\n\n|\. ', text) if len(para.strip()) >= 20]
        
        # Generate embeddings
        embeddings = model.encode(paragraphs)
        
        # Store processed data
        pdf_data["text"] = text
        pdf_data["paragraphs"] = paragraphs
        pdf_data["embeddings"] = embeddings.tolist()
        
        return {"status": "success", "paragraphs": len(paragraphs), "message": "PDF processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# Regular ask-question endpoint (non-streaming)
@app.post("/ask-question")
async def ask_question(request: Dict[str, str]):
    question = request.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question field is required.")
    
    if not pdf_data["paragraphs"]:
        raise HTTPException(status_code=400, detail="No PDF has been processed. Please upload a PDF first.")
    
    try:
        # Encode the question
        question_embedding = model.encode(question)
        
        # Find most relevant paragraphs
        similarities = util.pytorch_cos_sim(
            question_embedding, 
            torch.tensor(pdf_data["embeddings"])
        )[0]
        
        # Get top 3 most relevant paragraphs
        top_indices = similarities.argsort(descending=True)[:3]
        context = "\n\n".join([pdf_data["paragraphs"][idx] for idx in top_indices.tolist()])
        
        # Send question with context to the LLM
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
            return {"answer": result.get("response", "No response from LLM") , "context": context}
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Error from LLM server: {response.text}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# New streaming endpoint for questions
@app.post("/ask-question/stream")
async def ask_question_stream(request: Dict[str, str]):
    question = request.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question field is required.")
    
    if not pdf_data["paragraphs"]:
        raise HTTPException(status_code=400, detail="No PDF has been processed. Please upload a PDF first.")
    
    try:
        # Encode the question
        question_embedding = model.encode(question)
        
        # Find most relevant paragraphs
        similarities = util.pytorch_cos_sim(
            question_embedding, 
            torch.tensor(pdf_data["embeddings"])
        )[0]
        
        # Get top 3 most relevant paragraphs
        top_indices = similarities.argsort(descending=True)[:3]
        context = "\n\n".join([pdf_data["paragraphs"][idx] for idx in top_indices.tolist()])
        
        # Send question with context to the LLM
        prompt = f"""
        You are an AI tutor. Answer the question based on the provided textbook excerpts.

        Answer the following question based on the provided context:

        Answer in a simple and friendly manner:
        If the context below is not relevant, please write a message that the question is not relevant to the context.
                
        Context: {context}

        Question: {question}

        Answer:"""

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

        return StreamingResponse(
            stream_response(),
            media_type="application/x-ndjson"
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating streaming response: {str(e)}")

# Route to check if PDF is processed
@app.get("/status")
async def status():
    if pdf_data["paragraphs"]:
        return {"status": "PDF processed", "paragraphs": len(pdf_data["paragraphs"])}
    else:
        return {"status": "No PDF processed"}
    
if __name__ == "__main__":
    import uvicorn
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    uvicorn.run(app, host="0.0.0.0", port=8001)