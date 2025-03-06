import io
import re
from typing import List, Tuple
import PyPDF2
from fastapi import UploadFile, HTTPException
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from ..core.data_store import data_store

# Initialize Sentence-BERT model
model = SentenceTransformer(settings.EMBEDDING_MODEL)

async def process_pdf(file: UploadFile) -> Tuple[int, str]:
    """
    Process a PDF file and extract embeddings
    
    Args:
        file: The uploaded PDF file
        
    Returns:
        Tuple containing paragraph count and success message
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        # Clean text
        text = clean_text(text)
        
        # Split into paragraphs
        paragraphs = split_into_paragraphs(text)
        
        # Generate embeddings
        embeddings = model.encode(paragraphs)
        
        # Store processed data
        data_store.store_pdf_data(text, paragraphs, embeddings.tolist())
        
        return len(paragraphs), "PDF processed successfully"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def clean_text(text: str) -> str:
    """Clean the extracted text by removing extra whitespace"""
    return re.sub(r'\s+', ' ', text).strip()

def split_into_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """Split text into paragraphs"""
    paragraphs = [para.strip() for para in re.split(r'\n\n|\. ', text) if len(para.strip()) >= min_length]
    return paragraphs
