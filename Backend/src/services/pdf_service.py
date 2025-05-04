import os
import io
import re
import json
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import PyPDF2
from fastapi import UploadFile, HTTPException
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.core.data_store import (
    store_pdf_metadata, 
    update_pdf_metadata, 
    get_pdf_metadata,
    save_vector_db,
    load_vector_db
)

# Initialize the sentence transformer model 
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

async def process_and_store_pdf(
    file_content: bytes,
    filename: str,
    user_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process a PDF file:
    1. Save the file to disk
    2. Store metadata in MongoDB
    3. Process content with embeddings
    4. Save vector database
    """
    # Create user directory if it doesn't exist
    user_dir = os.path.join("uploads", user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Save file to disk
    file_path = os.path.join(user_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Store metadata in MongoDB
    pdf_metadata = await store_pdf_metadata(
        filename=filename,
        size=len(file_content),
        user_id=user_id,
        file_path=file_path,
        title=title,
        description=description,
        tags=tags
    )
    
    # Process PDF content
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        page_count = len(pdf_reader.pages)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into paragraphs
        paragraphs = [para.strip() for para in re.split(r'\n\n|\. ', text) if len(para.strip()) >= 20]
        
        # Generate embeddings
        embeddings = model.encode(paragraphs)
        
        # Create vector database
        vector_data = {
            "text": text,
            "paragraphs": paragraphs,
            "embeddings": embeddings.tolist(),
            "processed_date": datetime.now().isoformat()
        }
        
        # Save vector database
        vector_db_path = await save_vector_db(pdf_metadata["id"], vector_data)
        
        # Update PDF metadata
        await update_pdf_metadata(
            pdf_id=pdf_metadata["id"],
            update_data={
                "processed": True,
                "page_count": page_count,
                "vector_db_path": vector_db_path
            }
        )
        
        return await get_pdf_metadata(pdf_metadata["id"])
    except Exception as e:
        # Update PDF metadata with error
        await update_pdf_metadata(
            pdf_id=pdf_metadata["id"],
            update_data={
                "processed": False,
                "processing_error": str(e)
            }
        )
        raise e

async def get_pdf_content(pdf_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Get PDF content and metadata
    """
    # Get PDF metadata
    pdf_metadata = await get_pdf_metadata(pdf_id)
    if not pdf_metadata:
        raise Exception(f"PDF with ID {pdf_id} not found")
    
    if not pdf_metadata.get("processed"):
        raise Exception(f"PDF with ID {pdf_id} has not been processed yet")
    
    # Load vector database
    vector_data = await load_vector_db(pdf_id)
    
    return pdf_metadata, vector_data

async def get_relevant_context(pdf_id: str, question: str, top_k: int = 3) -> Tuple[str, List[int]]:
    """
    Get relevant context for a question from a specific PDF
    """
    # Get PDF content
    _, vector_data = await get_pdf_content(pdf_id)
    
    # Get paragraphs and embeddings
    paragraphs = vector_data["paragraphs"]
    embeddings = torch.tensor(vector_data["embeddings"])
    
    # Encode the question
    question_embedding = model.encode(question)
    
    # Find most relevant paragraphs
    similarities = util.pytorch_cos_sim(
        question_embedding, 
        embeddings
    )[0]
    
    # Get top k most relevant paragraphs
    top_indices = similarities.argsort(descending=True)[:top_k]
    context = "\n\n".join([paragraphs[idx] for idx in top_indices.tolist()])
    
    return context, top_indices.tolist()

def preprocess_questions(text):
    raw_questions = re.split(r'\n?\d{1,2}[.)]', text)[1:]  # Split on 1., 2. etc.
    cleaned = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for q in raw_questions:
        q = q.strip().replace('\n', ' ')
        q = re.sub(r'[^\w\s]', '', q.lower())
        tokens = nltk.word_tokenize(q)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
        cleaned.append(" ".join(tokens))
    return cleaned, raw_questions


def select_sample_questions(cleaned_questions, original_questions, num_q=10):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_questions)

    num_clusters = min(num_q, len(cleaned_questions))
    if num_clusters == 0:
        return []

    km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    km.fit(X)
    labels = km.labels_

    selected = []
    for cluster in range(num_clusters):
        idx = np.where(labels == cluster)[0][0]  # Pick one from each cluster
        selected.append(original_questions[idx].strip())
    return selected

def export_sample_paper(questions, output_path="Sample_Exam_Paper.txt"):
    with open(output_path, "w") as f:
        f.write("SAMPLE EXAM PAPER\n")
        f.write("="*40 + "\n\n")
        for i, q in enumerate(questions, 1):
            f.write(f"{i}. {q.strip().capitalize()}\n\n")
    return output_path



async def process_questions(file: UploadFile) -> Tuple[int, str]:
    """
    Process a PDF file of questions and extract embeddings
    
    Args:
        file: The uploaded PDF file
        
    Returns:
        Tuple containing question count and success message
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        print("Upload Past Paper PDF:")
        content = await file.read()
        # pdf_path = next(iter(UploadFile))

        text = extract_text_from_pdf(content)
        cleaned, raw = preprocess_questions(text)
        sample_questions = select_sample_questions(cleaned, raw, num_q=10)
        export_sample_paper(sample_questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing questions: {str(e)}")



