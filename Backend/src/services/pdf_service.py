import io
import re
from typing import List, Tuple
import PyPDF2
from fastapi import UploadFile, HTTPException
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from ..core.data_store import data_store

import pdfplumber
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



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


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

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
    
    
    
