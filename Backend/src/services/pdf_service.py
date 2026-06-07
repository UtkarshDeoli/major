import os
import io
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from fastapi import HTTPException
from pypdf import PdfReader
from src.core.data_store import (
    store_pdf_metadata,
    update_pdf_metadata,
    get_pdf_metadata,
    save_vector_db,
    load_vector_db
)

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return _model


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
        pdf_reader = PdfReader(io.BytesIO(file_content))
        page_count = len(pdf_reader.pages)

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"

        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into paragraphs
        paragraphs = [para.strip() for para in re.split(r'\n\n|\. ', text) if len(para.strip()) >= 20]

        # Generate embeddings (model loaded lazily on first use)
        model = _get_model()
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


async def get_relevant_context(pdf_id: str, question: str, top_k: int = 5) -> Tuple[str, List[int]]:
    """
    Get relevant context for a question from a specific PDF
    """
    # Get PDF content
    _, vector_data = await get_pdf_content(pdf_id)

    # Get paragraphs and embeddings
    paragraphs = vector_data["paragraphs"]
    import torch
    embeddings = torch.tensor(vector_data["embeddings"])

    # Encode the question (model loaded lazily on first use)
    model = _get_model()
    question_embedding = model.encode(question)

    # Find most relevant paragraphs
    from sentence_transformers import util
    similarities = util.pytorch_cos_sim(
        question_embedding,
        embeddings
    )[0]

    # Get top k most relevant paragraphs
    top_indices = similarities.argsort(descending=True)[:top_k]
    context = "\n\n".join([paragraphs[idx] for idx in top_indices.tolist()])

    return context, top_indices.tolist()
