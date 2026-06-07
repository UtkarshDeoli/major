import pytest
import uuid
import asyncio
import shutil
import os

from src.services.document_processor import detect_doc_type, chunk_document
from src.services.vector_store import VectorStore
from src.services.bm25_index import BM25IndexService
from src.services.query_engine import QueryEngine


def test_full_pipeline(monkeypatch):
    """Test the full multi-document RAG pipeline."""
    user_id = "test_user"

    # 1. Create test chunks
    text = "The photoelectric effect is the emission of electrons. Einstein explained it in 1905."
    chunks = chunk_document(text, doc_type="txt")

    # 2. Generate embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    chroma_chunks = []
    for chunk in chunks:
        chroma_id = str(uuid.uuid4())
        embedding = model.encode(chunk["content"]).tolist()
        chroma_chunks.append({
            "chroma_id": chroma_id,
            "user_id": user_id,
            "doc_id": "test_doc",
            "doc_name": "test.txt",
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "embedding": embedding,
            "section": chunk.get("section"),
            "doc_type": "txt",
        })

    # Patch get_chunks_by_chroma_ids to avoid MongoDB event-loop issues in tests
    async def mock_get_chunks_by_chroma_ids(chroma_ids):
        chunk_map = {c["chroma_id"]: c for c in chroma_chunks}
        return [chunk_map[cid] for cid in chroma_ids if cid in chunk_map]

    import src.core.data_store as ds_module
    monkeypatch.setattr(ds_module, "get_chunks_by_chroma_ids", mock_get_chunks_by_chroma_ids)

    # Reset VectorStore singleton for test isolation
    VectorStore._instance = None
    VectorStore._client = None

    # 3. Store in ChromaDB
    test_db_path = "./test_chroma_pipeline"
    vector_store = VectorStore(test_db_path)
    vector_store.add_chunks(user_id, chroma_chunks)

    # 4. Build BM25 index
    bm25 = BM25IndexService()
    bm25.build_index(user_id, chroma_chunks)

    # 5. Query
    query_engine = QueryEngine(vector_store, bm25)
    context, sources, _ = asyncio.run(
        query_engine.query(user_id, "What is the photoelectric effect?")
    )

    # Verify
    assert "photoelectric" in context.lower()
    assert len(sources) > 0
    assert sources[0]["doc_name"] == "test.txt"

    # Cleanup
    VectorStore._instance = None
    VectorStore._client = None
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path, ignore_errors=True)
