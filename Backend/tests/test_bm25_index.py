import pytest
from src.services.bm25_index import BM25IndexService


def test_tokenize():
    service = BM25IndexService()
    tokens = service.tokenize("Hello, World!")
    assert tokens == ["hello", "world"]


def test_build_and_search():
    service = BM25IndexService()
    chunks = [
        {"chroma_id": "id1", "content": "The quick brown fox", "doc_id": "doc1", "subject": "Animals", "tags": ["mammals"]},
        {"chroma_id": "id2", "content": "The lazy dog sleeps", "doc_id": "doc1", "subject": "Animals", "tags": ["mammals"]},
        {"chroma_id": "id3", "content": "Python programming", "doc_id": "doc2", "subject": "Tech", "tags": ["coding"]},
    ]
    service.build_index("user1", chunks)
    
    results = service.search("user1", "quick fox", top_k=2)
    assert "id1" in results


def test_search_with_doc_filter():
    service = BM25IndexService()
    chunks = [
        {"chroma_id": "id1", "content": "The quick brown fox", "doc_id": "doc1", "subject": "Animals", "tags": ["mammals"]},
        {"chroma_id": "id2", "content": "The lazy dog sleeps", "doc_id": "doc2", "subject": "Animals", "tags": ["mammals"]},
    ]
    service.build_index("user1", chunks)
    
    results = service.search("user1", "quick fox", doc_ids=["doc2"], top_k=2)
    assert "id1" not in results  # id1 is from doc1, filtered out
    assert len(results) == 0  # doc2 doesn't contain "quick fox"
