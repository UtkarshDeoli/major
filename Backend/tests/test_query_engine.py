import pytest
from src.services.query_engine import QueryEngine, reciprocal_rank_fusion

def test_rrf_fusion():
    vector_results = [
        {"chroma_id": "A", "score": 0.9},
        {"chroma_id": "B", "score": 0.8},
        {"chroma_id": "C", "score": 0.7},
    ]
    bm25_results = ["B", "C", "D"]
    
    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    
    assert "A" in fused
    assert "B" in fused
    assert "C" in fused
    assert "D" in fused
    # B should have highest score (appears in both lists)
    assert fused["B"] > fused["A"]
    assert fused["B"] > fused["D"]

def test_build_context_with_citations():
    engine = QueryEngine(None, None)  # Mocks, we test the helper
    chunks = [
        {"chroma_id": "id1", "doc_name": "test.pdf", "section": "Intro", "page": 1, "content": "Hello world"},
    ]
    context, sources = engine._build_context(chunks)
    assert "[1] test.pdf, Intro, Page 1" in context
    assert sources[0]["index"] == 1
    assert sources[0]["doc_name"] == "test.pdf"
    assert sources[0]["page"] == 1
    assert sources[0]["section"] == "Intro"
    assert sources[0]["chroma_id"] == "id1"

def test_build_context_with_locator():
    engine = QueryEngine(None, None)
    chunks = [
        {"chroma_id": "id2", "doc_name": "test2.pdf", "locator": "Section 2.1", "content": "Some content"},
    ]
    context, sources = engine._build_context(chunks)
    assert "[1] test2.pdf, Section 2.1" in context
    assert sources[0]["locator"] == "Section 2.1"

def test_build_context_no_section_no_page():
    engine = QueryEngine(None, None)
    chunks = [
        {"chroma_id": "id3", "doc_name": "test3.pdf", "content": "Plain content"},
    ]
    context, sources = engine._build_context(chunks)
    assert "[1] test3.pdf" in context
    assert "Page" not in context

def test_build_prompt():
    engine = QueryEngine(None, None)
    question = "What is the photoelectric effect?"
    context = "[1] doc.pdf, Intro, Page 1\nThe photoelectric effect shows..."
    sources = [{"index": 1, "doc_name": "doc.pdf", "page": 1, "section": "Intro"}]
    
    prompt = engine.build_prompt(question, context, sources)
    assert question in prompt
    assert "[1] doc.pdf, Intro, Page 1" in prompt
    assert "cite sources inline" in prompt.lower() or "[index]" in prompt

def test_parse_cited_response():
    engine = QueryEngine(None, None)
    response = "The effect [1] was discovered by Einstein [2] and explained in doc [1]."
    parsed, indices = engine.parse_cited_response(response)
    assert parsed == response
    assert sorted(indices) == [1, 2]
