import pytest
from src.services.document_processor import detect_doc_type, chunk_document


def test_detect_doc_type():
    assert detect_doc_type("file.pdf") == "pdf"
    assert detect_doc_type("file.txt") == "txt"
    assert detect_doc_type("file.md") == "md"
    assert detect_doc_type("file.docx") == "docx"
    assert detect_doc_type("file.png") == "image"
    assert detect_doc_type("file.jpg") == "image"
    assert detect_doc_type("file.unknown") == "unknown"


def test_chunk_document_short():
    text = "This is paragraph one.\n\nThis is paragraph two."
    chunks = chunk_document(text, doc_type="txt", headings=[])
    assert len(chunks) == 2
    assert chunks[0]["content"] == "This is paragraph one."
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
