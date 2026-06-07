import pytest
from src.services.vector_store import VectorStore

def test_get_collection_name():
    store = VectorStore("./test_chroma")
    name = store.get_collection_name("user@example.com")
    assert name.startswith("user_")
    assert len(name) == 37  # "user_" + 32 hex chars
    assert name != store.get_collection_name("user_example.com")  # different inputs, different hashes

def test_get_collection_name_same_input():
    store = VectorStore("./test_chroma")
    name1 = store.get_collection_name("user@example.com")
    name2 = store.get_collection_name("user@example.com")
    assert name1 == name2  # stable hash
