import hashlib
import os
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.core.config import CHROMA_DB_PATH

class VectorStore:
    """Manages ChromaDB collections per user for vector search."""
    
    _instance = None
    _client = None
    _embedding_model = None
    
    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            path = db_path or CHROMA_DB_PATH
            os.makedirs(path, exist_ok=True)
            cls._client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False)
            )
        return cls._instance
    
    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedding_model
    
    @staticmethod
    def get_collection_name(user_id: str) -> str:
        """Generate a unique, deterministic collection name from user_id."""
        hashed = hashlib.sha256(user_id.encode()).hexdigest()[:32]
        return f"user_{hashed}"
    
    def get_or_create_collection(self, user_id: str):
        """Get or create a user's ChromaDB collection."""
        collection_name = self.get_collection_name(user_id)
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"user_id": user_id}
        )
    
    def add_chunks(self, user_id: str, chunks: List[Dict[str, Any]]):
        """Add document chunks to a user's collection.
        
        Args:
            chunks: List of dicts with keys: chroma_id, embedding, content, 
                    doc_id, doc_name, page, section, chunk_index, subject, tags
        """
        if not chunks:
            return
        
        collection = self.get_or_create_collection(user_id)
        
        ids = [c["chroma_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "doc_id": c["doc_id"],
                "doc_name": c["doc_name"],
                "chunk_index": c["chunk_index"],
            }
            for key in ("page", "section", "subject"):
                val = c.get(key)
                if val is not None:
                    meta[key] = val
            tags = c.get("tags")
            if tags:
                meta["tags"] = ",".join(tags)
            metadatas.append(meta)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def query(self, user_id: str, question: str, 
              doc_ids: Optional[List[str]] = None,
              subject: Optional[str] = None,
              tags: Optional[List[str]] = None,
              top_k: int = 20) -> List[Dict[str, Any]]:
        """Query a user's collection with optional filters."""
        collection = self.get_or_create_collection(user_id)
        model = self.get_embedding_model()
        embedding = model.encode(question).tolist()
        
        # Build where clause
        where_clause = {}
        if doc_ids:
            if len(doc_ids) == 1:
                where_clause["doc_id"] = doc_ids[0]
            else:
                where_clause["doc_id"] = {"$in": doc_ids}
        if subject:
            where_clause["subject"] = subject
        if tags:
            if len(tags) == 1:
                where_clause["tags"] = tags[0]
            else:
                where_clause["tags"] = {"$in": tags}
        
        where_filter = where_clause if where_clause else None
        
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chroma_id": results["ids"][0][i],
                "score": 1 - results["distances"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        return formatted
    
    def delete_document_chunks(self, user_id: str, doc_id: str):
        """Delete all chunks for a specific document."""
        collection = self.get_or_create_collection(user_id)
        collection.delete(where={"doc_id": doc_id})
