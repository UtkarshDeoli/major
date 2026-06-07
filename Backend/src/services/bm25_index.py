import re
import numpy as np
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi


class BM25IndexService:
    """Per-user BM25 keyword search index."""

    def __init__(self):
        self.indexes: Dict[str, Optional[BM25Okapi]] = {}
        self.chunk_maps: Dict[str, Dict[int, str]] = {}
        self.chunk_metadata: Dict[str, Dict[str, Dict]] = {}

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def build_index(self, user_id: str, chunks: List[Dict]):
        """Build BM25 index for a user from chunks.

        Args:
            chunks: List of dicts with keys: chroma_id, content, doc_id, subject, tags
        """
        if not chunks:
            self.indexes[user_id] = None
            self.chunk_maps[user_id] = {}
            self.chunk_metadata[user_id] = {}
            return

        tokenized_corpus = [self.tokenize(c["content"]) for c in chunks]
        self.indexes[user_id] = BM25Okapi(tokenized_corpus)
        self.chunk_maps[user_id] = {i: c["chroma_id"] for i, c in enumerate(chunks)}

        self.chunk_metadata[user_id] = {
            c["chroma_id"]: {
                "doc_id": c["doc_id"],
                "subject": c.get("subject"),
                "tags": c.get("tags", []),
            }
            for c in chunks
        }

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 20,
        doc_ids: Optional[List[str]] = None,
        subject: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Search BM25 index with optional filters.

        Returns: List of chroma_ids.
        """
        if user_id not in self.indexes or self.indexes[user_id] is None:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.indexes[user_id].get_scores(tokenized_query)

        # Apply filters by zeroing excluded chunks
        for i, chroma_id in enumerate(self.chunk_maps[user_id].values()):
            chunk = self.chunk_metadata[user_id].get(chroma_id)
            if not chunk:
                scores[i] = 0
                continue
            if doc_ids and chunk["doc_id"] not in doc_ids:
                scores[i] = 0
            if subject and chunk.get("subject") != subject:
                scores[i] = 0
            if tags and not any(t in chunk.get("tags", []) for t in tags):
                scores[i] = 0

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.chunk_maps[user_id][i] for i in top_indices if scores[i] > 0]
