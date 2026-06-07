import re
from typing import List, Optional, Dict, Any, Tuple
from src.services.vector_store import VectorStore
from src.services.bm25_index import BM25IndexService

def reciprocal_rank_fusion(vector_results: List[Dict], bm25_results: List[str], k: int = 60) -> Dict[str, float]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Each result at rank r contributes 1 / (k + r + 1) to its score.
    Results appearing in both lists accumulate higher scores.
    """
    scores = {}

    for rank, result in enumerate(vector_results):
        doc_id = result["chroma_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    return scores

class QueryEngine:
    """Orchestrates vector + keyword search with RRF fusion."""

    def __init__(self, vector_store: VectorStore, bm25_service: BM25IndexService):
        self.vector_store = vector_store
        self.bm25_service = bm25_service

    async def query(self, user_id: str, question: str,
                    doc_ids: Optional[List[str]] = None,
                    subject: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    top_k: int = 5) -> Tuple[str, List[Dict], List[Dict]]:
        """Full query pipeline.

        Returns: (context_string, sources_list, chunks)
        """
        # Step 1: Vector search
        vector_results = self.vector_store.query(
            user_id, question,
            doc_ids=doc_ids, subject=subject, tags=tags,
            top_k=20
        )

        # Step 2: BM25 search
        bm25_results = self.bm25_service.search(
            user_id, question, top_k=20,
            doc_ids=doc_ids, subject=subject, tags=tags
        )

        # Step 3: RRF fusion
        fused_scores = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        top_chroma_ids = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True
        )[:top_k]

        # Step 4: Fetch full metadata from data_store
        from src.core.data_store import get_chunks_by_chroma_ids
        chunks = await get_chunks_by_chroma_ids(top_chroma_ids)

        # Step 5: Build context
        context, sources = self._build_context(chunks)

        return context, sources, chunks

    def _build_context(self, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """Build context string with inline citations."""
        context_parts = []
        sources = []

        for i, chunk in enumerate(chunks, 1):
            citation = f"[{i}] {chunk['doc_name']}"
            if chunk.get("section"):
                citation += f", {chunk['section']}"
            if chunk.get("page"):
                citation += f", Page {chunk['page']}"
            elif chunk.get("locator"):
                citation += f", {chunk['locator']}"

            context_parts.append(f"{citation}\n{chunk['content']}")
            sources.append({
                "index": i,
                "doc_name": chunk["doc_name"],
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "locator": chunk.get("locator"),
                "chroma_id": chunk["chroma_id"]
            })

        return "\n\n---\n\n".join(context_parts), sources

    def build_prompt(self, question: str, context: str, sources: List[Dict]) -> str:
        """Build LLM prompt with citation requirements."""
        sources_list = "\n".join([
            f"[{s['index']}] {s['doc_name']}" +
            (f", {s['section']}" if s.get("section") else "") +
            (f", Page {s['page']}" if s.get("page") else f", {s['locator']}" if s.get("locator") else "")
            for s in sources
        ])

        return f"""You are an AI tutor helping a student prepare for exams. Answer the following question STRICTLY based on the provided document excerpts.

Rules:
- Start directly with the answer. No introductions like "Based on the documents..."
- Cite sources inline using [index] format. Example: "The photoelectric effect [1] demonstrates that light behaves as particles."
- Focus on key points important for exams. Break down complex ideas simply.
- If the context doesn't contain enough information, say: "I don't have enough information about that in your uploaded documents."
- Do not make up information not present in the excerpts.

Sources:
{sources_list}

Document Excerpts:
{context}

Question: {question}

Answer (with inline citations):"""

    def parse_cited_response(self, response: str) -> Tuple[str, List[int]]:
        """Extract which source indices were actually cited."""
        cited_indices = list(set(map(int, re.findall(r'\[(\d+)\]', response))))
        return response, cited_indices
