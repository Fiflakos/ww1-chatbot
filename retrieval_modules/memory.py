# retrieval_modules/memory.py

from typing import List, Dict

class MemoryRetriever:
    """
    A very simple in‐memory “retriever”:
    - store(query) saves past queries
    - retrieve(top_k) returns the last top_k queries as zero-score hits
    """

    def __init__(self) -> None:
        self._history: List[str] = []

    def store(self, query: str) -> None:
        """Save a user query into memory."""
        self._history.append(query)

    def retrieve(self, top_k: int = 5) -> List[Dict]:
        """
        Return up to the last `top_k` queries as pseudo‐documents.
        Each hit is a dict with keys: id, score, text.
        """
        hits = []
        # take the last top_k items, in reverse (most recent first)
        for q in self._history[-top_k:][::-1]:
            hits.append({
                "id":    f"[Memory] {q}",
                "score": 0.0,
                "text":  q
            })
        return hits
