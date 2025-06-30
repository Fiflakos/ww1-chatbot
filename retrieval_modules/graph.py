# retrieval_modules/graph.py

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GraphRetriever:
    """
    A “graph” retriever implemented via TF-IDF + cosine similarity.
    """

    def __init__(self, documents: List[Dict], top_k: int = 5):
        """
        documents: list of {"id": ..., "text": ...}
        top_k: how many hits to return on each query
        """
        self.top_k = top_k
        self.ids = [doc["id"] for doc in documents]
        self.texts = [doc["text"] for doc in documents]

        # Build TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str) -> List[Dict]:
        """
        Returns a list of up to top_k dicts:
          {"id": <doc id>, "score": <cosine sim float>, "text": <doc text>}
        """
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idxs = sims.argsort()[::-1][: self.top_k]

        results = []
        for idx in top_idxs:
            results.append({
                "id":    self.ids[idx],
                "score": float(sims[idx]),
                "text":  self.texts[idx]
            })
        return results
