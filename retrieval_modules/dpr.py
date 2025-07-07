# retrieval_modules/dpr.py

import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class DPRRetriever:
    def __init__(self, texts, file_names, top_k=5):
        if len(texts) != len(file_names):
            raise ValueError("texts and file_names must be the same length")
        self.texts = texts
        self.file_names = file_names
        self.top_k = top_k

        # Load the encoder and build the FAISS index
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float, str]]:
        k = top_k or self.top_k
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        # Search
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            fname = self.file_names[idx]
            snippet = self.texts[idx]
            results.append((fname, float(score), snippet))
        return results
