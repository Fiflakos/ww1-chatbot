# retrieval_modules/dpr.py

import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class DPRRetriever:
    """
    Dense Passage Retriever backed by a Sentence-Transformer + FAISS index.
    Exposes:
      - __init__(documents, top_k=5)
      - retrieve(query) -> List[Dict]  # each dict has keys 'id', 'score', 'text'
    """

    def __init__(self, documents: List[Dict], top_k: int = 5):
        """
        documents: a list of dicts, each with at least:
           {'id': <unique-str>, 'text': <string>}
        top_k: how many passages to return per query
        """
        self.documents = documents
        self.top_k = top_k

        # load the Sentence-Transformer model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # encode all docs in a single shot
        texts = [doc["text"] for doc in documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        # build a simple flat L2 FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)  # add vectors to the index

    def retrieve(self, query: str) -> List[Dict]:
        """
        Encode the query, search the FAISS index, and return the top_k hits
        as a list of dicts: {'id', 'score', 'text'}.
        """
        # encode query
        q_emb = self.model.encode([query], convert_to_numpy=True)
        # search
        distances, indices = self.index.search(q_emb, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "score": float(dist),
                "text": doc["text"]
            })
        return results
