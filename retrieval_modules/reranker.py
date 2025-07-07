# --- retrieval_modules/reranker.py ---
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # only model_name is needed here
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[tuple[str, float, str]], top_k: int = 5):
        # hits = list of (filename, score, snippet)
        pairs = [(query, snippet) for _, _, snippet in hits]
        scores = self.cross_encoder.predict(pairs)
        # merge back, sort by final score
        reranked = sorted(
            zip(hits, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [(fn, sc, sn) for (fn, _, sn), sc in reranked[:top_k]]
