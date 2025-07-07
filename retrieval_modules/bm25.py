# retrieval_modules/bm25.py

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, texts: list[str], file_names: list[str]):
        self.texts = texts
        self.file_names = file_names

        # tokenize each document
        tokenized_corpus = [
            word_tokenize(doc.lower())
            for doc in texts
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in top_idxs:
            fname = self.file_names[idx]
            score = scores[idx]
            snippet = self.texts[idx][:200].replace("\n", " ").strip()
            hits.append((fname, score, snippet))
        return hits
