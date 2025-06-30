# retrieval_modules/bm25.py

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    Wraps the rank_bm25 BM25Okapi for a static corpus of texts.
    """

    def __init__(self, texts):
        """
        :param texts: List[str]  — raw text of each document
        """
        # tokenize once
        tokenized = [word_tokenize(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.texts = texts

    def retrieve(self, query: str, top_k: int = 5):
        """
        :param query: the user’s query
        :param top_k: how many hits to return
        :returns: List[(doc_index, score)]
        """
        q_toks = word_tokenize(query.lower())
        scores = self.bm25.get_scores(q_toks)
        # pick top k
        top_ix = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in top_ix]
