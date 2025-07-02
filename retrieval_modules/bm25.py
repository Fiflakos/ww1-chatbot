# retrieval_modules/bm25.py

import os
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, texts, file_names):
        """
        texts: list of raw document strings
        file_names: parallel list of document file paths or IDs
        """
        # 1) check that we actually got documents
        if not texts or not file_names or len(texts) != len(file_names):
            raise ValueError(
                "BM25Retriever initialization error: "
                "no documents loaded (texts and file_names lists are empty or mismatched). "
                "Check your corpus_dir path."
            )

        # 2) filter out any truly empty docs
        non_empty = [
            (t, fn) for t, fn in zip(texts, file_names)
            if isinstance(t, str) and t.strip()
        ]
        if not non_empty:
            raise ValueError(
                "BM25Retriever initialization error: "
                "all loaded documents are empty. "
                "Nothing to index."
            )
        filtered_texts, filtered_names = zip(*non_empty)

        # 3) tokenize
        tokenized = [word_tokenize(doc.lower()) for doc in filtered_texts]
        if not tokenized:
            raise ValueError(
                "BM25Retriever initialization error: "
                "tokenization produced zero documents."
            )

        # 4) build the BM25 index
        self.bm25 = BM25Okapi(tokenized)
        self.file_names = list(filtered_names)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Returns a list of (file_name, score, snippet) for the top_k hits.
        """
        if not query or not query.strip():
            return []

        tokens = word_tokenize(query.lower())
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        # pair up and sort
        pairs = sorted(
            zip(self.file_names, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for fn, score in pairs:
            # grab a brief snippet
            try:
                text = open(fn, encoding="utf-8").read()
                snippet = text.replace("\n", " ").strip()[:200]
            except Exception:
                snippet = ""
            results.append((fn, float(score), snippet))
        return results
