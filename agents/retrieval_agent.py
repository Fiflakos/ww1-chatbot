# agents/retrieval_agent.py

import os
from retrieval_modules.bm25 import BM25Retriever

class RetrievalAgent:
    def __init__(self, corpus_dir: str):
        self.texts = []
        self.names = []

        # walk through corpus_dir and load every .txt
        for root, _, files in os.walk(corpus_dir):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    path = os.path.join(root, fn)
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()
                    self.texts.append(txt)
                    # store relative path as ID
                    rel = os.path.relpath(path, corpus_dir)
                    self.names.append(rel)

        # build BM25 index
        self._bm25 = BM25Retriever(self.texts, self.names)

    def search(self, query: str, top_k: int = 5):
        return self._bm25.retrieve(query, top_k)
