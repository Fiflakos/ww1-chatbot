# agents/retrieval_agent.py

import os
from pathlib import Path
from retrieval_modules.bm25 import BM25Retriever

class RetrievalAgent:
    """
    Walks your data_cleaned folder, loads all .txt’s,
    builds a BM25 index over them, and on search() returns
    the top filenames, scores and a 200-char snippet.
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        # find all .txt
        self.file_paths = sorted(self.corpus_dir.rglob("*.txt"))
        # relative filenames for display
        self.names = [str(p.relative_to(self.corpus_dir)) for p in self.file_paths]
        # load all texts
        self.texts = [p.read_text(encoding="utf-8") for p in self.file_paths]
        # BUILD BM25 with only texts
        self._bm25 = BM25Retriever(self.texts)

    def search(self, query: str, top_k: int = 5):
        # retrieve (index, score)
        hits = self._bm25.retrieve(query, top_k)
        out = []
        for idx, score in hits:
            txt = self.texts[idx]
            # single‐line snippet
            snippet = txt.replace("\n", " ")[:200].strip()
            fn = self.names[idx]
            out.append((fn, float(score), snippet))
        return out
