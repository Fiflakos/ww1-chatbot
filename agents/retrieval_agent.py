# agents/retrieval_agent.py

import os, glob
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

class RetrievalAgent:
    def __init__(self, corpus_dir: str):
        # ensure punkt is installed
        nltk.download("punkt", quiet=True)
        
        # load all .txt files
        self.file_paths = sorted(glob.glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True))
        self.file_names = [os.path.relpath(p, corpus_dir) for p in self.file_paths]
        self.texts = []
        for p in self.file_paths:
            with open(p, encoding="utf-8", errors="ignore") as f:
                self.texts.append(f.read())
        
        # build BM25
        tokenized_corpus = [word_tokenize(t.lower()) for t in self.texts]
        if not tokenized_corpus:
            raise ValueError(f"no documents found under {corpus_dir}")
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for idx in ranked:
            score = float(scores[idx])
            snippet = self.texts[idx][:300].replace("\n", " ").strip()
            hits.append((self.file_names[idx], score, snippet))
        return hits
