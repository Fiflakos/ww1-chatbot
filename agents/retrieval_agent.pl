# agents/retrieval_agent.py
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class RetrievalAgent:
    def __init__(self, corpus_dir: str):
        self.file_paths = []
        self.texts = []
        for root, _, files in os.walk(corpus_dir):
            for fn in sorted(files):
                if fn.lower().endswith(".txt"):
                    path = os.path.join(root, fn)
                    self.file_paths.append(path.replace("\\","/"))
                    with open(path, "r", encoding="utf-8") as f:
                        self.texts.append(f.read())
        # vectorize
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_term_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 5):
        q_vec = self.vectorizer.transform([query])
        sims = (self.doc_term_matrix @ q_vec.T).toarray().ravel()
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            score = float(sims[i])
            snippet = self.texts[i][:200].replace("\n"," ").strip()
            fn = self.file_paths[i].split("data_cleaned/")[-1]
            results.append((fn, score, snippet))
        return results
