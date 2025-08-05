# agents/retrieval_agent.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os, glob
import json
import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet', quiet=True)
from rank_bm25 import BM25Okapi

class RetrievalAgent:
    def __init__(self, json_path: str):
        import json, nltk
        from nltk.tokenize import word_tokenize
        from rank_bm25 import BM25Okapi

        nltk.download("punkt", quiet=True)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Combine fields to form searchable text (fallback-safe)
        self.texts = [
            f"Context: {item.get('context', '')}\nQ: {item.get('query', '')}\nA: {item.get('response', '')}"
            for item in data
        ]
        #self.file_names = [item.get("id", f"doc_{i}") for i, item in enumerate(data)]

        self.source_ids = [item.get("source", f"doc_{i}") for i, item in enumerate(data)]
        self.ids = [item.get("id", f"qa_{i}") for i, item in enumerate(data)]


        # Tokenize and build BM25 index
        tokenized_corpus = [word_tokenize(text.lower()) for text in self.texts]
        if not tokenized_corpus:
            raise ValueError(f"No valid documents found in {json_path}")
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3):
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in ranked:
            score = float(scores[idx])
            snippet = self.texts[idx][:300].replace("\n", " ").strip()
            source = self.source_ids[idx]
            hits.append((source, score, snippet))
        return hits




# test_retrieval.py
from retrieval_modules.bm25 import BM25Retriever  # or RetrievalAgent if using that version

def main():
    json_path = "data/annotated2_ww1_qa.json"  # ✅ Replace with actual path to your dataset
    query = "What was life like in the trenches?"

    retriever = BM25Retriever(json_path)  # or RetrievalAgent(json_path)
    results = retriever.search(query, top_k=3)  # or .search(query) if using RetrievalAgent

    print(f"\nTop results for query: \"{query}\"\n")
    for i, (doc_id, score, snippet) in enumerate(results, 1):
        print(f"Rank {i}:")
        print(f"Document ID: {doc_id}")
        print(f"BM25 Score: {score:.4f}")
        print(f"Snippet: {snippet}")
        print("-" * 80)

if __name__ == "__main__":
    main()
