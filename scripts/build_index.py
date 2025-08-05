# scripts/build_index.py

import os
import json
import pickle
from retrieval_modules.bm25 import BM25Retriever

if __name__ == "__main__":
    json_path  = "data/annotated2_ww1_qa.json"
    out_path    = os.path.join("logs", "bm25_index.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    retriever = BM25Retriever(json_path)
    with open(out_path, "wb") as f:
        pickle.dump(retriever, f)

    print(f"✅ Pre-built BM25 index at {out_path}")
