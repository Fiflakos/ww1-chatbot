# scripts/build_vectorstore.py
import os
from glob import glob
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = "data_cleaned"
INDEX_DIR = "vectorstore/faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# 1. Load all your docs
paths = glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
texts = []
metadatas = []
for p in paths:
    with open(p, encoding="utf-8") as f:
        texts.append(f.read())
    # store the relative filename so we can refer back
    metadatas.append({"source": os.path.relpath(p, DATA_DIR)})

# 2. Embed them
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = embedder.encode(texts, convert_to_numpy=True)

# 3. Build FAISS index
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

# 4. Save metadata and raw texts
with open(os.path.join(INDEX_DIR, "metadatas.pkl"), "wb") as f:
    pickle.dump(metadatas, f)
with open(os.path.join(INDEX_DIR, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

print("✅ Built FAISS index at", INDEX_DIR)
