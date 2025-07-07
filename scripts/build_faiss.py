# scripts/build_faiss.py
import os
import pickle
import faiss
import numpy as np
from openai import OpenAIError, OpenAI
from pathlib import Path

# 1. Load your cleaned-text files
DATA_DIR = Path("data_cleaned")
paths = sorted(DATA_DIR.rglob("*.txt"))
texts = [p.read_text(encoding="utf-8") for p in paths]
names = [str(p.relative_to(DATA_DIR))     for p in paths]

# 2. Ask OpenAI for embeddings in batches
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

embeddings = []
batch_size = 100
for i in range(0, len(texts), batch_size):
    resp = openai.Embeddings.create(
        model="text-embedding-ada-002",
        input=texts[i : i + batch_size],
    )
    embeddings.extend([e["embedding"] for e in resp["data"]])

# 3. Build FAISS index
d = len(embeddings[0])
index = faiss.IndexFlatIP(d)
matrix = np.array(embeddings, dtype="float32")
faiss.normalize_L2(matrix)
index.add(matrix)

# 4. Persist index + metadata
os.makedirs("vectorstore/faiss_index", exist_ok=True)
faiss.write_index(index, "vectorstore/faiss_index/index.faiss")
with open("vectorstore/faiss_index/metadata.pkl", "wb") as f:
    pickle.dump((names, matrix), f)

print("✅ Built FAISS index at vectorstore/faiss_index/")
