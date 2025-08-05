# scripts/build_vectorstore.py
from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import json

JSON_PATH = "data/annotated2_ww1_qa.json"
INDEX_DIR = Path("vectorstore/faiss_index")

def load_corpus_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item.get("text", "") for item in data]
    metadatas = [{"source": item.get("id", f"doc_{i}")} for i, item in enumerate(data)]
    return texts, metadatas

def main():
    texts, metadatas = load_corpus_from_json(JSON_PATH)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # build the FAISS index
    db = FAISS.from_texts(texts, embed, metadatas=metadatas)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print(f"✅ Built and saved FAISS index under {INDEX_DIR}")


if __name__ == "__main__":
    main()
