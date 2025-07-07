# scripts/build_vectorstore.py
from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

DATA_DIR = "data_cleaned"
INDEX_DIR = Path("vectorstore/faiss_index")

def load_corpus():
    texts = []
    metadatas = []
    for txt in Path(DATA_DIR).rglob("*.txt"):
        text = txt.read_text(encoding="utf-8")
        # store filename in metadata so we can surface it in the UI
        metadatas.append({"source": txt.name})
        texts.append(text)
    return texts, metadatas

def main():
    texts, metadatas = load_corpus()
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # build the FAISS index
    db = FAISS.from_texts(texts, embed, metadatas=metadatas)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print(f"✅ Built and saved FAISS index under {INDEX_DIR}")

if __name__ == "__main__":
    main()
