# scripts/qa_pipeline/qa_pipeline.py

import os, uuid, json
import nltk, openai, numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# 0) CONFIGURATION passed in via env or function args:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")       # you set this before running
INPUT_DIR        = "data_cleaned"                  # or wherever your txt lives
OUTPUT_FILE      = "output/annotated_ww1_qa.json"

openai.api_key = OPENAI_API_KEY

nltk.download('punkt')

# === STEP 1: Load and preprocess TXT files ===
def load_documents(data_path):
    docs = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                docs.append(f.read().strip())
    return docs

# === and so on… just paste in your chunk_text, clustering, QA generation, annotation, evaluation, and run_pipeline() ===
