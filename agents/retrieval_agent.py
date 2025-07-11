# retrieval_agent.py

import os
import glob
import json
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

class RetrievalAgent:
    def __init__(
        self,
        txt_dir: str = None,
        json_dir: str = None,
        csv_path: str = None,
        csv_text_col: str = 'ontext',
    ):
        """
        Unified RetrievalAgent that ingests multiple data sources:
        - Plain text files under txt_dir
        - JSON files under json_dir (extracts 'context' field)
        - CSV file at csv_path (extracts text from csv_text_col)
        """
        # Ensure tokenizer is available
        nltk.download("punkt", quiet=True)

        self.texts = []
        self.file_names = []

        # 1. Load .txt files
        if txt_dir:
            for path in sorted(glob.glob(os.path.join(txt_dir, '**', '*.txt'), recursive=True)):
                try:
                    with open(path, encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    rel = os.path.relpath(path, txt_dir)
                    self.texts.append(text)
                    self.file_names.append(f"txt/{rel}")
                except Exception:
                    continue

        # 2. Load JSON files
        if json_dir:
            for jf in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
                with open(jf, encoding='utf-8') as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    ctx = item.get('context') or item.get('response') or ''
                    if not ctx:
                        continue
                    doc_id = item.get('id', '')
                    name = f"json/{os.path.basename(jf)}:{doc_id}"
                    self.texts.append(ctx)
                    self.file_names.append(name)

        # 3. Load CSV file
        if csv_path:
            df = pd.read_csv(csv_path)
            if csv_text_col not in df.columns:
                raise ValueError(f"CSV column '{csv_text_col}' not found")
            for idx, row in df.iterrows():
                text = str(row[csv_text_col])
                name = f"csv/{idx}"
                self.texts.append(text)
                self.file_names.append(name)

        if not self.texts:
            raise ValueError("No documents found in provided sources")

        # 4. Build BM25 index
        tokenized_corpus = [word_tokenize(t.lower()) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        """
        Retrieve top_k documents for query across all sources.
        Returns list of (identifier, score, snippet).
        """
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in top_idxs:
            score = float(scores[idx])
            if score <= 0:
                continue
            snippet = self.texts[idx][:300].replace("\n", " ").strip()
            hits.append((self.file_names[idx], score, snippet))
        return hits
