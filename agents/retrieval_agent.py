import os, glob, json
from typing import List, Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


class RetrievalAgent:
    """
    Unified retriever over:
      - TXT files under txt_dir
      - JSON files under json_dir (uses 'context' + 'response' if present, else 'context' or 'text')

    Returns tuples: (file_name, score, snippet)
    """
    def __init__(self, txt_dir: str = None, json_dir: str = None):

        nltk.download("punkt", quiet=True)
        try:

            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

        self.file_names: List[str] = []
        self.texts: List[str] = []

        if txt_dir and os.path.isdir(txt_dir):
            for p in sorted(glob.glob(os.path.join(txt_dir, "**", "*.txt"), recursive=True)):
                try:
                    with open(p, encoding="utf-8", errors="ignore") as f:
                        t = f.read()
                    self.file_names.append(os.path.relpath(p, txt_dir))
                    self.texts.append(t)
                except Exception:
                    continue

        if json_dir and os.path.isdir(json_dir):
            for p in sorted(glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)):
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                items = None
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # common shapes: {"items": [...]}, {"data":[...]}, or a single object
                    items = data.get("items") or data.get("data")
                    if items is None:
                        items = [data]

                if not items:
                    continue

                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    text = None
                    # Prefer context + response (QA-style), then context, then generic text fields
                    if item.get("context") and item.get("response"):
                        text = f"{item['context']}\n\n{item['response']}"
                    elif item.get("context"):
                        text = str(item["context"])
                    elif item.get("text"):
                        text = str(item["text"])
                    elif item.get("content"):
                        text = str(item["content"])

                    if not text:
                        continue

                    name = f"{os.path.relpath(p, json_dir)}#item{i}"
                    self.file_names.append(name)
                    self.texts.append(text)

        if not self.texts:
            raise ValueError("No documents found; check txt_dir/json_dir paths.")

        # Build BM25 index
        tokenized_corpus = [word_tokenize(t.lower()) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_contains: Optional[List[str]] = None
    ) -> List[Tuple[str, float, str]]:
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits: List[Tuple[str, float, str]] = []
        for idx in ranked:
            fname = self.file_names[idx]
            if filter_contains:
                low = fname.lower()
                if not any(key.lower() in low for key in filter_contains):
                    continue
            score = float(scores[idx])
            snippet = (self.texts[idx] or "")[:500].replace("\n", " ").strip()
            hits.append((fname, score, snippet))
            if len(hits) >= top_k:
                break
        return hits
