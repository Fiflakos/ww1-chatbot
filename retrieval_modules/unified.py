import json
from pathlib import Path
from typing import List, Dict

class UnifiedRetriever:
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.qa_data = json.load(f)

    def retrieve(self, query: str) -> List[Dict]:
        # Naive keyword match (replace with embedding or BM25 later)
        results = []
        for qa in self.qa_data:
            if any(word in qa['question'].lower() for word in query.lower().split()):
                results.append({
                    "question": qa["question"],
                    "answer": qa["answers"][0] if qa["answers"] else "No answer available.",
                    "context": qa["contexts"][0] if qa["contexts"] else "",
                    "source": qa["meta"].get("source", "unknown")
                })
        return results[:5]  # return top 5 hits
