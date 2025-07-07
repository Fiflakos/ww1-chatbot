import json
from pathlib import Path

class JsonRAG:
    def __init__(self, json_path):
        self.entries = []
        with open(json_path, 'r') as f:
            self.entries = json.load(f)

    def search(self, query, top_k=3):
        # Naive: rank by # of query word matches in context + response
        results = []
        q_words = set(query.lower().split())
        for entry in self.entries:
            # Boost if query words in question, response, or context
            content = (
                entry["query"].lower() + " " +
                entry["response"].lower() + " " +
                entry["context"].lower()
            )
            score = sum(w in content for w in q_words)
            if score > 0:
                results.append((score, entry))
        results.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in results[:top_k]]
