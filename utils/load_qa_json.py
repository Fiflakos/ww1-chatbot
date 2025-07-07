import json

def load_qa_pairs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # This depends on the exact structure, but assuming:
    # [{"question": "...", "answer": "...", "source_id": "...", ...}, ...]
    return data
