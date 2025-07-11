# make_predictions.py
"""
Script to generate RAG predictions for each question in training_data.csv and save to predictions.json
"""
import os
import json
import time
import pandas as pd
import openai
from dotenv import load_dotenv
from openai.error import APIError, ServiceUnavailableError, RateLimitError
from agents.retrieval_agent import RetrievalAgent


def load_api_key():
    """
    Load OpenAI API key from environment or .env
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("🛑 Please set OPENAI_API_KEY in your environment or .env file")
    openai.api_key = key


def safe_chat_completion(**kwargs):
    """
    Retry ChatCompletion up to 3 times on transient errors.
    """
    last_error = None
    for attempt in range(3):
        try:
            return openai.ChatCompletion.create(**kwargs)
        except (APIError, ServiceUnavailableError, RateLimitError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"⚠️ API error on attempt {attempt+1}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    # final attempt, let exception propagate if it fails
    return openai.ChatCompletion.create(**kwargs)


def main():
    # Load API key
    load_api_key()
    # Instantiate retriever
    retriever = RetrievalAgent(txt_dir="data_cleaned", json_dir="data")

    # Load Q&A pairs
    csv_path = os.path.join("data_cleaned", "training_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"training_data.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # Generate predictions
    results = []
    for idx, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        # Retrieve top-5 passages
        hits = retriever.search(question, top_k=5)
        snippets = [snippet for _, _, snippet in hits]
        context = "\n\n".join(snippets)

        # Build prompts
        system_prompt = (
            "You are a knowledgeable WW1 historian assistant. "
            "Answer concisely, cite entries, and ground your answer in the provided context."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        # Generate via OpenAI ChatCompletion
        resp = safe_chat_completion(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"[{idx}] Generated: {answer[:80]}...")

        results.append({
            "id": idx,
            "question": question,
            "response": answer
        })

    # Save to JSON
    out_path = "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved {len(results)} predictions to {out_path}")

if __name__ == "__main__":
    main()
