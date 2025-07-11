# evaluation.py
"""
Evaluation script for RAG pipeline outputs:
  - Computes ROUGE-L, OpenAI-embedding cosine similarity, and RAGAS metrics
  - Supports two modes:
      * JSON mode: uses provided prediction and reference JSON files
      * Folder mode: reads training_data.csv, generates predictions via RAG over data_cleaned, and uses CSV's answers as references

Usage:
  # JSON evaluation
  python evaluation.py --mode json --preds data/annotated2_ww1_qa.json --refs data/annotated200_ww1_qa.json --out results_json.csv

  # Folder-mode evaluation (generation + metrics)
  python evaluation.py --mode folder --out results_cleaned.csv
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from openai.error import APIError, ServiceUnavailableError, RateLimitError
from agents.retrieval_agent import RetrievalAgent

# ---- Load OpenAI API Key ----
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("🛑 Please set OPENAI_API_KEY in your environment or .env file")

# ---- Instantiate Retriever ----
retriever = RetrievalAgent(txt_dir="data_cleaned", json_dir="data")

# ---- Safe Chat Completion with Retry ----
def safe_chat_completion(**kwargs):
    last_error = None
    for attempt in range(3):
        try:
            return openai.ChatCompletion.create(**kwargs)
        except (APIError, ServiceUnavailableError, RateLimitError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"⚠️ OpenAI error on attempt {attempt+1}: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    # final attempt
    return openai.ChatCompletion.create(**kwargs)

# ---- Data Loaders ----

def load_json_mode(pred_path, ref_path):
    # Load predictions
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    # Load references (JSON or CSV)
    refs = []
    if ref_path.lower().endswith('.csv'):
        df = pd.read_csv(ref_path)
        if 'answer' not in df.columns:
            raise ValueError(f"CSV at {ref_path} must contain an 'answer' column.")
        for idx, row in df.iterrows():
            refs.append({'id': idx, 'reference': str(row['answer']).strip()})
    else:
        with open(ref_path, 'r', encoding='utf-8') as f:
            refs = json.load(f)
    # Pair preds with refs
    examples = []
    for p in preds:
        ex_id = p.get('id')
        pred_text = p.get('response') or p.get('prediction', '')
        ref_item = next((r for r in refs if str(r.get('id')) == str(ex_id)), {})
        ref_text = ref_item.get('response') or ref_item.get('reference', '')
        examples.append({'id': ex_id, 'pred': pred_text, 'ref': ref_text, 'context': ''})
    return examples


def load_folder_mode(_unused=None):
    # Read CSV of Q&A
    csv_path = os.path.join('data_cleaned', 'training_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"training_data.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    examples = []
    for idx, row in df.iterrows():
        q = str(row.get('question', '')).strip()
        ref = str(row.get('answer', '')).strip()
        # Retrieve context snippets
        hits = retriever.search(q, top_k=5)
        snippets = [sn for _, _, sn in hits]
        context = "\n\n".join(snippets)
        # Generate prediction
        system_prompt = (
            "You are a knowledgeable WW1 historian assistant. "
            "Answer concisely, cite entries, and ground your answer in the provided context."
        )
        user_prompt = (
  f"Context:\n{context}\n\n"
  f"Question: {q}\n\n"
  "Give me one concrete detail mentioned in this letter, and cite which passage it came from."
)
        resp = safe_chat_completion(
            model='gpt-4',
            messages=[
                {'role':'system','content':system_prompt},
                {'role':'user','content':user_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        pred = resp.choices[0].message.content.strip()
        examples.append({'id': idx, 'pred': pred, 'ref': ref, 'context': context})
    return examples

# ---- Metrics ----

def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]


def compute_embedding_cosine(preds, refs, model='text-embedding-ada-002'):
    sims = []
    for p_text, r_text in zip(preds, refs):
        emb_p = openai.Embedding.create(model=model, input=p_text)['data'][0]['embedding']
        emb_r = openai.Embedding.create(model=model, input=r_text)['data'][0]['embedding']
        p_arr = np.array(emb_p, dtype=np.float32)
        r_arr = np.array(emb_r, dtype=np.float32)
        sims.append(float(np.dot(p_arr, r_arr) / (np.linalg.norm(p_arr) * np.linalg.norm(r_arr))))
    return sims


def compute_ragas(examples):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    faithfulness, relevancy, correctness, cprec, crec = [], [], [], [], []
    import re
    for ex in examples:
        pred, ref, ctx = ex['pred'], ex['ref'], ex.get('context','')
        # surface overlap
        r_ref = scorer.score(ref, pred)['rougeL'].fmeasure
        r_ctxp = scorer.score(ctx, pred)['rougeL'].fmeasure
        r_ctxr = scorer.score(pred, ctx)['rougeL'].fmeasure
        # named-entity overlap
        ents_pred = set(re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', pred))
        ents_ctx = set(re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', ctx))
        faith = len(ents_pred & ents_ctx) / len(ents_pred) if ents_pred else 0.0
        faithfulness.append(faith)
        relevancy.append(r_ref)
        correctness.append(r_ref)
        cprec.append(r_ctxp)
        crec.append(r_ctxr)
    return {
        'faithfulness': faithfulness,
        'answer_relevancy': relevancy,
        'answer_correctness': correctness,
        'context_precision': cprec,
        'context_recall': crec
    }

# ---- Runner ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['json','folder'], required=True)
    parser.add_argument('--preds', help='Predictions JSON (json mode)')
    parser.add_argument('--refs', help='References JSON/CSV (json mode)')
    parser.add_argument('--out',  default='results.csv', help='Output CSV')
    args = parser.parse_args()

    if args.mode == 'json':
        examples = load_json_mode(args.preds, args.refs)
    else:
        examples = load_folder_mode()

    ids = [ex['id'] for ex in examples]
    preds = [ex['pred'] for ex in examples]
    refs  = [ex['ref'] for ex in examples]

    rouge_l = compute_rouge_l(preds, refs)
    emb_cos = compute_embedding_cosine(preds, refs)
    ragas   = compute_ragas(examples)

    df = pd.DataFrame({
        'id': ids,
        'rougeL_f1': rouge_l,
        'emb_cosine': emb_cos,
        **ragas
    })
    df.to_csv(args.out, index=False)
    print(f"✅ Saved evaluation results to {args.out}")

if __name__ == '__main__':
    main()
