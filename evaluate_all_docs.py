# evaluate_all_docs.py
"""
Evaluate every .txt document under data_cleaned/ by:
  - Retrieving context via BM25
  - Generating a summary answer via OpenAI
  - Computing ROUGE-L, embedding-cosine, and RAGAS metrics
  - Writing results incrementally to CSV so you can track progress via tail -f

To speed up and reduce overloads, this version uses gpt-3.5-turbo.
"""
import os, glob, csv, time
import argparse
import numpy as np
import pandas as pd
import openai
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from openai.error import APIError, ServiceUnavailableError, RateLimitError
from agents.retrieval_agent import RetrievalAgent

# ---- Config ----
MODEL_NAME = 'gpt-3.5-turbo'  # switched from gpt-4 for speed and reliability
OUTPUT_CSV = 'results_all_docs.csv'

# ---- Load API Key ----
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("🛑 Please set OPENAI_API_KEY in your environment or .env file.")

# ---- Initialize Retriever ----
retriever = RetrievalAgent(txt_dir='data_cleaned', json_dir='data')

# ---- Safe ChatCompletion ----
def safe_chat_completion(**kwargs):
    last_error = None
    for attempt in range(3):
        try:
            return openai.ChatCompletion.create(**kwargs)
        except (APIError, ServiceUnavailableError, RateLimitError) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"⚠️ API error on attempt {attempt+1}/{3} ({e}). Retrying in {wait}s...")
            time.sleep(wait)
    # final attempt
    return openai.ChatCompletion.create(**kwargs)

# ---- Metrics ----
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_rouge_l_single(pred, ref):
    return scorer.score(ref, pred)['rougeL'].fmeasure


def compute_embedding_cosine_single(pred, ref, model='text-embedding-ada-002'):
    emb_p = openai.Embedding.create(model=model, input=pred)['data'][0]['embedding']
    emb_r = openai.Embedding.create(model=model, input=ref)['data'][0]['embedding']
    p_arr, r_arr = np.array(emb_p), np.array(emb_r)
    return float(p_arr.dot(r_arr) / (np.linalg.norm(p_arr)*np.linalg.norm(r_arr)))

import re

def compute_ragas_single(pred, ref, ctx):
    # faithfulness: named-entity overlap
    ents_pred = set(re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", pred))
    ents_ctx  = set(re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", ctx))
    faith = len(ents_pred & ents_ctx)/len(ents_pred) if ents_pred else 0.0
    # relevancy/correctness: same as rougeL
    rel = compute_rouge_l_single(pred, ref)
    corr = rel
    # context precision & recall
    cp = scorer.score(ctx, pred)['rougeL'].fmeasure
    cr = scorer.score(pred, ctx)['rougeL'].fmeasure
    return faith, rel, corr, cp, cr

# ---- Runner ----
if __name__ == '__main__':
    # Prepare output CSV with header
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow([
            'id','pred','ref','context',
            'rougeL_f1','emb_cosine',
            'faithfulness','answer_relevancy','answer_correctness',
            'context_precision','context_recall'
        ])
        outf.flush()

        # Iterate documents
        for file_path in glob.glob('data_cleaned/**/*.txt', recursive=True):
            doc_id = os.path.relpath(file_path, 'data_cleaned')
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                full_ref = f.read().strip()

            question = f"What are the key points of '{doc_id}'?"
            hits = retriever.search(question, top_k=5)
            context = "\n\n".join(sn for _,_,sn in hits)

            resp = safe_chat_completion(
                model=MODEL_NAME,
                messages=[
                    {'role':'system','content':'You are a WW1 historian assistant.'},
                    {'role':'user','content':f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.7,
                max_tokens=300
            )
            pred = resp.choices[0].message.content.strip()

            # Compute metrics
            rL   = compute_rouge_l_single(pred, full_ref)
            eC   = compute_embedding_cosine_single(pred, full_ref)
            f,rv,crr,cp,cr = compute_ragas_single(pred, full_ref, context)

            # Write row
            writer.writerow([doc_id, pred, full_ref, context, rL, eC, f, rv, crr, cp, cr])
            outf.flush()
            print(f"✅ [{doc_id}] metrics written (ROUGE-L={rL:.3f}, emb_cos={eC:.3f})")
    print(f"🎉 All done. Results in {OUTPUT_CSV}")
