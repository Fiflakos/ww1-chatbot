import os
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
import json

# Setup
BASE_DIR = Path("data_cleaned").resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Chatbot response with retrieval context
from interface_agent import get_chatbot_answer_and_context

# Metrics
from rouge_score import rouge_scorer
import bert_score

# RAGAS
from ragas.schema import Record, Dataset as RagasDataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

# -------- CONFIG --------
DATA_FOLDER = BASE_DIR / "data_cleaned"
OUTPUT_RESULTS = BASE_DIR / "results" / "folder_eval_with_ragas.json"
NUM_FILES = 100  # Set to None for all files


# ------------------------

def load_text_files(root_folder):
    text_files = []
    for subdir in ["Letters", "Diaries", "Others"]:
        full_path = root_folder / subdir
        text_files.extend(glob(str(full_path / "*.txt")))
    return text_files[:NUM_FILES] if NUM_FILES else text_files


def extract_question_and_reference(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    lines = text.split("\n")
    if len(lines) > 1:
        question = lines[0]
        reference = "\n".join(lines[1:]).strip()
    else:
        question = text
        reference = text
    return question, reference


def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores)


def compute_bertscore(preds, refs):
    P, R, F1 = bert_score.score(preds, refs, lang="en", rescale_with_baseline=True)
    return float(F1.mean())


def main():
    text_files = load_text_files(DATA_FOLDER)
    print(f"[INFO] Found {len(text_files)} files.")

    predictions = []
    references = []
    questions = []
    contexts_used = []
    ragas_records = []

    for path in tqdm(text_files, desc="Evaluating"):
        question, reference = extract_question_and_reference(path)

        try:
            answer, contexts = answer(question)
        except Exception as e:
            print(f"[ERROR] Chatbot failed for {path.name}: {e}")
            answer, contexts = "", []

        predictions.append(answer)
        references.append(reference)
        questions.append(question)
        contexts_used.append(contexts)

        # Prepare RAGAS record
        ragas_records.append(Record(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=reference
        ))

    rouge_l = compute_rouge_l(predictions, references)
    bertscore_f1 = compute_bertscore(predictions, references)

    ragas_dataset = RagasDataset.from_records(ragas_records)
    ragas_scores = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    ).dict()

    results = {
        "num_samples": len(predictions),
        "rougeL": rouge_l,
        "bertscore": bertscore_f1,
        "ragas": ragas_scores,
        "samples": [
            {
                "file": str(path),
                "question": q,
                "reference": r,
                "prediction": p,
                "contexts": c
            }
            for path, q, r, p, c in zip(text_files, questions, references, predictions, contexts_used)
        ]
    }

    os.makedirs(OUTPUT_RESULTS.parent, exist_ok=True)
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete. Saved to {OUTPUT_RESULTS}")
    print(f"ROUGE-L: {rouge_l:.4f} | BERTScore: {bertscore_f1:.4f}")
    print(f"RAGAS Scores:\n" + json.dumps(ragas_scores, indent=2))


if __name__ == "__main__":
    main()
