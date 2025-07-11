# generate_training_data.py
import os
import glob
import json
import pandas as pd

# 1) Crawl all .txt files under data_cleaned/
rows = []
for path in glob.glob("data_cleaned/**/*.txt", recursive=True):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\n", " ")
    # Use the first 200 characters as a context snippet
    ontext = text[:200].strip()
    question = "What is this document about?"
    answer = ontext  # here we simply use the snippet as the 'gold' answer
    rows.append({
        "ontext": ontext,
        "question": question,
        "answer": answer
    })

# 2) (Optional) Also include your JSON contexts as rows
for jf in glob.glob("data/*.json"):
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    for item in items:
        ctx = (item.get("context") or item.get("response") or "").replace("\n"," ")
        if not ctx: 
            continue
        ontext = ctx[:200].strip()
        question = "What is this document about?"
        answer = ontext
        rows.append({
            "ontext": ontext,
            "question": question,
            "answer": answer
        })

# 3) Write out the CSV
df = pd.DataFrame(rows)
out_path = os.path.join("data_cleaned", "training_data.csv")
df.to_csv(out_path, index=False)
print(f"✅ Wrote {len(df)} rows to {out_path}")
