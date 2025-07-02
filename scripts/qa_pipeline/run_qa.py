#!/usr/bin/env python3
from qa_pipeline import run_pipeline

if __name__ == "__main__":
    # you can override these via env vars if you like
    run_pipeline(
        data_path = "data_cleaned",
        output_file = "output/annotated_ww1_qa.json",
        max_qas   = 100,
        clusters  = 20,
    )
