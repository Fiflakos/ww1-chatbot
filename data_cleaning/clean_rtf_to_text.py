#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

# pip install striprtf
from striprtf.striprtf import rtf_to_text

def clean_text(text: str) -> str:
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # strip leading/trailing
    return text.strip()

def process_subfolder(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for rtf_path in sorted(in_dir.glob("*.rtf")):
        try:
            raw = rtf_path.read_text(encoding="utf-8", errors="ignore")
            # convert RTF → plain
            plain = rtf_to_text(raw)
            cleaned = clean_text(plain)
            # write out as .txt
            txt_path = out_dir / (rtf_path.stem + ".txt")
            txt_path.write_text(cleaned, encoding="utf-8")
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed to parse {rtf_path}: {e}")

    print(f"[✅] Processed {count} RTF files in {in_dir.name}")

def main():
    p = argparse.ArgumentParser(
        description="Recursively convert .rtf → cleaned .txt",
    )
    p.add_argument(
        "--input_dir", "-i",
        required=True,
        help="root folder of your raw RTFs (e.g. data/Letters)",
    )
    p.add_argument(
        "--output_dir", "-o",
        required=True,
        help="where to put cleaned TXT (e.g. data_cleaned/Letters)",
    )
    args = p.parse_args()

    root_in  = Path(args.input_dir)
    root_out = Path(args.output_dir)

    # expect subfolders Letters, Diaries, Others
    for sub in ["Letters", "Diaries", "Others"]:
        in_sub  = root_in  / sub
        out_sub = root_out / sub
        if in_sub.exists():
            process_subfolder(in_sub, out_sub)
        else:
            print(f"[⚠️] Skipping missing folder: {in_sub}")

if __name__ == "__main__":
    main()
