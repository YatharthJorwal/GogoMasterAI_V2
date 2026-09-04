"""
Pulls a modest text sample from FineWeb-Edu + Wikipedia and writes it to
data/raw/*.txt for tokenizer/train_tokenizer.py to consume.

Tokenizer training doesn't need billions of tokens -- a representative few
hundred MB is plenty for a stable 32k BPE vocab. This is deliberately much
smaller and faster than the full prepare_data.py pull.

Usage:
    python -m scripts.fetch_tokenizer_sample --fineweb-docs 100000 --wiki-docs 50000
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--fineweb-docs", type=int, default=100_000)
    ap.add_argument("--wiki-docs", type=int, default=50_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from datasets import load_dataset

    fw_path = os.path.join(args.out_dir, "fineweb_edu_sample.txt")
    print(f"Pulling {args.fineweb_docs:,} FineWeb-Edu docs -> {fw_path}")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    with open(fw_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= args.fineweb_docs:
                break
            f.write(row["text"].replace("\n", " ") + "\n")
            if i % 10_000 == 0:
                print(f"  {i:,} docs written")

    wiki_path = os.path.join(args.out_dir, "wikipedia_sample.txt")
    print(f"Pulling {args.wiki_docs:,} Wikipedia docs -> {wiki_path}")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    with open(wiki_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= args.wiki_docs:
                break
            f.write(row["text"].replace("\n", " ") + "\n")
            if i % 10_000 == 0:
                print(f"  {i:,} docs written")

    print("Done. Now run:")
    print('  python -m tokenizer.train_tokenizer --input "data/raw/*.txt" --vocab-size 32000 '
          "--out tokenizer/wiki200m_tokenizer.json")


if __name__ == "__main__":
    main()
