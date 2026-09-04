"""
Tokenize a text corpus and pack it into flat binary shards of uint16 token ids
(nanoGPT-style). This is the single biggest practical win for single-GPU training:
you tokenize ONCE, then every epoch just memory-maps the file and reads raw ints --
no tokenizer in the hot loop, no dataset object overhead.

Two corpora, concatenated:
  - HuggingFaceFW/fineweb-edu  (high-quality filtered web text -- does a lot of
    heavy lifting for small models per token spent)
  - wikimedia/wikipedia        (also becomes your retrieval corpus later, so
    training on it too means the model already has priors about its own
    tool-call source)

Usage:
    python -m data.prepare_data \
        --tokenizer tokenizer/wiki200m_tokenizer.json \
        --out-dir data/packed \
        --fineweb-docs 2_000_000 \
        --wiki-docs 500_000

Requires: pip install datasets tokenizers numpy
Needs network access to huggingface.co (not available in this sandbox --
run this on your own machine).
"""
import argparse
import os

import numpy as np
from tokenizers import Tokenizer


def iter_fineweb_edu(num_docs: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= num_docs:
            break
        yield row["text"]


def iter_wikipedia(num_docs: int, lang: str = "20231101.en"):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", lang, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= num_docs:
            break
        yield row["text"]


def tokenize_stream_to_shard(doc_iter, tokenizer: Tokenizer, out_path: str, shard_token_capacity: int):
    """Tokenize a stream of documents, append <eos> between docs, write uint16 shards."""
    buf = []
    shard_idx = 0
    total_tokens = 0
    eos_id = tokenizer.token_to_id("<eos>")

    def flush(buf, shard_idx):
        arr = np.array(buf, dtype=np.uint16)
        path = f"{out_path}.{shard_idx:04d}.bin"
        arr.tofile(path)
        print(f"  wrote {path}: {len(arr):,} tokens")

    for doc in doc_iter:
        ids = tokenizer.encode(doc).ids
        ids.append(eos_id)
        buf.extend(ids)
        total_tokens += len(ids)
        if len(buf) >= shard_token_capacity:
            flush(buf, shard_idx)
            shard_idx += 1
            buf = []

    if buf:
        flush(buf, shard_idx)

    print(f"Total tokens for {out_path}: {total_tokens:,}")
    return total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", default="data/packed")
    ap.add_argument("--fineweb-docs", type=int, default=2_000_000)
    ap.add_argument("--wiki-docs", type=int, default=500_000)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000,
                     help="tokens per .bin shard file (uint16 -> 2 bytes/token)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() < 65536, "uint16 packing requires vocab_size < 65536"

    print("Tokenizing FineWeb-Edu...")
    fw_tokens = tokenize_stream_to_shard(
        iter_fineweb_edu(args.fineweb_docs), tok,
        os.path.join(args.out_dir, "fineweb_edu"), args.shard_tokens,
    )

    print("Tokenizing Wikipedia...")
    wiki_tokens = tokenize_stream_to_shard(
        iter_wikipedia(args.wiki_docs), tok,
        os.path.join(args.out_dir, "wikipedia"), args.shard_tokens,
    )

    total = fw_tokens + wiki_tokens
    print(f"\nGrand total: {total:,} tokens (~{total / 1e9:.2f}B)")
    print("Chinchilla-optimal for a 200M model is ~4B tokens -- adjust --fineweb-docs / --wiki-docs to hit that.")


if __name__ == "__main__":
    main()
