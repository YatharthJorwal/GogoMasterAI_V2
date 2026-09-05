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


def iter_fineweb_edu(num_docs: int, start: int = 0):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    if start:
        ds = ds.skip(start)
    for i, row in enumerate(ds):
        if i >= num_docs:
            break
        yield row["text"]


def iter_wikipedia(num_docs: int, lang: str = "20231101.en", start: int = 0):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", lang, split="train", streaming=True)
    if start:
        ds = ds.skip(start)
    for i, row in enumerate(ds):
        if i >= num_docs:
            break
        yield row["text"]


def tokenize_stream_to_shard(doc_iter, tokenizer: Tokenizer, out_path: str, shard_token_capacity: int,
                              shard_idx_start: int = 0, progress_every: int = 20_000):
    """Tokenize a stream of documents, append <eos> between docs, write uint16 shards.

    Buffers tokens in an array.array('H') rather than a plain Python list: a list
    of N Python int objects costs ~28+ bytes/element (plus pointer overhead and
    extra headroom during list.extend's internal resizing), so a 50M-token list
    can balloon past 1.8GB. array('H') stores each token as a raw 2-byte C
    unsigned short, so the same buffer is ~100MB.

    Also runs periodic gc.collect() -- on memory-constrained machines, letting
    reference-cycle garbage pile up across hundreds of thousands of loop
    iterations can itself become a meaningful chunk of resident memory.
    """
    import array
    import gc

    buf = array.array("H")
    shard_idx = shard_idx_start
    total_tokens = 0
    doc_count = 0
    eos_id = tokenizer.token_to_id("<eos>")

    def flush(buf, shard_idx):
        path = f"{out_path}.{shard_idx:04d}.bin"
        with open(path, "wb") as f:
            buf.tofile(f)
        print(f"  wrote {path}: {len(buf):,} tokens  (after {doc_count:,} docs)")

    for doc in doc_iter:
        ids = tokenizer.encode(doc).ids
        ids.append(eos_id)
        buf.extend(ids)
        total_tokens += len(ids)
        doc_count += 1

        if doc_count % progress_every == 0:
            print(f"  ...{doc_count:,} docs processed, {total_tokens:,} tokens so far")
        if doc_count % (progress_every * 10) == 0:
            gc.collect()

        if len(buf) >= shard_token_capacity:
            flush(buf, shard_idx)
            shard_idx += 1
            buf = array.array("H")

    if len(buf) > 0:
        flush(buf, shard_idx)

    print(f"Total tokens for {out_path}: {total_tokens:,} ({doc_count:,} docs)")
    return total_tokens, doc_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", default="data/packed")
    ap.add_argument("--fineweb-docs", type=int, default=2_000_000)
    ap.add_argument("--wiki-docs", type=int, default=500_000)
    ap.add_argument("--shard-tokens", type=int, default=10_000_000,
                     help="tokens per .bin shard file (uint16 -> 2 bytes/token). Lowered from "
                          "50M to 10M by default -- smaller shards flush (and free the buffer) "
                          "more often, which helps on memory-constrained machines.")
    ap.add_argument("--fineweb-start-doc", type=int, default=0,
                     help="resume: skip this many FineWeb-Edu docs (see doc counts printed before a crash)")
    ap.add_argument("--fineweb-shard-idx-start", type=int, default=0,
                     help="resume: start numbering FineWeb-Edu shards from here, so you don't overwrite existing ones")
    ap.add_argument("--wiki-start-doc", type=int, default=0, help="resume: skip this many Wikipedia docs")
    ap.add_argument("--wiki-shard-idx-start", type=int, default=0,
                     help="resume: start numbering Wikipedia shards from here")
    ap.add_argument("--skip-fineweb", action="store_true", help="skip the FineWeb-Edu pass entirely (e.g. already done)")
    ap.add_argument("--skip-wiki", action="store_true", help="skip the Wikipedia pass entirely (e.g. already done)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() < 65536, "uint16 packing requires vocab_size < 65536"

    fw_tokens = 0
    if not args.skip_fineweb:
        print("Tokenizing FineWeb-Edu...")
        if args.fineweb_start_doc:
            print(f"  (resuming: skipping first {args.fineweb_start_doc:,} docs, "
                  f"shards will start at index {args.fineweb_shard_idx_start})")
        fw_tokens, _ = tokenize_stream_to_shard(
            iter_fineweb_edu(args.fineweb_docs, start=args.fineweb_start_doc), tok,
            os.path.join(args.out_dir, "fineweb_edu"), args.shard_tokens,
            shard_idx_start=args.fineweb_shard_idx_start,
        )

    wiki_tokens = 0
    if not args.skip_wiki:
        print("Tokenizing Wikipedia...")
        if args.wiki_start_doc:
            print(f"  (resuming: skipping first {args.wiki_start_doc:,} docs, "
                  f"shards will start at index {args.wiki_shard_idx_start})")
        wiki_tokens, _ = tokenize_stream_to_shard(
            iter_wikipedia(args.wiki_docs, start=args.wiki_start_doc), tok,
            os.path.join(args.out_dir, "wikipedia"), args.shard_tokens,
            shard_idx_start=args.wiki_shard_idx_start,
        )

    total = fw_tokens + wiki_tokens
    print(f"\nGrand total this run: {total:,} tokens (~{total / 1e9:.2f}B)")
    print("Chinchilla-optimal for a 200M model is ~4B tokens -- adjust --fineweb-docs / --wiki-docs to hit that.")
    print("(If this was a resumed run, add previous runs' totals to get the real grand total.)")


if __name__ == "__main__":
    main()
