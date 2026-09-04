"""
Train a 32k-vocab byte-level BPE tokenizer on your own corpus, instead of reusing
GPT-2's. A tokenizer trained on your actual data (Wikipedia + web text) will encode
it more efficiently, which directly translates to more "effective" tokens per
training step.

Usage:
    python -m tokenizer.train_tokenizer \
        --input data/raw/*.txt \
        --vocab-size 32000 \
        --out tokenizer/wiki200m_tokenizer.json

Input: one or more plain-text files, one document per line (or just large text
files -- BPE training doesn't care about line boundaries much, but one-doc-per-line
is convenient for streaming from datasets later).
"""
import argparse
import glob

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


SPECIAL_TOKENS = [
    "<pad>",   # id 0
    "<bos>",   # id 1
    "<eos>",   # id 2
    "<unk>",   # id 3
    # tool-calling / retrieval special tokens -- reserved now so the format is
    # stable vocab IDs the SFT stage and inference harness can rely on later.
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
]


def build_tokenizer(vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="glob pattern(s) for input text files")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--out", default="tokenizer/wiki200m_tokenizer.json")
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    files = []
    for pattern in args.input:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise SystemExit(f"No files matched: {args.input}")
    print(f"Training on {len(files)} file(s): {files[:5]}{' ...' if len(files) > 5 else ''}")

    tok = build_tokenizer(args.vocab_size)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train(files, trainer)

    tok.save(args.out)
    print(f"Saved tokenizer to {args.out}")
    print(f"Vocab size: {tok.get_vocab_size()}")

    # sanity check
    sample = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."
    enc = tok.encode(sample)
    print(f"Sample encode ({len(enc.ids)} tokens): {enc.tokens[:15]}...")
    print(f"Decoded back: {tok.decode(enc.ids)}")


if __name__ == "__main__":
    main()
