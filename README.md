# wiki200m

A ~200M parameter, Llama-style decoder-only transformer, trained from scratch,
with a Wikipedia tool-calling / retrieval feature. Designed for single-GPU
training on a 12GB card (developed against an RTX 3060 12GB).

Everything in this repo has been smoke-tested (forward/backward/generate,
tokenizer training, data packing shape, and a full multi-step training run
including checkpoint save + resume) on CPU with tiny configs. The 200M-scale
run itself needs to happen on your GPU — this repo is validated for
correctness, not yet for wall-clock performance on your hardware.

## Status

- [x] Model architecture (RoPE, GQA, SwiGLU, RMSNorm) — `model/`
- [x] Tokenizer trainer — `tokenizer/train_tokenizer.py`
- [x] Data pipeline (tokenize + pack to binary shards) — `data/prepare_data.py`
- [x] Pretraining loop (grad accum, grad checkpointing, cosine LR, resume) — `train.py`
- [ ] Wikipedia retrieval backend (BM25/FAISS index over the Wikipedia dump)
- [ ] Tool-calling SFT dataset + fine-tuning script
- [ ] Inference harness (detect `<tool_call>`, run search, splice result, resume generation)

The last three are the next phase — this repo currently gets you a trained
*base* model. Ping me when you're ready to build those and we'll do the same
thing: design, implement, smoke-test.

## Setup (on your machine, with the GPU)

```bash
pip install -r requirements.txt
```

## 1. Train the tokenizer

Point it at whatever text you have locally to start (a Wikipedia dump export,
or even just a folder of .txt files). 32k vocab is the default and is a
reasonable size for a 200M model.

```bash
python -m tokenizer.train_tokenizer \
    --input "data/raw/*.txt" \
    --vocab-size 32000 \
    --out tokenizer/wiki200m_tokenizer.json
```

## 2. Check the param count for your target config

```bash
python -m model.config
```

Edit `model/config.py` defaults (`d_model`, `n_layer`, `d_ff`) or pass
overrides to `train.py` (`--d-model`, `--n-layer`, `--n-head`, `--n-kv-head`,
`--d-ff`) until `count_params()` lands near 200M. The defaults land at ~213M
total / ~180M non-embedding, which is a good starting point.

## 3. Prepare training data

This needs network access to huggingface.co, so run it on your machine, not
in a sandboxed environment:

```bash
python -m data.prepare_data \
    --tokenizer tokenizer/wiki200m_tokenizer.json \
    --out-dir data/packed \
    --fineweb-docs 2000000 \
    --wiki-docs 500000
```

Watch the "Grand total" token count it prints at the end. Chinchilla-optimal
for 200M params is ~4B tokens — adjust `--fineweb-docs` / `--wiki-docs` up or
down to hit that (or go higher; small models tend to keep improving well past
the "optimal" point if you have the compute budget).

## 4. Pretrain

```bash
python train.py \
    --data-dir data/packed \
    --tokenizer tokenizer/wiki200m_tokenizer.json \
    --micro-batch-size 8 \
    --grad-accum-steps 32 \
    --max-steps 60000
```

Tuning notes for a 3060 12GB:
- Start with `--micro-batch-size 8`, watch `nvidia-smi` during the first few
  steps, and push it up until you're near the memory ceiling. Let
  `--grad-accum-steps` make up the rest of your target effective batch size
  (aim for ~250k-500k tokens/optimizer-step; the script prints this at
  startup).
- Gradient checkpointing is on by default (`--grad-checkpointing`); it trades
  ~20-30% more compute for a lot more memory headroom, worth it at this
  model size on 12GB. Disable with `--no-grad-checkpointing` if you have
  memory to spare and want the speed.
- `--compile` (torch.compile) can give a meaningful speedup on Ampere but
  occasionally has rough edges — try it, fall back to without if it errors.
- Checkpoints save to `checkpoints/ckpt_<step>.pt` and include optimizer
  state, so `--resume checkpoints/ckpt_010000.pt` picks up exactly where you
  left off — expect to need this since a multi-day run on a home machine
  will get interrupted.

Rough expectation: with a Chinchilla-optimal ~4B tokens, an effective batch
of ~300k tokens/step is ~13,000 optimizer steps. Real wall-clock time depends
heavily on your actual achieved tokens/sec (the script logs this every 10
steps) — watch the first few hundred steps and extrapolate.

## Repo layout

```
model/
  config.py           ModelConfig dataclass + param counter
  model.py             The transformer (RoPE, GQA attention, SwiGLU, RMSNorm)
tokenizer/
  train_tokenizer.py   Trains a byte-level BPE tokenizer, reserves
                        <tool_call>/</tool_call>/<tool_result>/</tool_result>
                        special tokens for the later tool-calling stage
data/
  prepare_data.py       Streams FineWeb-Edu + Wikipedia, tokenizes, packs
                        into uint16 binary shards
  packed_dataset.py     Memory-mapped random-batch sampler over the shards
train.py                Pretraining loop
checkpoints/             (empty, gitignored — your checkpoints land here)
```
