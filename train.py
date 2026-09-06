"""
Pretraining loop for the 200M model, tuned for a single 12GB GPU.

Key single-GPU tricks used here:
  - bf16 autocast (no GradScaler needed, unlike fp16)
  - gradient accumulation to reach a large effective batch size
  - gradient checkpointing (trade compute for activation memory)
  - fused AdamW when available
  - cosine LR schedule with linear warmup

Usage:
    python train.py --data-dir data/packed --tokenizer tokenizer/wiki200m_tokenizer.json

All the knobs below are deliberately conservative defaults for a 3060 12GB.
Watch nvidia-smi on your first run and push --micro-batch-size up until you're
near the memory ceiling, then let --grad-accum-steps make up the rest of the
target effective batch size.
"""
import argparse
import math
import os
import time

import torch

from model import ModelConfig, Wiki200M
from data.packed_dataset import PackedDataset


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/packed")
    ap.add_argument("--tokenizer", default="tokenizer/wiki200m_tokenizer.json")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layer", type=int, default=16)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--n-kv-head", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=2816)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch-size", type=int, default=8,
                     help="per-step batch size that fits in GPU memory")
    ap.add_argument("--grad-accum-steps", type=int, default=32,
                     help="micro_batch_size * grad_accum_steps * seq_len = tokens/optimizer-step")
    ap.add_argument("--max-steps", type=int, default=60_000)
    ap.add_argument("--warmup-steps", type=int, default=1_000)
    ap.add_argument("--max-lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action="store_true", default=False,
                     help="torch.compile -- try it, disable if it errors on your setup")
    ap.add_argument("--resume", default=None, help="path to a checkpoint .pt to resume from")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found, running on CPU (fine for smoke-testing only)")
    os.makedirs(args.out_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    cfg = ModelConfig(
        vocab_size=vocab_size, max_seq_len=args.seq_len,
        d_model=args.d_model, n_layer=args.n_layer,
        n_head=args.n_head, n_kv_head=args.n_kv_head, d_ff=args.d_ff,
    )
    model = Wiki200M(cfg).to(device)
    print(f"Model: {model.num_params():,} params ({model.num_params() / 1e6:.1f}M)")

    if args.grad_checkpointing:
        # simple manual checkpointing hook over the block list
        import torch.utils.checkpoint as cp
        orig_forward = Wiki200M.forward

        def checkpointed_forward(self, idx, targets=None):
            x = self.tok_embed(idx)
            for block in self.blocks:
                x = cp.checkpoint(lambda x, b=block: b(x, self.rope_cos, self.rope_sin), x, use_reentrant=False)
            x = self.final_norm(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                import torch.nn.functional as F
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss

        model.forward = checkpointed_forward.__get__(model)

    if args.compile:
        model = torch.compile(model)

    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    use_fused = device == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=args.max_lr, betas=(0.9, 0.95), fused=use_fused)

    start_step = 0
    if args.resume:
        # weights_only=False: we trust our own checkpoints and store a ModelConfig
        # dataclass alongside the tensors, which the default weights_only=True
        # loader (PyTorch >= 2.6) refuses to unpickle.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")

    train_ds = PackedDataset(args.data_dir, seq_len=args.seq_len)

    tokens_per_step = args.micro_batch_size * args.grad_accum_steps * args.seq_len
    print(f"Effective batch: {tokens_per_step:,} tokens/optimizer-step")

    log_path = os.path.join(args.out_dir, "train_log.csv")
    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    if log_is_new:
        log_file.write("step,loss,lr,tok_per_sec,wall_time\n")

    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model.train()
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = train_ds.get_batch(args.micro_batch_size, device=device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
            loss = loss / args.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 10 == 0:
            dt = time.time() - t0
            tok_per_sec = tokens_per_step * 10 / dt if step > start_step else 0
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s")
            log_file.write(f"{step},{loss_accum:.6f},{lr:.8e},{tok_per_sec:.1f},{time.time():.1f}\n")
            log_file.flush()
            t0 = time.time()

        if step > 0 and step % args.save_every == 0:
            path = os.path.join(args.out_dir, f"ckpt_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": cfg}, path)
            print(f"  saved checkpoint: {path}")

    final_path = os.path.join(args.out_dir, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": args.max_steps, "config": cfg}, final_path)
    print(f"Training complete. Saved {final_path}")
    log_file.close()


if __name__ == "__main__":
    main()
