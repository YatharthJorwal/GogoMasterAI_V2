"""
Model configuration for the ~200M param Llama-style decoder-only transformer.

Param count sanity check (roughly, non-embedding + tied embedding):
  embed_params        = vocab_size * d_model
  per_layer_attn       = 4 * d_model^2                      (q,k,v,o projections)
  per_layer_mlp         = 3 * d_model * d_ff                 (SwiGLU: gate, up, down)
  per_layer            = per_layer_attn + per_layer_mlp
  total                = embed_params + n_layer * per_layer   (embedding tied with lm_head)

With the defaults below (vocab=32000, d_model=1024, n_layer=16, d_ff=2816):
  embed   = 32000 * 1024                      ≈ 32.8M
  attn/layer  = 4 * 1024^2                     ≈ 4.19M
  mlp/layer   = 3 * 1024 * 2816                ≈ 8.65M
  per layer   ≈ 12.84M  ->  16 layers          ≈ 205.4M
  total                                        ≈ 238M params

That's a bit over 200M -- tune n_layer / d_model / d_ff to hit your exact target.
Run `python -m model.config` to print an exact count for any config.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_layer: int = 16
    n_head: int = 16
    n_kv_head: int = 4          # grouped-query attention; set equal to n_head to disable GQA
    d_ff: int = 2816            # SwiGLU hidden size (~2.75x d_model is the common Llama ratio)
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0        # pretraining from scratch on lots of tokens -> usually 0
    tie_embeddings: bool = True

    # special tokens (index positions, set to match your trained tokenizer)
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head


def count_params(cfg: ModelConfig) -> int:
    embed = cfg.vocab_size * cfg.d_model
    attn_per_layer = 2 * cfg.d_model * cfg.d_model  # q proj (full) + o proj (full)
    attn_per_layer += 2 * cfg.d_model * (cfg.d_model * cfg.n_kv_head // cfg.n_head)  # k,v proj (GQA-shrunk)
    mlp_per_layer = 3 * cfg.d_model * cfg.d_ff  # gate + up + down
    norm_per_layer = 2 * cfg.d_model  # two RMSNorms per block
    per_layer = attn_per_layer + mlp_per_layer + norm_per_layer
    total = embed + cfg.n_layer * per_layer + cfg.d_model  # + final norm
    if not cfg.tie_embeddings:
        total += embed
    return total


if __name__ == "__main__":
    cfg = ModelConfig()
    n = count_params(cfg)
    print(f"Config: {cfg}")
    print(f"Total params: {n:,} ({n / 1e6:.1f}M)")
