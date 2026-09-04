"""
Reads the uint16 .bin shards produced by prepare_data.py via numpy memmap, so
random-access batch sampling never loads the whole corpus into RAM.
"""
import glob
import os

import numpy as np
import torch


class PackedDataset:
    def __init__(self, data_dir: str, pattern: str = "*.bin", seq_len: int = 1024):
        self.seq_len = seq_len
        self.shard_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards matching {pattern} in {data_dir}")

        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.shard_paths]
        self.shard_lens = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_lens)
        print(f"Loaded {len(self.shards)} shard(s), {self.total_tokens:,} tokens total")

    def get_batch(self, batch_size: int, device: str = "cpu"):
        xs, ys = [], []
        for _ in range(batch_size):
            # pick a shard weighted by its length, then a random window in it
            shard_idx = np.random.choice(len(self.shards), p=np.array(self.shard_lens) / self.total_tokens)
            shard = self.shards[shard_idx]
            max_start = len(shard) - self.seq_len - 1
            start = np.random.randint(0, max_start)
            chunk = shard[start:start + self.seq_len + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        if device != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y
