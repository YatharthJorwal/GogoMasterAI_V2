"""
Quick sanity check that PyTorch actually sees your GPU before you kick off a
multi-day training run on it.

Usage:
    python -m scripts.check_env
"""
import torch


def main():
    print(f"torch version:      {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA device visible to PyTorch. Common causes:")
        print("  - You installed the CPU-only torch wheel (check: pip show torch,")
        print("    should mention a +cu12x build, not +cpu)")
        print("  - NVIDIA driver / CUDA toolkit not installed or out of date")
        print("  - You're inside a venv/conda env that doesn't have the GPU build")
        return

    n = torch.cuda.device_count()
    print(f"CUDA device count:  {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Total memory:      {total_gb:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  bf16 supported:     {torch.cuda.is_bf16_supported()}")

    # actually allocate + run something on the GPU, not just check availability
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"\nTest matmul on GPU succeeded. Allocated: {allocated_mb:.1f} MB")
    print("You're good to train on CUDA.")


if __name__ == "__main__":
    main()
