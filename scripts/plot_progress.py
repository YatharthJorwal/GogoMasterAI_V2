"""
Plots the actual training progress from checkpoints/train_log.csv -- the loss
curve and LR schedule across every session so far, resumed runs included
(the log is appended to across --resume sessions, not overwritten).

Usage:
    python -m scripts.plot_progress
    python -m scripts.plot_progress --out my_progress.png --smooth 20
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="checkpoints/train_log.csv")
    ap.add_argument("--out", default="checkpoints/progress.png")
    ap.add_argument("--smooth", type=int, default=10,
                     help="rolling-average window for the loss curve (raw loss is noisy step-to-step)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"No log found at {args.log} -- run some training first (train.py writes this file).")

    steps, losses, lrs = [], [], []
    with open(args.log) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))

    if len(steps) < 2:
        raise SystemExit(f"Only {len(steps)} logged point(s) so far -- train a bit longer before plotting.")

    # simple rolling average to smooth step-to-step noise in the raw loss
    def rolling_avg(values, window):
        if window <= 1:
            return values
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            out.append(sum(values[lo:i + 1]) / (i - lo + 1))
        return out

    smoothed = rolling_avg(losses, args.smooth)

    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save a PNG
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, losses, color="#cccccc", linewidth=0.8, label="raw loss")
    ax1.plot(steps, smoothed, color="#D85A30", linewidth=2, label=f"{args.smooth}-step avg")
    ax1.set_ylabel("training loss")
    ax1.set_title(f"wiki200m training progress -- {steps[-1]:,} steps so far")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, lrs, color="#1D9E75", linewidth=1.5)
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    print(f"Latest: step {steps[-1]:,}, loss {losses[-1]:.4f} (smoothed: {smoothed[-1]:.4f}), lr {lrs[-1]:.2e}")


if __name__ == "__main__":
    main()
