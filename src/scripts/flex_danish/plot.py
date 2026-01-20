#!/usr/bin/env python3
"""
Read training logs from stdin, extract (step, train/CE loss), de-duplicate repeated
copies per step, and plot the curve.

Usage:
  cat train.log | ./plot_ce_loss.py
  tail -f train.log | ./plot_ce_loss.py   # will plot when stdin closes (Ctrl-C)
"""
import re
import sys

# Use a non-interactive backend so this works over ssh / pipes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"\[step=(\d+)/")
CE_RE   = re.compile(r"^\s*train/CE loss\s*=\s*([0-9]*\.?[0-9]+)\s*$")

def main() -> int:
    steps = []
    losses = []

    pending_step = None
    last_seen = {}  # step -> loss (dedupe); keeps last value per step

    for line in sys.stdin:
        m = STEP_RE.search(line)
        if m:
            pending_step = int(m.group(1))
            continue

        m = CE_RE.match(line)
        if m and pending_step is not None:
            loss = float(m.group(1))
            # Multiple identical (or even slightly different) copies can appear;
            # keep the last one for that step.
            last_seen[pending_step] = loss

    if not last_seen:
        print("No (step, train/CE loss) pairs found on stdin.", file=sys.stderr)
        return 1

    for s in sorted(last_seen):
        steps.append(s)
        losses.append(last_seen[s])

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("train/CE loss")
    plt.title("CE loss vs step")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out = "flex_danish_loss.png"
    plt.savefig(out, dpi=160)
    print(out)  # print output filename so you can script around it
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

