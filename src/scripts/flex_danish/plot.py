#!/usr/bin/env python3
"""
Parse (step, train/CE loss) from stdin, dedupe repeated copies per step, and plot.
Smoothing can be: none | ema | savgol.

Examples:
  cat train.log | ./plot_ce_loss.py --smooth none
  cat train.log | ./plot_ce_loss.py --smooth ema --ema-alpha 0.1
  cat train.log | ./plot_ce_loss.py --smooth savgol --savgol-window 51 --savgol-poly 3

Requires:
  pip install matplotlib typer
Optional (only if using savgol):
  pip install scipy
"""

import re
import sys
from enum import Enum
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer


STEP_RE = re.compile(r"\[step=(\d+)/")
CE_RE = re.compile(r"^\s*train/CE loss\s*=\s*([0-9]*\.?[0-9]+)\s*$")


class Smoothing(str, Enum):
    none = "none"
    ema = "ema"
    savgol = "savgol"


def parse_stdin() -> Tuple[List[int], List[float]]:
    pending_step: Optional[int] = None
    last_seen: dict[int, float] = {}

    for line in sys.stdin:
        m = STEP_RE.search(line)
        if m:
            pending_step = int(m.group(1))
            continue

        m = CE_RE.match(line)
        if m and pending_step is not None:
            last_seen[pending_step] = float(m.group(1))

    if not last_seen:
        raise ValueError("No (step, train/CE loss) pairs found on stdin.")

    steps = sorted(last_seen.keys())
    losses = [last_seen[s] for s in steps]
    return steps, losses


def ema_smooth(y: List[float], alpha: float) -> List[float]:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1].")
    if not y:
        return []
    out = [y[0]]
    for v in y[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def savgol_smooth(y: List[float], window: int, poly: int) -> List[float]:
    # Import only when used.
    try:
        from scipy.signal import savgol_filter  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Savgol smoothing requires scipy. Install with: pip install scipy"
        ) from e

    n = len(y)
    if n < 5:
        return y[:]  # not enough points to smooth meaningfully

    if window < 3:
        raise ValueError("--savgol-window must be >= 3.")
    if window % 2 == 0:
        raise ValueError("--savgol-window must be odd.")
    if poly < 1:
        raise ValueError("--savgol-poly must be >= 1.")
    if poly >= window:
        raise ValueError("--savgol-poly must be < --savgol-window.")
    if window > n:
        # clamp to largest odd <= n
        window = n if (n % 2 == 1) else n - 1
        if window < 3:
            return y[:]
        if poly >= window:
            poly = max(1, window - 2)

    return list(savgol_filter(y, window_length=window, polyorder=poly))


def main(
    out: str = typer.Option("flex_danish_loss.png", help="Output PNG filename."),
    title: str = typer.Option("CE loss vs step", help="Plot title."),
    smooth: Smoothing = typer.Option(Smoothing.none, help="Smoothing: none|ema|savgol"),
    # EMA params
    ema_alpha: float = typer.Option(
        0.1, "--ema-alpha", help="EMA alpha in (0,1]. Higher = less smoothing."
    ),
    # Savgol params
    savgol_window: int = typer.Option(
        51, "--savgol-window", help="Savgol window length (odd integer)."
    ),
    savgol_poly: int = typer.Option(
        3, "--savgol-poly", help="Savgol polynomial order (< window)."
    ),
    # Plot cosmetics
    show_raw: bool = typer.Option(True, help="Overlay raw curve on top of smoothed."),
    raw_alpha: float = typer.Option(0.35, help="Alpha for raw curve (if shown)."),
    dpi: int = typer.Option(160, help="PNG DPI."),
) -> None:
    steps, losses = parse_stdin()

    y_plot = losses
    label = "raw"

    if smooth == Smoothing.none:
        y_smooth = None
    elif smooth == Smoothing.ema:
        y_smooth = ema_smooth(losses, ema_alpha)
        y_plot = y_smooth
        label = f"ema(alpha={ema_alpha:g})"
    elif smooth == Smoothing.savgol:
        y_smooth = savgol_smooth(losses, savgol_window, savgol_poly)
        y_plot = y_smooth
        label = f"savgol(win={savgol_window},poly={savgol_poly})"
    else:
        raise ValueError(f"Unknown smoothing mode: {smooth}")

    plt.figure()

    if show_raw and smooth != Smoothing.none:
        plt.plot(steps, losses, alpha=raw_alpha, label="raw")

    plt.plot(steps, y_plot, label=label)

    plt.xlabel("step")
    plt.ylabel("train/CE loss")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    print(out)


if __name__ == "__main__":
    typer.run(main)

