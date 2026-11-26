"""Plot IntuBote training progress from ``<checkpoint>.metrics.jsonl``.

Three small panels, one per measure (each with its own y-axis, never two
scales on one axis):

    Move accuracy   validation top-1 / top-3 (% of positions where Stockfish's
                    best move is the net's first / within its first three)
    Policy loss     train vs validation cross-entropy
    Value loss      train vs validation BCE on the win probability
    Puzzles solved  share of the held-out puzzle benchmark solved with no
                    search (only when the run used --puzzle-bench)

Usage (CLI writes a PNG):
    python intubote/plot_metrics.py intubote/intubote.pt.metrics.jsonl --out progress.png
In a notebook:
    from plot_metrics import plot_metrics; fig = plot_metrics(path)
"""

from __future__ import annotations

import argparse
import json

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Reference palette (light mode): categorical slots 1-2, surface, text, grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_metrics(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _steps_fmt(x, _pos) -> str:
    return f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_fmt))
    ax.set_xlabel("step", color=TEXT_2, fontsize=9)


def _line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", label=label)


def _end_label(ax, x, y, text, color):
    """Direct label on the newest point: a ringed dot plus the value in text ink."""
    ax.plot([x], [y], "o", markersize=7, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=TEXT)


def plot_metrics(path: str):
    rows = read_metrics(path)
    puzzles = [r for r in rows if "puzzles" in r]
    n_panels = 4 if puzzles else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.2), facecolor=SURFACE)
    if not rows:
        fig.text(0.5, 0.5, "No evaluations logged yet", ha="center", color=TEXT_2)
        for ax in axes:
            ax.set_axis_off()
        return fig

    steps = [r["step"] for r in rows]
    last = rows[-1]

    ax = axes[0]
    _style(ax, "Move accuracy (validation)")
    _line(ax, steps, [r["top1"] for r in rows], BLUE, "top-1")
    _line(ax, steps, [r["top3"] for r in rows], ORANGE, "top-3")
    _end_label(ax, steps[-1], last["top1"], f"{last['top1']:.1%}", BLUE)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(bottom=0)

    for ax, key, title in ((axes[1], "policy", "Policy loss"), (axes[2], "value", "Value loss")):
        _style(ax, title)
        _line(ax, steps, [r[f"train_{key}"] for r in rows], BLUE, "train")
        _line(ax, steps, [r[f"val_{key}"] for r in rows], ORANGE, "validation")
        _end_label(ax, steps[-1], last[f"val_{key}"], f"{last[f'val_{key}']:.3f}", ORANGE)

    for ax in axes[:3]:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="best")
    if puzzles:
        # One series: the title names it, so no legend box.
        ax = axes[3]
        _style(ax, "Puzzles solved (benchmark)")
        _line(ax, [r["step"] for r in puzzles], [r["puzzles"] for r in puzzles], BLUE, "solved")
        _end_label(ax, puzzles[-1]["step"], puzzles[-1]["puzzles"],
                   f"{puzzles[-1]['puzzles']:.1%}", BLUE)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
        ax.set_ylim(bottom=0)
    for ax in axes:
        ax.margins(x=0.08)

    fig.suptitle(
        f"step {last['step']:,}  ·  {last['positions'] / 1e6:,.1f}M positions seen  ·  "
        f"lr {last['lr']:.1e}  ·  {last['pos_per_s']:,.0f} pos/s",
        x=0.01, ha="left", color=TEXT_2, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot IntuBote training metrics")
    ap.add_argument("metrics", help="<checkpoint>.metrics.jsonl")
    ap.add_argument("--out", default="progress.png")
    args = ap.parse_args()
    plot_metrics(args.metrics).savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
