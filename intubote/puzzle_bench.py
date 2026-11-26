"""Score IntuBote checkpoints on the held-out Lichess puzzle benchmark.

A puzzle counts as **solved** only if the model, playing with no search, finds
every one of the solver's moves (any mating move is accepted on the last move,
as on Lichess). The model picks its highest-probability *legal* move.

Because a puzzle fails at the first wrong move, checking each solver move with
the correct line played before it gives exactly the same verdict as playing the
puzzle through. So all positions are scored in one batched pass -- seconds on a
GPU, so train.py can run it at every evaluation (``--puzzle-bench``).

CLI (one column per checkpoint, so models can be compared side by side):
    python intubote/puzzle_bench.py --bench intubote/data/puzzles/bench.npz \
        --ckpt base.pt finetuned.pt [--themes]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding import NUM_MOVES  # noqa: E402
from prepare_puzzles import BANDS, band_of  # noqa: E402

# Themes shown with --themes (the most common, plus a few telling ones).
THEMES = ["mateIn1", "mateIn2", "mateIn3", "fork", "pin", "skewer", "discoveredAttack",
          "hangingPiece", "sacrifice", "deflection", "attraction", "quietMove",
          "defensiveMove", "endgame", "middlegame", "short", "long", "veryLong"]


class Bench:
    """The benchmark tensors, moved to a device once."""

    def __init__(self, path: str, device: torch.device | str = "cpu") -> None:
        d = np.load(path)
        unpack = lambda a: np.unpackbits(a, axis=1, count=NUM_MOVES).astype(bool)  # noqa: E731
        self.device = torch.device(device)
        self.tokens = torch.from_numpy(d["tokens"]).to(self.device)
        self.legal = torch.from_numpy(unpack(d["legal"])).to(self.device)
        self.accepted = torch.from_numpy(unpack(d["accepted"])).to(self.device)
        self.puzzle = torch.from_numpy(d["puzzle"].astype(np.int64)).to(self.device)
        self.rating = d["rating"].astype(np.int64)
        self.n_puzzles = len(self.rating)
        self.band = np.array([band_of(int(r)) for r in self.rating])
        rows_path = os.path.join(os.path.dirname(path), "bench_puzzles.jsonl")
        self.themes: list[list[str]] = []
        if os.path.exists(rows_path):
            with open(rows_path) as f:
                self.themes = [json.loads(ln)["themes"] for ln in f if ln.strip()]


@torch.no_grad()
def solve(model: torch.nn.Module, bench: Bench, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Return (move_correct [positions], puzzle_solved [puzzles]) as bool arrays."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    use_amp = dev.type == "cuda"
    correct = torch.empty(len(bench.tokens), dtype=torch.bool, device=bench.device)
    for s in range(0, len(bench.tokens), batch):
        tok = bench.tokens[s:s + batch].to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tok)[0]
        logits = logits.float().masked_fill(~bench.legal[s:s + batch].to(dev), float("-inf"))
        pick = logits.argmax(1, keepdim=True).to(bench.device)
        correct[s:s + batch] = bench.accepted[s:s + batch].gather(1, pick).squeeze(1)
    if was_training:
        model.train()
    # A puzzle is solved when none of its moves is wrong.
    wrong = torch.zeros(bench.n_puzzles, dtype=torch.long, device=bench.device)
    wrong.index_add_(0, bench.puzzle, (~correct).long())
    return correct.cpu().numpy(), (wrong == 0).cpu().numpy()


def summarize(bench: Bench, correct: np.ndarray, solved: np.ndarray) -> dict:
    """Overall and per-band solve rates (fractions)."""
    out = {"puzzles": float(solved.mean()) if len(solved) else 0.0,
           "puzzle_moves": float(correct.mean()) if len(correct) else 0.0, "bands": {}}
    for i, (label, _, _) in enumerate(BANDS):
        sel = bench.band == i
        if sel.any():
            out["bands"][label] = float(solved[sel].mean())
    return out


def evaluate(model: torch.nn.Module, bench: Bench) -> dict:
    return summarize(bench, *solve(model, bench))


def main() -> None:
    ap = argparse.ArgumentParser(description="IntuBote puzzle benchmark")
    ap.add_argument("--bench", required=True, help="bench.npz from prepare_puzzles.py")
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints")
    ap.add_argument("--themes", action="store_true", help="also break results down by theme")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", help="also write the results to this JSON file")
    args = ap.parse_args()

    from model import load_checkpoint

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    bench = Bench(args.bench, device)
    names = [os.path.basename(p).removesuffix(".pt") for p in args.ckpt]
    results = {}
    for name, path in zip(names, args.ckpt):
        correct, solved = solve(load_checkpoint(path, device), bench)
        results[name] = (correct, solved, summarize(bench, correct, solved))

    width = max(12, *(len(n) for n in names)) + 2
    print(f"{'':22}{'puzzles':>8}" + "".join(f"{n:>{width}}" for n in names))

    def row(label: str, sel: np.ndarray) -> None:
        if not sel.any():
            return
        cells = "".join(f"{results[n][1][sel].mean():>{width}.1%}" for n in names)
        print(f"{label:22}{int(sel.sum()):>8,}{cells}")

    row("all puzzles solved", np.ones(bench.n_puzzles, dtype=bool))
    print(f"{'  moves correct':22}{len(bench.tokens):>8,}"
          + "".join(f"{results[n][0].mean():>{width}.1%}" for n in names))
    print("by puzzle rating")
    for i, (label, _, _) in enumerate(BANDS):
        row(f"  {label}", bench.band == i)
    if args.themes and bench.themes:
        print("by theme")
        for theme in THEMES:
            row(f"  {theme}", np.array([theme in t for t in bench.themes]))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({n: results[n][2] for n in names}, f, indent=1)


if __name__ == "__main__":
    main()
