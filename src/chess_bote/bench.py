"""Performance benchmark.

Measures throughput of the core hot path -- legal move generation plus
make/unmake -- via perft, reported as NPS (nodes per second). This is the
standard way to track engine speed and to judge whether a future optimization
(e.g. magic bitboards or a C++ port) actually helped.

Run with::

    uv run chess-bote-bench              # default suite, depth 3
    uv run chess-bote-bench --depth 5
    uv run chess-bote-bench --fen "<FEN>" --depth 4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from .board import Board
from .movegen import generate_legal
from .perft import perft

# (name, FEN) -- a representative mix: open start, tactical middlegame, endgame.
SUITE: list[tuple[str, str]] = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("midgame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
]


@dataclass
class Result:
    name: str
    depth: int
    nodes: int
    seconds: float

    @property
    def nps(self) -> float:
        return self.nodes / self.seconds if self.seconds > 0 else 0.0


def bench_perft(name: str, fen: str, depth: int) -> Result:
    board = Board(fen)
    start = time.perf_counter()
    nodes = perft(board, depth)
    elapsed = time.perf_counter() - start
    return Result(name, depth, nodes, elapsed)


def bench_movegen(fen: str, iterations: int = 100_000) -> float:
    """Pure legal-move-generation rate (calls/sec) for a single position."""
    board = Board(fen)
    start = time.perf_counter()
    for _ in range(iterations):
        generate_legal(board)
    elapsed = time.perf_counter() - start
    return iterations / elapsed if elapsed > 0 else 0.0


def _fmt(n: float) -> str:
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:6.1f}{unit}"
        n /= 1000
    return f"{n:6.1f}T"


def run(positions: list[tuple[str, str]], depth: int) -> list[Result]:
    results = [bench_perft(name, fen, depth) for name, fen in positions]

    print(f"perft depth {depth}")
    print(f"{'position':<12}{'nodes':>12}{'time (s)':>12}{'NPS':>12}")
    print("-" * 48)
    total_nodes = 0
    total_time = 0.0
    for r in results:
        print(f"{r.name:<12}{r.nodes:>12}{r.seconds:>12.3f}{_fmt(r.nps):>12}")
        total_nodes += r.nodes
        total_time += r.seconds
    print("-" * 48)
    overall = total_nodes / total_time if total_time else 0.0
    print(f"{'TOTAL':<12}{total_nodes:>12}{total_time:>12.3f}{_fmt(overall):>12}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="chess-bote performance benchmark")
    parser.add_argument("--depth", type=int, default=3, help="perft depth (default: 3)")
    parser.add_argument(
        "--fen", type=str, default=None, help="benchmark a single FEN instead of the suite"
    )
    args = parser.parse_args()

    positions = [("custom", args.fen)] if args.fen else SUITE
    run(positions, args.depth)


if __name__ == "__main__":
    main()
