"""Generate NNUE training data: (FEN, white-relative centipawn score) pairs.

Positions come from short random playouts (using chess-bote's own legal move
generator) for variety, and are labelled by a persistent Stockfish process at a
fixed search depth. One Stockfish process is reused across all positions, which
is far faster than spawning it per position.

Usage:
    uv run python nnue/gen_data.py --count 200000 --depth 8 --out nnue/data/positions.txt

The output is one record per line: ``<fen>|<cp>`` where ``cp`` is from White's
point of view, clamped, with mates mapped to large values. Quality note: random
playouts cover a broad range of positions but are not "realistic" games; good
enough for a first net, and easy to upgrade later to engine self-play or a game
database.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from chess_bote.board import Board
from chess_bote.movegen import generate_legal
from chess_bote.types import Color

MATE_CP = 10000  # mates mapped near this magnitude
CLAMP = 10000


class Stockfish:
    """A persistent Stockfish UCI process used to score positions."""

    def __init__(self, path: str, depth: int) -> None:
        self.depth = depth
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait("uciok")
        self._send("isready")
        self._wait("readyok")

    def _send(self, s: str) -> None:
        assert self.p.stdin
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _wait(self, token: str) -> None:
        assert self.p.stdout
        for line in self.p.stdout:
            if line.startswith(token):
                return

    def eval_white_cp(self, fen: str, white_to_move: bool) -> int | None:
        """Return Stockfish's eval in centipawns from White's perspective."""
        assert self.p.stdout
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        score: int | None = None
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("info") and " score " in line:
                toks = line.split()
                i = toks.index("score")
                kind, val = toks[i + 1], int(toks[i + 2])
                if kind == "cp":
                    score = max(-CLAMP, min(CLAMP, val))
                else:  # mate in `val`
                    score = (MATE_CP - abs(val)) * (1 if val > 0 else -1)
            elif line.startswith("bestmove"):
                break
        if score is None:
            return None
        # Stockfish scores from the side to move; convert to White-relative.
        return score if white_to_move else -score

    def close(self) -> None:
        self._send("quit")
        self.p.wait()


def random_position(max_plies: int) -> Board:
    """Play a random legal line from the start for variety."""
    board = Board()
    for _ in range(random.randint(0, max_plies)):
        moves = generate_legal(board)
        if not moves:
            break
        board.make_move(random.choice(moves))
    return board


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate NNUE training data")
    ap.add_argument("--count", type=int, default=200000, help="positions to generate")
    ap.add_argument("--depth", type=int, default=8, help="Stockfish search depth")
    ap.add_argument("--max-plies", type=int, default=30, help="random playout length")
    ap.add_argument("--out", type=str, default="nnue/data/positions.txt")
    ap.add_argument("--stockfish", type=str, default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sf = Stockfish(args.stockfish, args.depth)
    seen: set[str] = set()
    written = 0
    start = time.perf_counter()
    try:
        with out_path.open("w") as f:
            while written < args.count:
                board = random_position(args.max_plies)
                if not generate_legal(board):
                    continue  # skip checkmate/stalemate
                fen = board.fen()
                key = fen.rsplit(" ", 2)[0]  # ignore move clocks for dedup
                if key in seen:
                    continue
                seen.add(key)
                cp = sf.eval_white_cp(fen, board.side_to_move == Color.WHITE)
                if cp is None:
                    continue
                f.write(f"{fen}|{cp}\n")
                written += 1
                if written % 1000 == 0:
                    rate = written / (time.perf_counter() - start)
                    print(f"  {written}/{args.count}  ({rate:.0f} pos/s)", file=sys.stderr)
    finally:
        sf.close()

    elapsed = time.perf_counter() - start
    print(f"wrote {written} positions to {out_path} in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
