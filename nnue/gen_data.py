"""Generate NNUE training data: (FEN, white-relative centipawn score) pairs.

Positions come from short random playouts (using chess-bote's own legal move
generator) for variety, and are labelled by a persistent Stockfish process at a
fixed search depth. One Stockfish process is reused across all positions, which
is far faster than spawning it per position.

Quiet-position filtering (``--quiet-filter``)
---------------------------------------------
Static evaluation is only meaningful in "quiet" positions; a position captured
mid-tactic gets a label (the search eval) that reflects a tactic the static net
cannot see. With ``--quiet-filter`` a position is kept only if:

  * the side to move is not in check,
  * it has at least two legal moves (not forced),
  * the best move is *quiet* (not a capture or promotion), and
  * the gap between the best and second-best move is small
    (``<= --max-spread`` cp) -- i.e. no single move dominates.

The best-vs-second-best gap (not best-vs-worst: a terrible move exists almost
everywhere, so that gap is huge everywhere) needs Stockfish MultiPV, which is
enabled automatically when filtering.

Usage:
    uv run python nnue/gen_data.py --count 200000 --depth 8 \
        --quiet-filter --max-spread 80 --out nnue/data/positions.txt

Output: one record per line, ``<fen>|<cp>``, ``cp`` from White's perspective.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from chess_bote.board import Board
from chess_bote.movegen import generate_legal, in_check, parse_move
from chess_bote.types import Color

MATE_CP = 10000  # mates mapped near this magnitude
CLAMP = 10000


class Stockfish:
    """A persistent Stockfish UCI process used to analyse positions."""

    def __init__(self, path: str, depth: int, multipv: int = 1) -> None:
        self.depth = depth
        self.multipv = multipv
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait("uciok")
        self._send(f"setoption name MultiPV value {multipv}")
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

    def analyse(self, fen: str) -> list[tuple[int, str]]:
        """Return ``[(score, first_move_uci), ...]`` for the top moves, best first.

        Scores are centipawns from the side-to-move's perspective (mates mapped
        to large values); the list is ordered by MultiPV rank (index 0 = best).
        """
        assert self.p.stdout
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        lines: dict[int, tuple[int, str]] = {}
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("bestmove"):
                break
            if not line.startswith("info") or " multipv " not in line or " pv " not in line:
                continue
            toks = line.split()
            try:
                mi = int(toks[toks.index("multipv") + 1])
                si = toks.index("score")
                kind, val = toks[si + 1], int(toks[si + 2])
                if kind == "cp":
                    score = max(-CLAMP, min(CLAMP, val))
                else:  # mate in `val`
                    score = (MATE_CP - abs(val)) * (1 if val > 0 else -1)
                move = toks[toks.index("pv") + 1]
            except (ValueError, IndexError):
                continue
            lines[mi] = (score, move)  # keep the deepest line per rank
        return [lines[k] for k in sorted(lines)]

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


def is_quiet(board: Board, info: list[tuple[int, str]], max_spread: int) -> bool:
    """Quiet-position test (see module docstring)."""
    if not info:
        return False
    best_score, best_uci = info[0]
    best = parse_move(board, best_uci)
    if best is None or best.is_capture() or best.is_promotion():
        return False  # best move is tactical
    if len(info) >= 2 and (best_score - info[1][0]) > max_spread:
        return False  # one move dominates -> likely a tactic
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate NNUE training data")
    ap.add_argument("--count", type=int, default=200000, help="positions to keep")
    ap.add_argument("--depth", type=int, default=8, help="Stockfish search depth")
    ap.add_argument("--max-plies", type=int, default=30, help="random playout length")
    ap.add_argument("--out", type=str, default="nnue/data/positions.txt")
    ap.add_argument("--stockfish", type=str, default="/usr/games/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--quiet-filter", action="store_true", help="keep only quiet positions"
    )
    ap.add_argument(
        "--max-spread",
        type=int,
        default=80,
        help="max best-vs-second-best cp gap when filtering",
    )
    ap.add_argument("--multipv", type=int, default=1, help="Stockfish MultiPV lines")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Filtering needs the second-best move, so ensure MultiPV >= 2.
    multipv = max(args.multipv, 2) if args.quiet_filter else args.multipv

    sf = Stockfish(args.stockfish, args.depth, multipv)
    seen: set[str] = set()
    written = 0
    candidates = 0
    start = time.perf_counter()
    try:
        with out_path.open("w") as f:
            while written < args.count:
                board = random_position(args.max_plies)
                legal = generate_legal(board)
                if args.quiet_filter:
                    if len(legal) < 2 or in_check(board):
                        continue
                elif not legal:
                    continue

                fen = board.fen()
                key = fen.rsplit(" ", 2)[0]  # ignore move clocks for dedup
                if key in seen:
                    continue
                seen.add(key)
                candidates += 1

                info = sf.analyse(fen)
                if not info:
                    continue
                if args.quiet_filter and not is_quiet(board, info, args.max_spread):
                    continue

                best = info[0][0]  # side-to-move relative
                white_cp = best if board.side_to_move == Color.WHITE else -best
                f.write(f"{fen}|{white_cp}\n")
                written += 1
                if written % 1000 == 0:
                    rate = written / (time.perf_counter() - start)
                    acc = 100 * written / candidates
                    print(
                        f"  {written}/{args.count}  ({rate:.0f} pos/s, {acc:.0f}% kept)",
                        file=sys.stderr,
                    )
    finally:
        sf.close()

    elapsed = time.perf_counter() - start
    accept = (100 * written / candidates) if candidates else 0.0
    print(
        f"wrote {written} positions to {out_path} in {elapsed:.1f}s "
        f"({accept:.0f}% of {candidates} candidates kept)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
