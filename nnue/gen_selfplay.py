"""Generate WDL (outcome-based) NNUE training data via engine self-play.

chess-bote plays games against itself. Each game starts with a short random
opening (for diversity) and is then played out with the engine's own search;
positions from the engine-played phase are sampled and labelled with the
**game result** (White-relative: 1.0 win / 0.5 draw / 0.0 loss).

This is the AlphaZero-flavoured signal: the label is the real outcome of the
game, not any engine's evaluation. Two things to keep in mind:

  * The label is noisy -- one win/draw/loss bit is shared by every sampled
    position in the game -- so you need *many* positions for it to average out.
  * The data's ceiling is roughly the strength of the engine generating it.

Output: one record per line, ``<fen>|<wdl>`` with wdl in {0.0, 0.5, 1.0}.
Train with ``train.py --label wdl``.

Usage:
    uv run python nnue/gen_selfplay.py --count 500000 --depth 6 --jobs 6 \
        --out nnue/data/selfplay.txt
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from chess_bote import Board, find_best_move, native
from chess_bote.movegen import generate_legal, in_check
from chess_bote.types import Color, PieceType


def _insufficient_material(board: Board) -> bool:
    """True for trivially drawn material (KvK, KNvK, KBvK)."""
    if board.piece_bb[PieceType.PAWN] or board.piece_bb[PieceType.ROOK] or board.piece_bb[PieceType.QUEEN]:
        return False
    minors = board.piece_bb[PieceType.KNIGHT] | board.piece_bb[PieceType.BISHOP]
    return minors.bit_count() <= 1


def _play_game(args: tuple) -> list[tuple[str, float]]:
    game_id, depth, random_plies, sample_per_game, max_plies = args
    rng = random.Random(game_id)
    native.new_game()
    board = Board()
    history: dict[int, int] = {}
    candidates: list[str] = []  # engine-played, not-in-check positions
    result_white = 0.5  # default: draw

    ply = 0
    while True:
        moves = generate_legal(board)
        if not moves:
            if in_check(board):  # checkmate: side to move loses
                result_white = 0.0 if board.side_to_move == Color.WHITE else 1.0
            break  # else stalemate -> draw
        if board.halfmove_clock >= 100 or _insufficient_material(board):
            break  # 50-move / material draw
        key = board.zobrist_key
        history[key] = history.get(key, 0) + 1
        if history[key] >= 3 or ply >= max_plies:
            break  # threefold / length cap -> draw

        if ply < random_plies:
            move = rng.choice(moves)
        else:
            if not in_check(board):
                candidates.append(board.fen())
            move = find_best_move(board, depth=depth).best_move
            if move is None:
                break
        board.make_move(move)
        ply += 1

    rng.shuffle(candidates)
    return [(fen, result_white) for fen in candidates[:sample_per_game]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate WDL self-play training data")
    ap.add_argument("--count", type=int, default=500000, help="positions to keep")
    ap.add_argument("--depth", type=int, default=6, help="engine search depth per move")
    ap.add_argument("--random-plies", type=int, default=10, help="random opening length")
    ap.add_argument("--sample-per-game", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=300, help="game length cap")
    ap.add_argument("--jobs", type=int, default=1, help="parallel self-play workers")
    ap.add_argument("--out", type=str, default="nnue/data/selfplay.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def game_args():
        for gid in itertools.count(args.seed):
            yield (gid, args.depth, args.random_plies, args.sample_per_game, args.max_plies)

    written = 0
    games = 0
    start = time.perf_counter()

    def record(f, batch):
        nonlocal written, games
        games += 1
        for fen, wdl in batch:
            f.write(f"{fen}|{wdl:.1f}\n")
            written += 1
        if games % 200 == 0:
            rate = written / (time.perf_counter() - start)
            print(f"  {written}/{args.count}  ({games} games, {rate:.0f} pos/s)", file=sys.stderr)

    with out_path.open("w") as f:
        if args.jobs <= 1:
            for ga in game_args():
                record(f, _play_game(ga))
                if written >= args.count:
                    break
        else:
            with Pool(args.jobs) as pool:
                for batch in pool.imap_unordered(_play_game, game_args(), chunksize=4):
                    record(f, batch)
                    if written >= args.count:
                        break
                pool.terminate()

    elapsed = time.perf_counter() - start
    print(
        f"wrote {written} positions from {games} games to {out_path} in {elapsed:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
