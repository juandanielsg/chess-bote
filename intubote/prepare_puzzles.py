"""Build IntuBote puzzle data: a fine-tuning set and a held-out benchmark.

Source: https://database.lichess.org/#puzzles -- ``lichess_db_puzzle.csv.zst``
(~5.5M puzzles), streamed like ``prepare_data.py``. Each row has

    PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, ...

``FEN`` is the position *before* the opponent's last move; ``Moves`` starts
with that opponent move, then alternates solver move / opponent reply. So the
solver's moves are at odd indices, and each one is a training position:
"in this position, the solution move is X".

Split (by a hash of the puzzle id, so it never depends on ``--count``):
  * 1% of puzzles form the **benchmark pool** and are never trained on. Up to
    ``--bench-per-band`` well-established puzzles per rating band are kept as
    the benchmark; reading continues past ``--count`` until every band is
    full (or the file ends), so the benchmark is the same for every run.
  * the rest are training puzzles, up to ``--count``.

Outputs (in ``--out``):
    tokens.npy / moves.npy / wins.npy / meta.json
        same format as prepare_data.py, so train.py can mix them in with
        ``--extra``. K = 1 (the solution move); wins are NaN ("no value
        target"), so the value head is trained on the base data only.
    bench.npz            benchmark tensors, read by puzzle_bench.py
    bench_puzzles.jsonl  the benchmark puzzles (id, rating, themes, url, ...)

Usage:
    python intubote/prepare_puzzles.py --count 2000000 --out intubote/data/puzzles
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding import NUM_MOVES, SEQ_LEN, black_to_move, encode_fen, encode_move  # noqa: E402

from chess_bote.board import Board  # noqa: E402
from chess_bote.move import (  # noqa: E402
    FLAG_CAPTURE, FLAG_DOUBLE_PAWN, FLAG_EN_PASSANT, FLAG_KING_CASTLE, FLAG_QUEEN_CASTLE,
    FLAG_QUIET, Move,
)
from chess_bote.movegen import generate_legal, in_check  # noqa: E402
from chess_bote.types import PieceType  # noqa: E402

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

# Rating bands for the benchmark report: (label, low, high) with low <= r < high.
BANDS = [("<1000", 0, 1000), ("1000-1399", 1000, 1400), ("1400-1799", 1400, 1800),
         ("1800-2199", 1800, 2200), ("2200-2599", 2200, 2600), ("2600+", 2600, 10_000)]

_EP_NONE = 15  # token id for "no en-passant file" (see encoding.py)


def band_of(rating: int) -> int:
    for i, (_, lo, hi) in enumerate(BANDS):
        if lo <= rating < hi:
            return i
    return len(BANDS) - 1


def in_bench_pool(puzzle_id: str) -> bool:
    """1% of puzzles, chosen by id: never used for training."""
    return zlib.crc32(puzzle_id.encode()) % 100 == 0


def open_rows(source: str):
    """Yield dict rows from a local/remote .csv or .csv.zst, streaming."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "intubote-data-prep"})
        raw = urllib.request.urlopen(req)
    else:
        raw = open(source, "rb")
    if source.endswith(".zst"):
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20)
    yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


def _sq(name: str) -> int:
    return (ord(name[0]) - 97) + 8 * (int(name[1]) - 1)


def uci_to_move(board: Board, uci: str) -> Move:
    """Build the Move for a UCI string *without* generating legal moves.

    Much faster than ``parse_move`` (which runs the move generator), and safe for
    Lichess puzzle lines, which are known to be legal. Tests check it against
    ``parse_move``.
    """
    frm, to = _sq(uci[0:2]), _sq(uci[2:4])
    pt = board.mailbox[frm]
    capture = board.mailbox[to] != PieceType.NONE
    if len(uci) == 5:
        flag = 0b1000 | "nbrq".index(uci[4]) | (0b0100 if capture else 0)
    elif pt == PieceType.KING and abs(to % 8 - frm % 8) == 2:
        flag = FLAG_KING_CASTLE if to % 8 == 6 else FLAG_QUEEN_CASTLE
    elif pt == PieceType.PAWN and not capture and frm % 8 != to % 8:
        flag = FLAG_EN_PASSANT
    elif pt == PieceType.PAWN and abs(to - frm) == 16:
        flag = FLAG_DOUBLE_PAWN
    else:
        flag = FLAG_CAPTURE if capture else FLAG_QUIET
    return Move.create(frm, to, flag)


def encode_board(board: Board, out: np.ndarray | None = None) -> np.ndarray:
    """Tokens for a Board. The en-passant file is kept only when a pawn of the side
    to move could actually capture there, like the FENs in Lichess's databases."""
    fen = board.fen()
    tok = encode_fen(fen, out)
    ep = fen.split()[3]
    if ep != "-":
        f, r = ord(ep[0]) - 97, int(ep[1]) - 1
        pawn_rank = r - 1 if r == 5 else r + 1  # rank of the capturing pawns
        us = board.side_to_move
        capturable = any(
            0 <= ff < 8 and board.piece_at(pawn_rank * 8 + ff) == (us, PieceType.PAWN)
            for ff in (f - 1, f + 1))
        if not capturable:
            tok[69] = _EP_NONE
    return tok


def _mating_moves(board: Board) -> list[str]:
    mates = []
    for m in generate_legal(board):
        board.make_move(m)
        if in_check(board) and not generate_legal(board):
            mates.append(m.uci())
        board.unmake_move(m)
    return mates


def puzzle_positions(fen: str, moves: list[str], bench: bool = False):
    """Yield one entry per solver move: (tokens, black_to_move, solution_uci, extra).

    With ``bench=True``, ``extra`` is ``(legal_ucis, accepted_ucis)``. Accepted is
    the solution move, plus -- on the final move -- any other mating move, which
    Lichess also accepts. Otherwise ``extra`` is None.
    """
    board = Board(fen)
    for i, uci in enumerate(moves):
        if i % 2 == 1:  # solver's turn
            black = black_to_move(board.fen())
            extra = None
            if bench:
                legal = [m.uci() for m in generate_legal(board)]
                accepted = {uci}
                if i == len(moves) - 1:
                    accepted.update(_mating_moves(board))
                extra = (legal, sorted(accepted))
            yield encode_board(board), black, uci, extra
        board.make_move(uci_to_move(board, uci))


def _pack(ucis, black: bool) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for u in ucis:
        idx = encode_move(u, black)
        if idx >= 0:
            mask[idx] = True
    return np.packbits(mask)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build IntuBote puzzle data and benchmark")
    ap.add_argument("--source", default=LICHESS_PUZZLE_URL, help="URL or local .csv(.zst)")
    ap.add_argument("--count", type=int, default=2_000_000, help="training puzzles to keep")
    ap.add_argument("--bench-per-band", type=int, default=2000,
                    help="benchmark puzzles per rating band")
    ap.add_argument("--bench-max-rd", type=int, default=90,
                    help="benchmark only puzzles whose rating deviation is at most this")
    ap.add_argument("--bench-min-plays", type=int, default=100,
                    help="benchmark only puzzles played at least this often")
    ap.add_argument("--min-popularity", type=int, default=0,
                    help="skip training puzzles rated below this by players (-100..100)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Training arrays grow in chunks (puzzles have 1-5+ solver moves each).
    chunk = 1 << 20
    tokens = np.zeros((chunk, SEQ_LEN), dtype=np.uint8)
    moves = np.zeros(chunk, dtype=np.int16)
    n_pos = n_train = 0

    bench_tok, bench_tgt, bench_legal, bench_ok, bench_pz = [], [], [], [], []
    bench_rows = []
    band_counts = [0] * len(BANDS)

    def bench_full() -> bool:
        return all(c >= args.bench_per_band for c in band_counts)

    read = skipped = 0
    t0 = time.time()
    for row in open_rows(args.source):
        read += 1
        pid = row["PuzzleId"]
        line = row["Moves"].split()
        if in_bench_pool(pid):
            rating = int(row["Rating"])
            b = band_of(rating)
            if (band_counts[b] >= args.bench_per_band
                    or int(row["RatingDeviation"]) > args.bench_max_rd
                    or int(row["NbPlays"]) < args.bench_min_plays):
                continue
            p_index = len(bench_rows)
            for tok, black, uci, (legal, accepted) in puzzle_positions(row["FEN"], line, True):
                bench_tok.append(tok)
                bench_tgt.append(encode_move(uci, black))
                bench_legal.append(_pack(legal, black))
                bench_ok.append(_pack(accepted, black))
                bench_pz.append(p_index)
            bench_rows.append({"id": pid, "rating": rating, "themes": row["Themes"].split(),
                               "fen": row["FEN"], "moves": row["Moves"], "url": row["GameUrl"]})
            band_counts[b] += 1
        elif n_train < args.count:
            if int(row["Popularity"]) < args.min_popularity:
                skipped += 1
                continue
            for tok, black, uci, _ in puzzle_positions(row["FEN"], line):
                idx = encode_move(uci, black)
                if idx < 0:  # cannot happen for legal moves; be safe
                    continue
                if n_pos == len(tokens):
                    tokens = np.concatenate([tokens, np.zeros_like(tokens[:chunk])])
                    moves = np.concatenate([moves, np.zeros_like(moves[:chunk])])
                tokens[n_pos] = tok
                moves[n_pos] = idx
                n_pos += 1
            n_train += 1
            if n_train % 100_000 == 0:
                rate = n_train / (time.time() - t0)
                print(f"  {n_train:,} training puzzles ({n_pos:,} positions), "
                      f"{len(bench_rows):,} benchmark, {read:,} read  ({rate:,.0f} puzzles/s)",
                      file=sys.stderr, flush=True)
        if n_train >= args.count and bench_full():
            break

    out = args.out
    np.save(os.path.join(out, "tokens.npy"), tokens[:n_pos])
    np.save(os.path.join(out, "moves.npy"), moves[:n_pos, None])
    np.save(os.path.join(out, "wins.npy"), np.full((n_pos, 1), np.nan, dtype=np.float16))
    ratings = np.array([r["rating"] for r in bench_rows], dtype=np.int16)
    np.savez_compressed(
        os.path.join(out, "bench.npz"),
        tokens=np.array(bench_tok, dtype=np.uint8).reshape(-1, SEQ_LEN),
        target=np.array(bench_tgt, dtype=np.int16),
        legal=np.array(bench_legal, dtype=np.uint8).reshape(-1, (NUM_MOVES + 7) // 8),
        accepted=np.array(bench_ok, dtype=np.uint8).reshape(-1, (NUM_MOVES + 7) // 8),
        puzzle=np.array(bench_pz, dtype=np.int32),
        rating=ratings,
    )
    with open(os.path.join(out, "bench_puzzles.jsonl"), "w") as f:
        for r in bench_rows:
            f.write(json.dumps(r) + "\n")
    meta = {
        "positions": n_pos, "training_puzzles": n_train, "skipped_unpopular": skipped,
        "rows_read": read, "top_k": 1, "source": args.source,
        "benchmark": {"puzzles": len(bench_rows), "positions": len(bench_tgt),
                      "per_band": {BANDS[i][0]: c for i, c in enumerate(band_counts)},
                      "per_band_target": args.bench_per_band, "max_rd": args.bench_max_rd,
                      "min_plays": args.bench_min_plays},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {n_train:,} training puzzles ({n_pos:,} positions) and a "
          f"{len(bench_rows):,}-puzzle benchmark to {out} in {time.time() - t0:.0f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
