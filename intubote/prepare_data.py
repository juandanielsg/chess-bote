"""Build IntuBote training data from the Lichess evaluation database.

Source: https://database.lichess.org/#evals -- ``lichess_db_eval.jsonl.zst``,
one JSON object per position, e.g.

    {"fen": "<4-field FEN>",
     "evals": [{"pvs": [{"cp": 31, "line": "e2e4 e7e5 ..."},
                        {"mate": -3, "line": "..."}],
                "knodes": 123456, "depth": 36}, ...]}

The file is large, so it is *streamed*: decompressed on the fly from the URL
(or a local file) and read only until ``--count`` positions are kept.

For each position the deepest evaluation is used. Each principal variation
contributes its first move and that move's value, so every sample stores up to
``--top-k`` candidate moves ranked best-first:

    tokens.npy  uint8   [N, 70]   side-to-move-relative board tokens
    moves.npy   int16   [N, K]    move-vocab indices, -1 = padding
    wins.npy    float16 [N, K]    win probability for the side to move
    meta.json                     counts and settings

Usage:
    python intubote/prepare_data.py --count 2000000 --out intubote/data/lichess2m
    python intubote/prepare_data.py --source lichess_db_eval.jsonl.zst --count 10000000 ...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding import SEQ_LEN, black_to_move, cp_to_win, encode_fen, encode_move  # noqa: E402

try:  # 2-3x faster JSON parsing when available
    import orjson as _json

    _loads = _json.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def open_lines(source: str):
    """Yield text lines from a local/remote .jsonl or .jsonl.zst, streaming."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "intubote-data-prep"})
        raw = urllib.request.urlopen(req)
    else:
        raw = open(source, "rb")
    if source.endswith(".zst"):
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20)
    yield from io.TextIOWrapper(raw, encoding="utf-8")


def pv_win(pv: dict, black: bool) -> float:
    """White-relative cp/mate of a PV -> side-to-move win probability."""
    if "mate" in pv:
        white_wins = pv["mate"] > 0
        return 1.0 if white_wins != black else 0.0
    cp = pv["cp"]
    return float(cp_to_win(-cp if black else cp))


def parse_record(rec: dict, top_k: int, min_depth: int):
    evals = rec.get("evals")
    if not evals:
        return None
    best = max(evals, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
    if best.get("depth", 0) < min_depth:
        return None
    fen = rec["fen"]
    black = black_to_move(fen)
    moves, wins, seen = [], [], set()
    for pv in best["pvs"]:
        line = pv.get("line")
        if not line:
            continue
        uci = line.split(" ", 1)[0]
        idx = encode_move(uci, black)
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        moves.append(idx)
        wins.append(pv_win(pv, black))
        if len(moves) == top_k:
            break
    if not moves:
        return None
    return fen, moves, wins


def main() -> None:
    ap = argparse.ArgumentParser(description="Build IntuBote data from Lichess evals")
    ap.add_argument("--source", default=LICHESS_EVAL_URL, help="URL or local .jsonl(.zst)")
    ap.add_argument("--count", type=int, default=1_000_000, help="positions to keep")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many input lines first (e.g. to build a disjoint test set)")
    ap.add_argument("--top-k", type=int, default=5, help="candidate moves stored per position")
    ap.add_argument("--min-depth", type=int, default=0, help="drop evals shallower than this")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, k = args.count, args.top_k
    tokens = np.zeros((n, SEQ_LEN), dtype=np.uint8)
    moves = np.full((n, k), -1, dtype=np.int16)
    wins = np.zeros((n, k), dtype=np.float16)

    kept = read = 0
    t0 = time.time()
    for line in open_lines(args.source):
        read += 1
        if read <= args.skip:
            continue
        r = parse_record(_loads(line), k, args.min_depth)
        if r is None:
            continue
        fen, mv, wn = r
        encode_fen(fen, tokens[kept])
        moves[kept, : len(mv)] = mv
        wins[kept, : len(wn)] = wn
        kept += 1
        if kept % 100_000 == 0:
            rate = kept / (time.time() - t0)
            print(f"  {kept:,} kept / {read:,} read  ({rate:,.0f} pos/s)", file=sys.stderr)
        if kept == n:
            break

    tokens, moves, wins = tokens[:kept], moves[:kept], wins[:kept]
    np.save(os.path.join(args.out, "tokens.npy"), tokens)
    np.save(os.path.join(args.out, "moves.npy"), moves)
    np.save(os.path.join(args.out, "wins.npy"), wins)
    meta = {"positions": kept, "lines_read": read, "top_k": k, "min_depth": args.min_depth,
            "source": args.source, "skip": args.skip}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {kept:,} positions to {args.out} in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
