"""Generate WDL + search-score NNUE training data via engine self-play.

chess-bote plays games against itself, entirely inside the C++ core (the
Python layer only fans games out over worker processes and writes the output).
Each game starts with a short random opening (for diversity) and is then
played out with the engine's own search; quiet positions from the
engine-played phase are sampled and labelled with:

  * the **game result** (White-relative: 1.0 win / 0.5 draw / 0.0 loss) -- the
    AlphaZero-flavoured outcome signal, noisy but unbiased, and
  * the **search score** in centipawns (White-relative) -- the depth-N eval
    the engine already computed to pick its move, a much stronger per-position
    signal than any static eval.

"Quiet" means: not in check, best move is not a capture/promotion, and the
score is not a forced mate -- a static net can't represent tactics-in-flight.

Output: one record per line, ``<fen>|<wdl>|<cp>``.
Train with ``train.py --label wdl`` (the cp column feeds --hce-mix).

Usage:
    uv run python nnue/gen_selfplay.py --count 500000 --nodes 5000 --jobs 10 \
        --out nnue/data/selfplay.txt
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from chess_bote import native


def _play_game(args: tuple) -> list[tuple[str, float, int]]:
    game_id, depth, nodes, random_plies, sample_per_game, max_plies = args
    return native.selfplay_game(
        game_id, depth, nodes, random_plies, sample_per_game, max_plies
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate WDL self-play training data")
    ap.add_argument("--count", type=int, default=500000, help="positions to keep")
    ap.add_argument("--depth", type=int, default=12,
                    help="max engine search depth per move")
    ap.add_argument("--nodes", type=int, default=5000,
                    help="node limit per move (0 = depth-limited only)")
    ap.add_argument("--random-plies", type=int, default=10, help="random opening length")
    ap.add_argument("--sample-per-game", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=300, help="game length cap")
    ap.add_argument("--jobs", type=int, default=1, help="parallel self-play workers")
    ap.add_argument("--out", type=str, default="nnue/data/selfplay.txt")
    ap.add_argument("--seed", type=int, default=0,
                    help="first game id (keep runs disjoint: gen * 10_000_000)")
    args = ap.parse_args()

    if not native.HAS_NATIVE:
        sys.exit("native core not built; self-play requires it (run `uv build`)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def game_args():
        for gid in itertools.count(args.seed):
            yield (gid, args.depth, args.nodes, args.random_plies,
                   args.sample_per_game, args.max_plies)

    written = 0
    games = 0
    start = time.perf_counter()

    bar = tqdm(total=args.count, unit="pos", desc="self-play", file=sys.stderr)

    def record(f, batch):
        nonlocal written, games
        games += 1
        n = 0
        for fen, wdl, cp in batch:
            f.write(f"{fen}|{wdl:.1f}|{cp}\n")
            written += 1
            n += 1
        bar.update(n)
        rate = written / (time.perf_counter() - start)
        bar.set_postfix(games=games, pos_s=f"{rate:.0f}")

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

    bar.close()
    elapsed = time.perf_counter() - start
    print(
        f"wrote {written:,} positions from {games:,} games to {out_path} in {elapsed:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
