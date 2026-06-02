# Performance Baseline

Throughput of the core hot path — legal move generation + make/unmake —
measured via perft and reported as **NPS** (nodes per second). NPS is the
standard engine-speed metric: perft exercises movegen and make/unmake with no
eval/search noise.

Reproduce with:

```bash
uv run chess-bote-bench            # default suite, perft depth 3
uv run chess-bote-bench --depth 4  # deeper / more stable NPS
```

## Environment

| | |
|---|---|
| Date | 2026-06-02 |
| CPU | Intel Core i7-8750H @ 2.20GHz |
| Python | 3.12.3 (CPython) |
| Implementation | pure Python; magic-bitboard sliders; direct legal move generation |

## Perft NPS progression

Total NPS over the default suite (depth 3), as optimizations landed:

| stage                          | TOTAL NPS | speedup vs baseline |
|--------------------------------|-----------|---------------------|
| baseline (ray attacks, make/unmake legality filter) | 55.7K | 1.0× |
| + magic bitboards              | 73.6K     | 1.3× |
| + direct legal move generation | 355.2K    | **6.4×** |

Direct legal movegen was by far the bigger win: it removed the ~45
make/unmake calls *per node* that the old legality filter performed (a
re-profile shows `make_move` calls dropped from 4.17M to 92K on midgame
perft d4). The remaining hot path is now pure generation — `scan_forward`
(bit iteration), `Move.create` (per-move object allocation), and the gen
loops — which is the natural target for the C++ port.

## C++ native core

The hot path (bitboards, magic attacks, legal movegen, make/unmake, perft) is
ported to C++ in [`src/cpp/core.cpp`](src/cpp/core.cpp), built with
scikit-build-core + nanobind and exposed as `chess_bote._core`. It is validated
to produce identical perft counts to the Python oracle (see
[`tests/test_cpp_core.py`](tests/test_cpp_core.py)).

| engine            | perft NPS | vs pure Python |
|-------------------|-----------|----------------|
| pure Python (d3)  | ~412K     | 1×             |
| **C++ core (d5)** | **~141M** | **~344×**      |

For scale: C++ runs Kiwipete perft(5) = 193,690,690 nodes in ~1.3s.

### Native search

The full search (negamax + alpha-beta, quiescence, MVV-LVA/killers/history,
PVS, null-move, transposition table) and the evaluation are also native. Eval
parameters are loaded from [`src/chess_bote/eval.py`](src/chess_bote/eval.py) at
import (single source of truth — weights are tunable from Python with no
recompile), and `chess_bote.find_best_move` / the UCI engine use the native
search. Start-position search:

| engine        | depth 7 | search NPS |
|---------------|---------|------------|
| pure Python   | ~8.3s   | ~32K       |
| **C++ search**| **~0.04s** | **~4–6M** |

That is roughly a **150–180×** search speedup; depth 8 from the opening
completes in a fraction of a second.

> Rebuilding after a C++ change: `uv sync --reinstall-package chess-bote`
> (uv does not hash the C++ sources, so a plain `uv sync` can skip the rebuild).

## Baseline

### perft depth 3 (quick, default)

| position | nodes  | time (s) | NPS    |
|----------|--------|----------|--------|
| startpos | 8902   | 0.166    | 53.8K  |
| kiwipete | 97862  | 1.870    | 52.3K  |
| endgame  | 2812   | 0.058    | 48.8K  |
| midgame  | 89890  | 1.488    | 60.4K  |
| **TOTAL**| 199466 | 3.581    | **55.7K** |

### perft depth 4 (larger sample)

| position | nodes    | time (s) | NPS    |
|----------|----------|----------|--------|
| startpos | 197281   | 3.296    | 59.9K  |
| kiwipete | 4085603  | 68.313   | 59.8K  |
| endgame  | 43238    | 0.807    | 53.6K  |
| midgame  | 3894594  | 65.147   | 59.8K  |
| **TOTAL**| 8220716  | 137.563  | **59.8K** |

**Summary: ~55–60K NPS, pure Python.**

## Notes

- The suite positions are defined in [`src/chess_bote/bench.py`](src/chess_bote/bench.py)
  (`SUITE`): open start, tactical (Kiwipete), endgame, middlegame.
- NPS is somewhat machine- and load-dependent; depth 4 gives a steadier figure
  than depth 3.
- Done since the baseline: **magic bitboards** (in
  [`src/chess_bote/attacks.py`](src/chess_bote/attacks.py)) and **direct legal
  move generation** (in [`src/chess_bote/movegen.py`](src/chess_bote/movegen.py)).
- Remaining headroom:
  - **C++ port** of the hot path (now concentrated in `scan_forward`,
    `Move.create`, and the generation loops). A tuned C++ engine reaches tens of
    millions of NPS — still ~100× over the current pure-Python figure.

## Search

Search speed is measured separately from perft (it includes eval + move
ordering + the transposition table). The TT is a pure optimization — same best
move and score, fewer nodes — measured from the start position:

| depth | TT off (nodes) | TT on (nodes) | reduction |
|-------|----------------|---------------|-----------|
| 4     | 23,057         | 6,170         | 73%       |
| 5     | 272,442        | 94,793        | 65%       |

A warm re-search of an already-searched position returns almost immediately
(~100 nodes) because the result is cached.

### Search refinements (killers + history, PVS, null-move pruning)

Node counts from the start position, stacking the refinements on top of the TT:

| depth | no TT   | TT only | TT + refinements |
|-------|---------|---------|------------------|
| 5     | 272,442 | 94,793  | 18,422           |
| 6     | —       | —       | 40,982           |

Refinements cut nodes a further ~80% over the TT alone and make depth 6
practical from the opening. Best move and tactical results (free captures,
mate-in-1) are unchanged.

## Strength (playing, not speed)

Measured with [`scripts/match.sh`](scripts/match.sh) against Stockfish capped at
a fixed Elo, score over 40 games at 4+0.04 (see `matches/results.log`). The
opponent level (SF Elo 2200) is held constant so eval/search changes are
comparable run to run.

| eval                       | score vs SF-2200 (40g) | Elo vs opponent |
|----------------------------|------------------------|-----------------|
| material + single PST      | 46.3% (17-20-3)        | −26 ± 107       |
| **tapered material + PST** | **62.5% (22-12-6)**    | **+89 ± 105**   |

Tapered eval (midgame/endgame interpolation) is worth roughly **+115 Elo** here.
The error bars are wide at 40 games, so treat the exact number as approximate —
but the direction and magnitude match expectations and the likelihood-of-
superiority rose from 31% to 96%.

## Change log

| Date       | Change                          | TOTAL NPS (d4) |
|------------|---------------------------------|----------------|
| 2026-06-02 | Initial pure-Python baseline    | 59.8K          |
| 2026-06-02 | Transposition table (search)    | — (perft unchanged; search nodes −65–73%) |
| 2026-06-02 | Killers + history, PVS, null-move | — (startpos d5 search nodes −80% vs TT) |
| 2026-06-02 | Magic bitboards                 | 73.6K (d3)     |
| 2026-06-02 | Direct legal move generation    | 355.2K (d3)    |
| 2026-06-02 | C++ native core (perft)         | ~141M (d5) — ~344× over pure Python |
| 2026-06-02 | C++ native search + eval        | search ~4–6M NPS — ~150–180× (find_best_move/UCI) |
| 2026-06-02 | Tapered eval (PeSTO MG/EG tables) | +16 pts vs SF-2200 (46%→62%), ~+115 Elo |
| 2026-06-02 | NNUE v1 (768→256→1, refresh eval) | pipeline works, but loses 0–60 self-play vs tapered (weak: tiny low-quality data + shallow refresh eval) |
