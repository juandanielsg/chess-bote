# chess-bote

A chess engine written in Python with a native C++ core. It speaks
[UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface), so you can play
against it in any standard chess GUI.

## Setup

```bash
uv sync   # builds the native extension; run once before first launch
```

## Play against ChessBote in a GUI

The engine ships as a UCI engine. Point your GUI at the launcher script:

```
scripts/chessbote-uci.sh
```

(Full path here: `/home/dani/Desktop/chess-bote/scripts/chessbote-uci.sh`.)
It runs the engine from any working directory and keeps stdout clean for the
protocol.

### CuteChess (recommended, free, cross-platform)

1. **Tools → Settings → Engines → "+"** (add engine).
2. **Command:** browse to `scripts/chessbote-uci.sh`.
3. **Protocol:** UCI. Click **OK** — CuteChess will detect `id name chess-bote`.
4. **Game → New** (`Ctrl+N`): set White = Human (you), Black = chess-bote (or
   vice versa), pick a time control, and play.

### Arena / BanksiaGUI

Same idea: "Install/New Engine", select `scripts/chessbote-uci.sh`, choose the
UCI protocol.

### Quick sanity check from a terminal

```bash
printf 'uci\nisready\nposition startpos\ngo movetime 1000\nquit\n' | scripts/chessbote-uci.sh
```

You should see `uciok`, `readyok`, an `info` line, and a `bestmove`.

## Benchmark against another engine (cutechess-cli)

[`scripts/match.sh`](scripts/match.sh) plays a match between ChessBote and
Stockfish and logs the result — useful for tracking strength as the engine
improves.

```bash
scripts/match.sh                                 # 10 games, 10+0.1, Stockfish@1320
GAMES=20 SF_ELO=1800 scripts/match.sh            # more games, stronger opponent
GAMES=50 SF_ELO=2200 TC=5+0.05 CONCURRENCY=4 scripts/match.sh   # fast bulk run
```

Settings are environment variables: `GAMES`, `TC` (`secs+inc`), `SF_ELO`,
`CONCURRENCY`, `LOG_DIR`, `PGN`, and `CUTECHESS_CLI` / `STOCKFISH` paths.

Each run writes to `matches/`:

| file | contents |
|------|----------|
| `<timestamp>.log` | full cutechess-cli output (with a settings header) |
| `<timestamp>.pgn` | the games — open in the CuteChess GUI to review |
| `results.log`     | one summary line per run, appended over time |

Find a fair opponent by raising `SF_ELO` until ChessBote scores ~50%. The match
auto-alternates colors and adjudicates dead-won/drawn positions to finish fast.

## Other entry points

| Command | What it does |
|---------|--------------|
| `uv run chess-bote-uci`   | the UCI engine (what the launcher wraps) |
| `uv run chess-bote-bench` | perft speed benchmark (see [BENCH.md](BENCH.md)) |
| `uv run chess-bote`       | print the start position |

## Supported UCI commands

`uci`, `isready`, `ucinewgame`, `position [startpos | fen <FEN>] [moves ...]`,
`go [depth N | movetime MS | wtime/btime/winc/binc/movestogo | infinite]`,
`stop`, `quit`.

During a search the engine emits a UCI `info` line per depth with `depth`,
`score` (`cp`/`mate`), `nodes`, `nps`, `time`, and the principal variation
(`pv`), so GUIs show its evaluation and best line as it thinks.
