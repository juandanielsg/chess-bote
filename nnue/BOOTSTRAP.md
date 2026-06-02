# Option C — Distill-then-iterate (make the NNUE actually competitive)

The self-play loop (`scripts/nnue_loop.sh`) can only *climb* once the net is
already roughly as strong as the engine's hand-crafted (PeSTO tapered) eval —
because a candidate is promoted only if it **beats the current champion**. A
from-scratch 768→256 net trained on low-depth WDL self-play starts far below
that bar, so it keeps getting rejected.

The fix is a two-phase plan:

1. **Distill Stockfish** into a net (`--label cp`) so it *reaches* the
   hand-crafted level in one shot. This sidesteps the "must first reach the bar"
   problem — Stockfish is ~3000+, so the cap is irrelevant to us.
2. **Iterate with self-play**, starting the loop *from that distilled net*, to
   push *beyond* the teacher.

Everything below uses tooling that already exists. Artifacts land in `nnue/`
and are git-ignored (`*.pt`, `*.bin`, `nnue/data/`, `nnue/optionc_runs/`).

---

## TL;DR — run the whole thing overnight (one command)

[`scripts/nnue_optionc.sh`](../scripts/nnue_optionc.sh) does all of Phase 1 and
Phase 2 below, unattended. Launch it so it survives a closed terminal **and**
keeps the machine awake:

```bash
systemd-inhibit --what=idle:sleep:handle-lid-switch --why="chess-bote NNUE" \
  nohup scripts/nnue_optionc.sh > optionc.out 2>&1 &
```

- Everything is logged to `nnue/optionc_runs/<timestamp>/optionc.log`.
- Per-generation verdicts: `grep "candidate scored" nnue/optionc_runs/*/loop/loop.log`.
- Final net (if anything was promoted): `nnue/optionc_runs/<timestamp>/loop/champion.bin`.
- Tunables (env vars) are documented in the script header; defaults aim at a
  ~one-night run. Scale `DISTILL_POSITIONS`, `DISTILL_DEPTH`, and `GENERATIONS`
  up for a stronger result.
- **Suspend caveat:** the `systemd-inhibit` flags block *automatic* idle/lid
  sleep, but a *manual* "Suspend" still freezes everything (the run pauses and
  resumes on wake — it won't die). Keep the laptop plugged in.

The sections below explain the same steps manually, for tuning or debugging.

---

## Phase 1 — Distill Stockfish into `net0`

### 1. Generate distillation data (Stockfish-labelled, quiet-filtered)

```bash
uv run python nnue/gen_data.py \
  --count 500000 --depth 10 \
  --quiet-filter --max-spread 80 \
  --out nnue/data/distill.txt
```

- `--quiet-filter` keeps only quiet positions (best move not a capture, side not
  in check, best-vs-second-best gap ≤ 80 cp) — meaningful static-eval targets.
- Time: roughly **3–5 hours** at depth 10 (≈30 quiet pos/s). Lower `--depth` (8)
  to go faster; raise `--count` (1–2M) for a stronger net if you have time.
- Tip: run it in the background and let it finish overnight.

> Better data still helps here. Random playouts (the current source) are broad
> but unrealistic; for higher quality, point a future version of `gen_data.py`
> at a PGN database or engine self-play positions, then quiet-filter + label.

### 2. Train `net0` on the eval (cp) labels

```bash
uv run --group train python nnue/train.py \
  --data nnue/data/distill.txt --label cp \
  --epochs 60 --batch 4096 --lr 0.01 \
  --out nnue/net0.pt
```

Watch `wdl-mse` fall; more data + epochs = better fit to Stockfish.

### 3. Quantize / export to the engine format

```bash
uv run --group train python nnue/export.py --in nnue/net0.pt --out nnue/net0.bin
```

The printed `float cp` vs `int cp` columns should be close (quantization sanity).

### 4. Sanity A/B: does `net0` beat the hand-crafted eval?

```bash
CC=/home/dani/Desktop/cutechess/cutechess/build/cutechess-cli
"$CC" \
  -engine name=net0 cmd=scripts/chessbote-uci.sh proto=uci "option.EvalFile=$PWD/nnue/net0.bin" \
  -engine name=handcrafted cmd=scripts/chessbote-uci.sh proto=uci option.EvalFile=handcrafted \
  -each tc=8+0.08 -games 100 -repeat -recover -concurrency 6 \
  -resign movecount=3 score=700 -draw movenumber=50 movecount=8 score=10
```

Read the `Score of net0 vs handcrafted: …` line.
- **≥ ~53%** → `net0` is your new baseline; go to Phase 2.
- **< 50%** → the net isn't strong enough yet. Most effective fixes, in order:
  more/better distillation data; more epochs; then a **bigger net** (see below).

---

## Phase 2 — Iterate beyond the teacher with self-play

Start the loop **from the distilled net** so each generation is gated against a
real champion and can only improve:

```bash
START_CHAMPION="$PWD/nnue/net0.bin" \
GENERATIONS=10 POSITIONS=200000 SELFPLAY_DEPTH=8 \
EPOCHS=40 GATE_GAMES=60 GATE_TC=8+0.08 GATE_THRESH=0.53 JOBS=6 \
scripts/nnue_loop.sh
```

Each generation: self-play (with the champion) → train candidate → export →
**gated 60-game match** vs the champion → promote only if it scores ≥ 53%.
The final champion is copied to `<RUN_DIR>/champion.bin`; per-generation logs,
nets, and PGNs are kept under `nnue/loop_runs/<timestamp>/`.

Notes:
- **Deeper `SELFPLAY_DEPTH` = stronger outcome labels = bigger steps** (but
  slower generation). 8 is a reasonable balance; 6 is faster but climbs slower.
- Expect **rejections**; that's the gate doing its job (no regressions). Net
  gains come from the *promoted* generations.
- `DATA_WINDOW` (default 3) trains on the last N generations' data — a replay
  buffer for stability. Increase it if training looks unstable.

---

## Advanced — a bigger net (often the real unlock)

768→256 is small. A larger hidden layer can represent much more. The hidden
size is defined in **three places that must agree**, then the C++ must be
rebuilt:

1. `nnue/train.py` — `HIDDEN = 256`
2. `nnue/nnue_infer.py` — `HIDDEN = 256`
3. `src/cpp/core.cpp` — `constexpr int NN_HIDDEN = 256;`

Change all three to the same value (e.g. 512), then:

```bash
uv sync --reinstall-package chess-bote   # rebuild the C++ with the new size
```

Retrain + re-export afterwards (old `.pt`/`.bin` files are the old size and
won't load). Larger nets are slower per eval (the v1 inference is "refresh"
per node); if speed becomes the bottleneck, the next engine-side step is an
**incremental accumulator** (update on make/unmake) — the current refresh path
is the oracle to validate that against.

---

## Why this works (one line)

WDL labels are *game outcomes from searched games*, which are stronger than the
eval that produced them — so each promoted generation lifts the ceiling, while
gating guarantees the engine never gets worse.
