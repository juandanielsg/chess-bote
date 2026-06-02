#!/usr/bin/env bash
#
# Self-improvement loop for the NNUE (policy/value iteration with gating).
#
# Each generation:
#   1. self-play games with the current champion eval  -> WDL-labelled data
#   2. train a candidate net on the recent data window
#   3. quantize/export the candidate
#   4. gated match: candidate vs champion (UCI EvalFile selects each net)
#   5. promote the candidate to champion only if it scores >= threshold
#
# Why this can exceed the eval it started from: the data labels are *game
# outcomes* from games played with eval+search, which is stronger than the eval
# alone. Each promoted generation raises the bar; gating prevents regressions.
#
# Settings (env vars, with defaults):
#   GENERATIONS   number of generations to run            (default 5)
#   POSITIONS     positions generated per generation      (default 150000)
#   SELFPLAY_DEPTH search depth during self-play          (default 6)
#   EPOCHS        training epochs per generation          (default 40)
#   DATA_WINDOW   train on the last N generations' data    (default 3)
#   GATE_GAMES    games in the gating match               (default 40)
#   GATE_TC       gating time control                     (default 4+0.04)
#   GATE_THRESH   min candidate score to promote          (default 0.53)
#   JOBS          parallelism for self-play and matches   (default 6)
#   START_CHAMPION starting champion eval: "handcrafted" or a net .bin path
#                  (default handcrafted; set to a distilled net for option C)
#   RUN_DIR       output dir            (default nnue/loop_runs/<timestamp>)
#   CUTECHESS_CLI path to cutechess-cli
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GENERATIONS="${GENERATIONS:-5}"
POSITIONS="${POSITIONS:-150000}"
SELFPLAY_DEPTH="${SELFPLAY_DEPTH:-6}"
EPOCHS="${EPOCHS:-40}"
DATA_WINDOW="${DATA_WINDOW:-3}"
GATE_GAMES="${GATE_GAMES:-40}"
GATE_TC="${GATE_TC:-4+0.04}"
GATE_THRESH="${GATE_THRESH:-0.53}"
JOBS="${JOBS:-6}"
CUTECHESS_CLI="${CUTECHESS_CLI:-/home/dani/Desktop/cutechess/cutechess/build/cutechess-cli}"
RUN_DIR="${RUN_DIR:-nnue/loop_runs/$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/loop.log"
echo "run dir: $RUN_DIR" | tee "$LOG"

# Champion eval: "handcrafted", or a net path (a distilled net for option C).
champion="${START_CHAMPION:-handcrafted}"
echo "starting champion: $champion" | tee -a "$LOG"

for gen in $(seq 1 "$GENERATIONS"); do
    gdir="$RUN_DIR/gen$gen"
    mkdir -p "$gdir"
    echo "=== generation $gen (champion: $champion) ===" | tee -a "$LOG"

    # 1. Self-play with the champion eval.
    if [ "$champion" = "handcrafted" ]; then
        unset CHESSBOTE_NNUE || true
    else
        export CHESSBOTE_NNUE="$champion"
    fi
    uv run python nnue/gen_selfplay.py --count "$POSITIONS" --depth "$SELFPLAY_DEPTH" \
        --jobs "$JOBS" --out "$gdir/data.txt" --seed "$gen" 2>>"$LOG"
    unset CHESSBOTE_NNUE || true

    # 2. Training data = the last DATA_WINDOW generations (replay buffer).
    : > "$gdir/train.txt"
    for ((g = gen; g > gen - DATA_WINDOW && g >= 1; g--)); do
        [ -f "$RUN_DIR/gen$g/data.txt" ] && cat "$RUN_DIR/gen$g/data.txt" >> "$gdir/train.txt"
    done

    # 3. Train + 4. export the candidate.
    uv run --group train python nnue/train.py --data "$gdir/train.txt" --label wdl \
        --epochs "$EPOCHS" --out "$gdir/net.pt" 2>>"$LOG"
    uv run --group train python nnue/export.py --in "$gdir/net.pt" --out "$gdir/net.bin" \
        >>"$LOG" 2>&1
    candidate="$PROJECT_DIR/$gdir/net.bin"

    # 5. Gated match: candidate vs champion (EvalFile picks each side's eval;
    #    "handcrafted" is the sentinel the engine maps to the hand-crafted eval).
    "$CUTECHESS_CLI" \
        -engine name=candidate cmd=scripts/chessbote-uci.sh proto=uci "option.EvalFile=$candidate" \
        -engine name=champion cmd=scripts/chessbote-uci.sh proto=uci "option.EvalFile=$champion" \
        -each tc="$GATE_TC" \
        -games "$GATE_GAMES" -repeat -recover -concurrency "$JOBS" \
        -resign movecount=3 score=700 -draw movenumber=50 movecount=8 score=10 \
        -pgnout "$gdir/gate.pgn" > "$gdir/gate.log" 2>&1

    score="$(grep '^Score of candidate vs champion' "$gdir/gate.log" | tail -1 |
        sed -E 's/.*\[([0-9.]+)\].*/\1/')"
    score="${score:-0}"

    # 6. Promote if the candidate clearly beat the champion.
    if awk "BEGIN{exit !($score >= $GATE_THRESH)}"; then
        champion="$candidate"
        cp "$candidate" "$RUN_DIR/champion.bin"
        verdict="PROMOTED -> champion"
    else
        verdict="rejected (kept champion)"
    fi
    echo "gen $gen: candidate scored $score vs champion  =>  $verdict" | tee -a "$LOG"
done

echo "=== done. final champion: $champion ===" | tee -a "$LOG"
if [ -f "$RUN_DIR/champion.bin" ]; then
    echo "champion net: $RUN_DIR/champion.bin" | tee -a "$LOG"
else
    echo "no candidate beat the hand-crafted eval (champion unchanged)" | tee -a "$LOG"
fi
