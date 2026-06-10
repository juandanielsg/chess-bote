#!/usr/bin/env bash
#
# Self-improvement loop for the NNUE (policy/value iteration with gating).
#
# Each generation:
#   1. self-play games with the current champion eval  -> WDL + search-score data
#   2. train a candidate net on the recent data window
#   3. quantize/export the candidate
#   4. gated match: candidate vs champion (UCI EvalFile selects each net)
#   5. promote the candidate to champion only if it scores >= threshold
#
# Why this can exceed the eval it started from: the data labels are *game
# outcomes* and *search scores* from games played with eval+search, which is
# stronger than the eval alone. Each promoted generation raises the bar;
# gating prevents regressions.
#
# The gating match is played at fixed nodes per move (GATE_NODES), not on a
# clock, so it measures eval quality rather than eval speed and is immune to
# machine-load noise. Set GATE_NODES=0 to gate on time (GATE_TC) instead.
#
# Settings (env vars, with defaults):
#   GENERATIONS    number of generations to run            (default 5)
#   POSITIONS      positions generated per generation      (default 150000)
#   SELFPLAY_NODES node limit per self-play move           (default 5000)
#   SELFPLAY_DEPTH max search depth per self-play move     (default 12)
#   EPOCHS         training epochs per generation          (default 20)
#   CP_MIX         search-score vs WDL label mix (0..1)    (default 0.5)
#   DATA_WINDOW    train on the last N generations' data   (default 3)
#   GATE_GAMES     games in the gating match               (default 200)
#   GATE_NODES     nodes per move in the gating match      (default 20000)
#   GATE_TC        gating time control if GATE_NODES=0     (default 4+0.04)
#   GATE_THRESH    min candidate score to promote          (default 0.53)
#   JOBS           parallelism for self-play               (default 10)
#   GATE_JOBS      parallelism for the gating match        (default JOBS)
#   START_CHAMPION starting champion eval: "handcrafted" or a net .bin path
#                  (default handcrafted; set to a distilled net for option C)
#   RUN_DIR        output dir            (default nnue/loop_runs/<timestamp>)
#   CUTECHESS_CLI  path to cutechess-cli
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GENERATIONS="${GENERATIONS:-5}"
POSITIONS="${POSITIONS:-150000}"
SELFPLAY_NODES="${SELFPLAY_NODES:-5000}"
SELFPLAY_DEPTH="${SELFPLAY_DEPTH:-12}"
EPOCHS="${EPOCHS:-20}"
CP_MIX="${CP_MIX:-0.5}"
DATA_WINDOW="${DATA_WINDOW:-3}"
GATE_GAMES="${GATE_GAMES:-200}"
GATE_NODES="${GATE_NODES:-20000}"
GATE_TC="${GATE_TC:-4+0.04}"
GATE_THRESH="${GATE_THRESH:-0.53}"
JOBS="${JOBS:-10}"
GATE_JOBS="${GATE_JOBS:-$JOBS}"
CUTECHESS_CLI="${CUTECHESS_CLI:-/home/dani/Desktop/cutechess/cutechess/build/cutechess-cli}"
RUN_DIR="${RUN_DIR:-nnue/loop_runs/$(date +%Y%m%d-%H%M%S)}"

if [ "$GATE_NODES" -gt 0 ]; then
    GATE_LIMIT=(tc=inf nodes="$GATE_NODES")
else
    GATE_LIMIT=(tc="$GATE_TC")
fi

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

    # 1. Self-play with the champion eval. Game-id seeds are spaced far apart
    #    so generations never replay each other's openings.
    if [ "$champion" = "handcrafted" ]; then
        unset CHESSBOTE_NNUE || true
    else
        export CHESSBOTE_NNUE="$champion"
    fi
    uv run python nnue/gen_selfplay.py --count "$POSITIONS" \
        --depth "$SELFPLAY_DEPTH" --nodes "$SELFPLAY_NODES" \
        --jobs "$JOBS" --out "$gdir/data.txt" \
        --seed $((gen * 10000000)) 2> >(tee -a "$LOG" >&2)
    unset CHESSBOTE_NNUE || true

    # 2. Training data = the last DATA_WINDOW generations (replay buffer).
    : > "$gdir/train.txt"
    for ((g = gen; g > gen - DATA_WINDOW && g >= 1; g--)); do
        [ -f "$RUN_DIR/gen$g/data.txt" ] && cat "$RUN_DIR/gen$g/data.txt" >> "$gdir/train.txt"
    done

    # 3. Train + 4. export the candidate.
    uv run --group train python nnue/train.py --data "$gdir/train.txt" --label wdl \
        --cp-mix "$CP_MIX" --epochs "$EPOCHS" --out "$gdir/net.pt" \
        2> >(tee -a "$LOG" >&2)
    uv run --group train python nnue/export.py --in "$gdir/net.pt" --out "$gdir/net.bin" \
        >>"$LOG" 2>&1
    candidate="$PROJECT_DIR/$gdir/net.bin"

    # 5. Gated match: candidate vs champion (EvalFile picks each side's eval;
    #    "handcrafted" is the sentinel the engine maps to the hand-crafted eval).
    "$CUTECHESS_CLI" \
        -engine name=candidate cmd=scripts/chessbote-uci.sh proto=uci "option.EvalFile=$candidate" \
        -engine name=champion cmd=scripts/chessbote-uci.sh proto=uci "option.EvalFile=$champion" \
        -each "${GATE_LIMIT[@]}" \
        -games "$GATE_GAMES" -repeat -recover -concurrency "$GATE_JOBS" \
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
