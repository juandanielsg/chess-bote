#!/usr/bin/env bash
#
# Option C, end-to-end and unattended: distill Stockfish into a net, check it
# beats the hand-crafted eval, then run the gated self-play iteration loop
# starting from it. Designed to run overnight.
#
#   Phase 1  generate distillation data (Stockfish-labelled, quiet-filtered)
#            -> train net0 (cp labels) -> export net0.bin
#            -> A/B net0 vs hand-crafted
#   Phase 2  run scripts/nnue_loop.sh starting from net0 (or hand-crafted if
#            net0 didn't clear the bar)
#
# RUN IT OVERNIGHT (keeps the machine awake + survives terminal close + logs):
#
#   systemd-inhibit --what=idle:sleep:handle-lid-switch --why="chess-bote NNUE" \
#     nohup scripts/nnue_optionc.sh > optionc.out 2>&1 &
#
# NOTE: this prevents *automatic* idle/lid sleep. A manual "Suspend" still
# freezes everything (the run pauses, resumes on wake). Stay plugged in.
#
# All settings are env vars; defaults aim at a ~one-night run. Scale up for a
# stronger net (more distill positions/depth, more generations).
#
#   Phase 1 (distillation):
#     DISTILL_POSITIONS  (default 300000)   DISTILL_DEPTH (default 8)
#     MAX_SPREAD (80)    NET0_EPOCHS (60)    NET0_BATCH (4096)
#     AB_GAMES (100)     AB_TC (8+0.08)      AB_THRESH (0.53)
#   Phase 2 (iteration, passed to nnue_loop.sh):
#     GENERATIONS (6)    POSITIONS (150000)  SELFPLAY_DEPTH (7)  EPOCHS (40)
#     GATE_GAMES (60)    GATE_TC (8+0.08)    GATE_THRESH (0.53)  DATA_WINDOW (3)
#   Shared: JOBS (6), CUTECHESS_CLI, RUN_DIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DISTILL_POSITIONS="${DISTILL_POSITIONS:-300000}"
DISTILL_DEPTH="${DISTILL_DEPTH:-8}"
MAX_SPREAD="${MAX_SPREAD:-80}"
NET0_EPOCHS="${NET0_EPOCHS:-60}"
NET0_BATCH="${NET0_BATCH:-4096}"
AB_GAMES="${AB_GAMES:-100}"
AB_TC="${AB_TC:-8+0.08}"
AB_THRESH="${AB_THRESH:-0.53}"

GENERATIONS="${GENERATIONS:-6}"
POSITIONS="${POSITIONS:-150000}"
SELFPLAY_DEPTH="${SELFPLAY_DEPTH:-7}"
EPOCHS="${EPOCHS:-40}"
GATE_GAMES="${GATE_GAMES:-60}"
GATE_TC="${GATE_TC:-8+0.08}"
GATE_THRESH="${GATE_THRESH:-0.53}"
DATA_WINDOW="${DATA_WINDOW:-3}"

JOBS="${JOBS:-6}"
CUTECHESS_CLI="${CUTECHESS_CLI:-/home/dani/Desktop/cutechess/cutechess/build/cutechess-cli}"
RUN_DIR="${RUN_DIR:-nnue/optionc_runs/$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$RUN_DIR"
# Mirror everything to a master log.
exec > >(tee -a "$RUN_DIR/optionc.log") 2>&1

echo "================ Option C: $RUN_DIR ================"
date

# ----- Phase 1.1: distillation data ---------------------------------------- #
echo "=== [1/4] distillation data: $DISTILL_POSITIONS positions @ depth $DISTILL_DEPTH (quiet-filtered) ==="
uv run python nnue/gen_data.py \
    --count "$DISTILL_POSITIONS" --depth "$DISTILL_DEPTH" \
    --quiet-filter --max-spread "$MAX_SPREAD" \
    --out "$RUN_DIR/distill.txt"

# ----- Phase 1.2: train net0 (cp / distillation) --------------------------- #
echo "=== [2/4] training net0: $NET0_EPOCHS epochs ==="
uv run --group train python nnue/train.py \
    --data "$RUN_DIR/distill.txt" --label cp \
    --epochs "$NET0_EPOCHS" --batch "$NET0_BATCH" --out "$RUN_DIR/net0.pt"

# ----- Phase 1.3: export --------------------------------------------------- #
echo "=== [3/4] exporting net0.bin ==="
uv run --group train python nnue/export.py --in "$RUN_DIR/net0.pt" --out "$RUN_DIR/net0.bin"
net0="$PROJECT_DIR/$RUN_DIR/net0.bin"

# ----- Phase 1.4: A/B net0 vs hand-crafted --------------------------------- #
echo "=== [4/4] A/B: net0 vs hand-crafted ($AB_GAMES games @ $AB_TC) ==="
"$CUTECHESS_CLI" \
    -engine name=net0 cmd=scripts/chessbote-uci.sh proto=uci "option.EvalFile=$net0" \
    -engine name=handcrafted cmd=scripts/chessbote-uci.sh proto=uci option.EvalFile=handcrafted \
    -each tc="$AB_TC" -games "$AB_GAMES" -repeat -recover -concurrency "$JOBS" \
    -resign movecount=3 score=700 -draw movenumber=50 movecount=8 score=10 \
    -pgnout "$RUN_DIR/net0_ab.pgn" > "$RUN_DIR/net0_ab.log" 2>&1
ab="$(grep '^Score of net0 vs handcrafted' "$RUN_DIR/net0_ab.log" | tail -1 |
    sed -E 's/.*\[([0-9.]+)\].*/\1/')"
ab="${ab:-0}"

if awk "BEGIN{exit !($ab >= $AB_THRESH)}"; then
    start_champion="$net0"
    echo "net0 scored $ab vs hand-crafted -> starting the loop FROM net0"
else
    start_champion="handcrafted"
    echo "net0 scored $ab vs hand-crafted (< $AB_THRESH) -> loop starts from hand-crafted;"
    echo "  it will only promote generations that beat the hand-crafted eval."
fi

# ----- Phase 2: gated self-play iteration ---------------------------------- #
echo "=== Phase 2: self-play iteration ($GENERATIONS generations) ==="
START_CHAMPION="$start_champion" \
GENERATIONS="$GENERATIONS" POSITIONS="$POSITIONS" SELFPLAY_DEPTH="$SELFPLAY_DEPTH" \
EPOCHS="$EPOCHS" GATE_GAMES="$GATE_GAMES" GATE_TC="$GATE_TC" GATE_THRESH="$GATE_THRESH" \
DATA_WINDOW="$DATA_WINDOW" JOBS="$JOBS" CUTECHESS_CLI="$CUTECHESS_CLI" \
RUN_DIR="$RUN_DIR/loop" \
    scripts/nnue_loop.sh

echo "================ Option C complete ================"
date
echo "net0:            $net0  (vs hand-crafted: $ab)"
if [ -f "$RUN_DIR/loop/champion.bin" ]; then
    echo "final champion:  $RUN_DIR/loop/champion.bin"
else
    echo "final champion:  none promoted beyond the starting point ($start_champion)"
fi
echo "to play with it: CHESSBOTE_NNUE=<champion.bin> scripts/chessbote-uci.sh"
