#!/usr/bin/env bash
#
# Play a match between ChessBote and Stockfish via cutechess-cli, print the
# result, and log it. Handy for tracking ChessBote's strength over time.
#
# All settings are overridable via environment variables, e.g.:
#   GAMES=20 TC=15+0.2 SF_ELO=1500 scripts/match.sh
#
#   GAMES        number of games            (default 10; colors alternate)
#   TC           time control "secs+inc"    (default 10+0.1)
#   SF_ELO       Stockfish strength cap      (default 1320, its minimum)
#   CONCURRENCY  games to run in parallel    (default 1)
#   LOG_DIR      where logs/PGNs are written (default <project>/matches)
#   PGN          PGN path                    (default <LOG_DIR>/<timestamp>.pgn)
#   CUTECHESS_CLI / STOCKFISH                paths, if not in the defaults
#
# Each run writes:
#   <LOG_DIR>/<timestamp>.log   full cutechess-cli output (with settings header)
#   <LOG_DIR>/<timestamp>.pgn   the games (open in the CuteChess GUI to review)
#   <LOG_DIR>/results.log       one summary line per run, appended over time
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CUTECHESS_CLI="${CUTECHESS_CLI:-/home/dani/Desktop/cutechess/cutechess/build/cutechess-cli}"
STOCKFISH="${STOCKFISH:-/usr/games/stockfish}"
GAMES="${GAMES:-10}"
TC="${TC:-10+0.1}"
SF_ELO="${SF_ELO:-1320}"
CONCURRENCY="${CONCURRENCY:-1}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/matches}"

TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
PGN="${PGN:-$LOG_DIR/$TS.pgn}"
RUN_LOG="$LOG_DIR/$TS.log"
RESULTS="$LOG_DIR/results.log"

SETTINGS="GAMES=$GAMES TC=$TC SF_ELO=$SF_ELO CONCURRENCY=$CONCURRENCY"
echo "ChessBote vs Stockfish(Elo $SF_ELO) - $GAMES games at $TC" >&2

# Run the match, echoing a settings header and teeing everything to the run log.
# pipefail is relaxed here so a non-zero engine exit still lets us log the score.
set +o pipefail
{
    echo "# $TS  $SETTINGS"
    "$CUTECHESS_CLI" \
      -engine name=chess-bote cmd="$SCRIPT_DIR/chessbote-uci.sh" proto=uci \
      -engine name=stockfish cmd="$STOCKFISH" proto=uci \
          option.UCI_LimitStrength=true "option.UCI_Elo=$SF_ELO" \
      -each tc="$TC" \
      -games "$GAMES" -repeat -recover -concurrency "$CONCURRENCY" \
      -resign movecount=3 score=700 \
      -draw movenumber=50 movecount=8 score=10 \
      -pgnout "$PGN" 2>&1
} | tee "$RUN_LOG"
set -o pipefail

# Append a one-line summary (settings + final score + Elo) to the results log.
score="$(grep '^Score of' "$RUN_LOG" | tail -1 || true)"
elo="$(grep '^Elo difference' "$RUN_LOG" | tail -1 || true)"
printf '%s | %s | %s | %s\n' "$TS" "$SETTINGS" "${score:-no score}" "${elo:-}" \
    >> "$RESULTS"

echo "" >&2
echo "logged to $RUN_LOG  (games: $PGN)" >&2
echo "history: $RESULTS" >&2
