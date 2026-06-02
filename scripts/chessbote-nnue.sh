#!/usr/bin/env bash
#
# Launch ChessBote using the NNUE evaluation (instead of the hand-crafted one).
# Same as chessbote-uci.sh but sets CHESSBOTE_NNUE so the engine loads the net.
# Used for A/B matches (NNUE vs hand-crafted) and for GUIs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

unset VIRTUAL_ENV
export CHESSBOTE_NNUE="${CHESSBOTE_NNUE:-$PROJECT_DIR/nnue/net.bin}"

exec uv run --quiet --project "$PROJECT_DIR" chess-bote-uci
