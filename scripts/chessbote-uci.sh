#!/usr/bin/env bash
#
# Launcher for the ChessBote UCI engine, for use in chess GUIs
# (CuteChess, Arena, BanksiaGUI) and lichess-bot.
#
# In your GUI's "add engine" dialog, browse to THIS file. It launches the
# engine via uv from the project directory, so it works no matter what working
# directory the GUI runs it from, and keeps stdout reserved for the UCI
# protocol.
#
# Tip: run `uv sync` once beforehand so the first launch doesn't block the GUI
# while the native extension compiles.
set -euo pipefail

# The project root is the directory that contains this script's parent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# uv ignores an inherited VIRTUAL_ENV and warns about it on stderr; clear it so
# the GUI's engine log stays clean.
unset VIRTUAL_ENV

# --quiet keeps uv's own output off the streams; --project lets the GUI invoke
# us from anywhere. exec replaces this shell so signals reach the engine.
exec uv run --quiet --project "$PROJECT_DIR" chess-bote-uci
