"""Minimal UCI front-end for IntuBote (instant moves -- there is no search).

    python intubote/uci.py --ckpt intubote/intubote.pt

Supported: uci, isready, ucinewgame, position [startpos|fen ...] [moves ...],
go (any limits are accepted and ignored), setoption name MateInOne value
true|false, quit.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding import win_to_cp  # noqa: E402
from engine import Engine  # noqa: E402

from chess_bote.board import Board  # noqa: E402
from chess_bote.movegen import parse_move  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def send(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def set_position(tokens: list[str]) -> Board:
    i = 1
    if i < len(tokens) and tokens[i] == "startpos":
        board, i = Board(START_FEN), i + 1
    elif i < len(tokens) and tokens[i] == "fen":
        board, i = Board(" ".join(tokens[i + 1:i + 7])), i + 7
    else:
        return Board(START_FEN)
    if i < len(tokens) and tokens[i] == "moves":
        for uci in tokens[i + 1:]:
            move = parse_move(board, uci)
            if move is None:
                send(f"info string illegal move ignored: {uci}")
                break
            board.make_move(move)
    return board


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "intubote.pt"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    engine = Engine(args.ckpt, args.device)
    board = Board(START_FEN)
    for raw in sys.stdin:
        tokens = raw.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            send("id name IntuBote")
            send("id author juandanielsg")
            send("option name MateInOne type check default true")
            send("uciok")
        elif cmd == "isready":
            send("readyok")
        elif cmd == "ucinewgame":
            board = Board(START_FEN)
        elif cmd == "position":
            board = set_position(tokens)
        elif cmd == "setoption" and "MateInOne" in tokens:
            engine.mate_in_one = tokens[-1].lower() == "true"
        elif cmd == "go":
            move, win = engine.best_move(board)
            send(f"info depth 0 score cp {round(win_to_cp(win))}")
            send(f"bestmove {move or '0000'}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
