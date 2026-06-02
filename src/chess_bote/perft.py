"""Perft: count leaf nodes of the move tree to a fixed depth.

Perft is the gold-standard correctness test for board + movegen. The node
counts for standard positions are known exactly, so any bug in move
generation, make/unmake, or legality shows up as a mismatch.
"""

from __future__ import annotations

from .board import Board
from .move import Move
from .movegen import generate_legal


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    moves = generate_legal(board)
    if depth == 1:
        return len(moves)
    nodes = 0
    for move in moves:
        board.make_move(move)
        nodes += perft(board, depth - 1)
        board.unmake_move(move)
    return nodes


def perft_divide(board: Board, depth: int) -> dict[str, int]:
    """Per-root-move node counts -- the standard tool for locating a discrepancy."""
    result: dict[str, int] = {}
    for move in generate_legal(board):
        board.make_move(move)
        result[move.uci()] = perft(board, depth - 1) if depth > 1 else 1
        board.unmake_move(move)
    return result
