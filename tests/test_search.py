"""Search and evaluation sanity tests."""

from chess_bote.board import Board
from chess_bote.eval import evaluate
from chess_bote.search import MATE, MATE_THRESHOLD, find_best_move


def test_eval_startpos_symmetric():
    assert evaluate(Board()) == 0


def test_eval_side_relative():
    # White up a clean rook; score is positive for whoever is to move.
    white_to_move = Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    black_to_move = Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1")
    assert evaluate(white_to_move) > 0
    assert evaluate(black_to_move) < 0


def test_grabs_free_queen():
    board = Board("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1")
    result = find_best_move(board, depth=3)
    assert result.best_move.uci() == "d1d4"


def test_finds_back_rank_mate_in_1():
    board = Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    result = find_best_move(board, depth=3)
    assert result.best_move.uci() == "a1a8"
    assert result.score >= MATE_THRESHOLD


def test_finds_mate_in_2():
    # Requires a deeper search, so it exercises PVS + null-move pruning.
    board = Board("5k2/8/5K2/8/8/8/8/5Q2 w - - 0 1")
    result = find_best_move(board, depth=5)
    assert result.score >= MATE_THRESHOLD
    assert (MATE - result.score + 1) // 2 == 2  # mate in 2 moves


def test_detects_stalemate():
    # Black to move, not in check, no legal moves -> score 0, no move.
    board = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    result = find_best_move(board, depth=2)
    assert result.best_move is None
    assert result.score == 0


def test_detects_checkmate():
    # Black to move and checkmated.
    board = Board("6rk/5Npp/8/8/8/8/8/6K1 b - - 0 1")
    result = find_best_move(board, depth=2)
    assert result.best_move is None
    assert result.score <= -MATE_THRESHOLD
