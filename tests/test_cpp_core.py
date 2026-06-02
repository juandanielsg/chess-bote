"""The native core must agree with the pure-Python perft oracle exactly."""

import pytest

from chess_bote.board import Board
from chess_bote.perft import perft as py_perft, perft_divide as py_divide

_core = pytest.importorskip("chess_bote._core")

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"

KNOWN = [
    (STARTPOS, 1, 20), (STARTPOS, 4, 197281),
    (KIWIPETE, 3, 97862), (POSITION3, 4, 43238),
    (POSITION4, 3, 9467), (POSITION5, 3, 62379),
]


@pytest.mark.parametrize("fen,depth,expected", KNOWN)
def test_cpp_perft_known_counts(fen, depth, expected):
    assert _core.perft(fen, depth) == expected


@pytest.mark.parametrize("fen", [STARTPOS, KIWIPETE, POSITION3, POSITION4, POSITION5])
def test_cpp_matches_python_oracle(fen):
    for depth in (1, 2, 3):
        assert _core.perft(fen, depth) == py_perft(Board(fen), depth)


def test_cpp_perft_divide_matches_python():
    # Per-move breakdown must agree, which checks move encoding too.
    assert _core.perft_divide(KIWIPETE, 3) == py_divide(Board(KIWIPETE), 3)


# Importing the package loads eval params into the native core (see native.py).
import chess_bote  # noqa: E402
from chess_bote import eval as pyeval  # noqa: E402
from chess_bote.search import MATE_THRESHOLD  # noqa: E402


@pytest.mark.parametrize("fen", [STARTPOS, KIWIPETE, POSITION3, POSITION4, POSITION5])
def test_native_eval_matches_python(fen):
    assert _core.evaluate(fen) == pyeval.evaluate(Board(fen))


def test_native_search_finds_free_queen():
    r = _core.search("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", 4, 0)
    assert r["bestmove"] == "d1d4"


def test_native_search_finds_mate_in_1():
    r = _core.search("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 4, 0)
    assert r["bestmove"] == "a1a8"
    assert r["score"] >= MATE_THRESHOLD


def test_package_find_best_move_is_native_and_works():
    assert chess_bote.HAS_NATIVE
    r = chess_bote.find_best_move(Board("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"), depth=4)
    assert r.best_move.uci() == "d1d4"


def test_native_info_callback_reports_each_depth_with_pv():
    _core.new_game()
    lines = []
    _core.search(STARTPOS, 5, 0, lines.append)
    # One info per completed depth, in increasing order.
    assert [d["depth"] for d in lines] == [1, 2, 3, 4, 5]
    last = lines[-1]
    assert {"depth", "score", "nodes", "time", "pv"} <= set(last)
    assert last["pv"] and last["pv"][0] == "e2e4" or len(last["pv"]) >= 1
    assert all(isinstance(m, str) for m in last["pv"])  # PV is UCI strings
