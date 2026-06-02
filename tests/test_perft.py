"""Perft regression tests against known node counts.

Deeper, slower cases are marked ``slow`` and skipped by default. Run them with::

    uv run pytest -m slow
"""

import pytest

from chess_bote.board import Board
from chess_bote.perft import perft

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

CASES = [
    (STARTPOS, 1, 20),
    (STARTPOS, 2, 400),
    (STARTPOS, 3, 8902),
    (KIWIPETE, 1, 48),
    (KIWIPETE, 2, 2039),
    (KIWIPETE, 3, 97862),
    (POSITION3, 1, 14),
    (POSITION3, 2, 191),
    (POSITION3, 3, 2812),
    (POSITION3, 4, 43238),
    (POSITION4, 1, 6),
    (POSITION4, 2, 264),
    (POSITION4, 3, 9467),
    (POSITION5, 1, 44),
    (POSITION5, 2, 1486),
    (POSITION5, 3, 62379),
    (POSITION6, 1, 46),
    (POSITION6, 2, 2079),
    (POSITION6, 3, 89890),
]

SLOW_CASES = [
    (STARTPOS, 4, 197281),
    (STARTPOS, 5, 4865609),
    (KIWIPETE, 4, 4085603),
    (POSITION3, 5, 674624),
    (POSITION4, 4, 422333),
    (POSITION5, 4, 2103487),
]


@pytest.mark.parametrize("fen,depth,expected", CASES)
def test_perft(fen, depth, expected):
    assert perft(Board(fen), depth) == expected


@pytest.mark.slow
@pytest.mark.parametrize("fen,depth,expected", SLOW_CASES)
def test_perft_slow(fen, depth, expected):
    assert perft(Board(fen), depth) == expected
