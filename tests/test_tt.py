"""Transposition table tests."""

import pytest

from chess_bote.board import Board
from chess_bote.search import Searcher
from chess_bote.tt import EXACT, LOWER, TranspositionTable


def test_store_and_probe_roundtrip():
    tt = TranspositionTable(1 << 8)
    tt.store(0xABCDEF, 5, 123, EXACT, 4242)
    entry = tt.probe(0xABCDEF)
    assert entry is not None
    key, depth, score, flag, move = entry
    assert (depth, score, flag, move) == (5, 123, EXACT, 4242)


def test_probe_miss_on_different_key():
    tt = TranspositionTable(1 << 8)
    tt.store(0x100, 3, 10, EXACT, 1)
    # Same slot (size 256) but different key -> treated as a miss (collision).
    assert tt.probe(0x100 + (1 << 8)) is None


def test_depth_preferred_replacement():
    tt = TranspositionTable(1 << 8)
    tt.store(0x55, 6, 100, EXACT, 1)
    tt.store(0x55, 2, 200, LOWER, 2)  # shallower -> ignored
    assert tt.probe(0x55)[1] == 6
    tt.store(0x55, 8, 300, EXACT, 3)  # deeper -> replaces
    assert tt.probe(0x55)[1:3] == (8, 300)


def test_clear():
    tt = TranspositionTable(1 << 8)
    tt.store(0x1, 1, 1, EXACT, 1)
    tt.clear()
    assert tt.probe(0x1) is None
    assert len(tt) == 0


def test_non_power_of_two_rejected():
    with pytest.raises(ValueError):
        TranspositionTable(1000)


def test_tt_does_not_change_result():
    """TT must be a pure optimization: same move and score, fewer nodes."""
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

    on = Searcher()
    r_on = on.search(Board(fen), max_depth=3)

    off = Searcher()
    off.tt.probe = lambda key: None  # type: ignore[assignment]
    off.tt.store = lambda *a, **k: None  # type: ignore[assignment]
    r_off = off.search(Board(fen), max_depth=3)

    assert r_on.best_move == r_off.best_move
    assert r_on.score == r_off.score
    assert r_on.nodes < r_off.nodes
