"""IntuBote encoding / data / model tests (model tests skip without torch)."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intubote"))

from encoding import (  # noqa: E402
    MOVE_VOCAB, NUM_MOVES, SEQ_LEN, TOKEN_VOCAB, decode_move, encode_fen, encode_move, flip_move,
)
from prepare_data import parse_record  # noqa: E402

from chess_bote.board import Board  # noqa: E402
from chess_bote.movegen import generate_legal  # noqa: E402


def test_vocab_size_and_uniqueness():
    assert NUM_MOVES == 1880
    assert len(set(MOVE_VOCAB)) == NUM_MOVES


def _random_positions(n_games=60, plies=80, seed=1):
    rng = random.Random(seed)
    for _ in range(n_games):
        b = Board()
        for _ in range(plies):
            moves = generate_legal(b)
            if not moves:
                break
            yield b
            b.make_move(rng.choice(moves))


def test_every_legal_move_is_encodable_and_round_trips():
    seen = 0
    for b in _random_positions():
        black = b.fen().split()[1] == "b"
        for m in generate_legal(b):
            u = m.uci()
            idx = encode_move(u, black)
            assert idx >= 0, (b.fen(), u)
            assert decode_move(idx, black) == u
            seen += 1
    assert seen > 10_000


def test_promotions_and_castling_in_vocab():
    for u in ("e7e8q", "a7b8n", "h7h8r", "e1g1", "e1c1"):
        assert u in MOVE_VOCAB
    assert encode_move("e2e1q", black=True) == MOVE_VOCAB.index("e7e8q")
    assert flip_move("e7e5") == "e2e4"


def _mirror_fen(fen: str) -> str:
    """Same position with colors swapped and the board flipped vertically."""
    placement, side, castling, ep = fen.split()[:4]
    rows = [r.swapcase() for r in placement.split("/")][::-1]
    castling = "".join(sorted(castling.swapcase(), key="KQkq".index)) if castling != "-" else "-"
    ep = "-" if ep == "-" else ep[0] + str(9 - int(ep[1]))
    return f"{'/'.join(rows)} {'b' if side == 'w' else 'w'} {castling} {ep} 0 1"


def test_tokens_are_side_to_move_invariant():
    for b in _random_positions(n_games=20, plies=40, seed=7):
        fen = b.fen()
        t1, t2 = encode_fen(fen), encode_fen(_mirror_fen(fen))
        assert t1.shape == (SEQ_LEN,) and t1.max() < TOKEN_VOCAB
        assert np.array_equal(t1, t2), fen


def test_parse_lichess_record():
    rec = {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -",
           "evals": [
               {"pvs": [{"cp": 40, "line": "c7c5 g1f3"}], "knodes": 10, "depth": 20},
               {"pvs": [{"cp": 25, "line": "e7e5 g1f3"}, {"cp": 60, "line": "d7d6"},
                        {"mate": 3, "line": "g8f6"}], "knodes": 99, "depth": 30},
           ]}
    fen, moves, wins = parse_record(rec, top_k=5, min_depth=0)
    assert [decode_move(m, True) for m in moves] == ["e7e5", "d7d6", "g8f6"]
    # cp is White-relative: +25 for White means Black (to move) is slightly worse.
    assert wins[0] < 0.5 and wins[1] < wins[0] and wins[2] == 0.0
    assert parse_record(rec, top_k=5, min_depth=31) is None


def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    from model import Config, IntuBote

    model = IntuBote(Config(d_model=32, layers=2, heads=4))
    tok = torch.from_numpy(np.stack([encode_fen(Board().fen())] * 3))
    logits, value = model(tok)
    assert logits.shape == (3, NUM_MOVES) and value.shape == (3,)
