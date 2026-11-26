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


# --------------------------------------------------------------------------- #
# Puzzles: data prep, benchmark, fine-tuning losses
# --------------------------------------------------------------------------- #

from prepare_puzzles import (  # noqa: E402
    in_bench_pool, puzzle_positions, uci_to_move,
)

_SPECIAL_FENS = [
    "r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # castling, promos
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",           # en passant
    "8/2P3k1/8/8/8/8/5Kp1/7R b - - 0 1",                                       # promo + capture
]


def test_uci_to_move_matches_move_generator():
    boards = list(_random_positions(n_games=20)) + [Board(f) for f in _SPECIAL_FENS]
    checked = 0
    for b in boards:
        for m in generate_legal(b):
            assert uci_to_move(b, m.uci()).value == m.value, (b.fen(), m.uci())
            checked += 1
    assert checked > 3000


def test_puzzle_positions_are_the_solver_moves():
    # Lichess puzzle 0000D: White plays d3d6, Black (the solver) answers.
    fen = "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27"
    out = list(puzzle_positions(fen, "d3d6 f8d8 d6d8 f6d8".split()))
    assert [(black, uci) for _, black, uci, _ in out] == [(True, "f8d8"), (True, "f6d8")]
    after = Board(fen)
    for u in ("d3d6",):
        after.make_move(uci_to_move(after, u))
    assert (out[0][0] == encode_fen(after.fen())).all()


def test_final_move_accepts_any_mate():
    # Black plays g8h8; then both Ra8# and Re8# mate, and Lichess accepts either.
    fen = "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 b - - 0 1"
    ((_, black, uci, (legal, accepted)),) = puzzle_positions(fen, ["g8h8", "a1a8"], bench=True)
    assert not black and uci == "a1a8"
    assert accepted == ["a1a8", "e1e8"]
    assert set(accepted) <= set(legal)


def test_prepare_puzzles_and_benchmark(tmp_path):
    torch = pytest.importorskip("torch")
    import subprocess

    from puzzle_bench import Bench, solve

    pool = [f"t{i}" for i in range(5000) if in_bench_pool(f"t{i}")][:2]
    train = [f"t{i}" for i in range(5000) if not in_bench_pool(f"t{i}")][:3]
    mate = "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 b - - 0 1"
    short = "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27"
    header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
    rows = [f"{pool[0]},{mate},g8h8 a1a8,900,80,95,500,mate mateIn1,u,\n",
            f"{pool[1]},{short},d3d6 f8d8 d6d8 f6d8,1500,80,95,500,advantage short,u,\n"]
    rows += [f"{pid},{short},d3d6 f8d8 d6d8 f6d8,1500,80,95,500,short,u,\n" for pid in train]
    csv_path = tmp_path / "p.csv"
    csv_path.write_text(header + "".join(rows))
    out = tmp_path / "pz"
    root = os.path.join(os.path.dirname(__file__), "..")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(root, "src"),
                                                       os.path.join(root, "intubote")]))
    subprocess.run([sys.executable, os.path.join(root, "intubote", "prepare_puzzles.py"),
                    "--source", str(csv_path), "--count", "10", "--out", str(out)],
                   check=True, env=env)

    assert np.load(out / "tokens.npy").shape == (6, SEQ_LEN)  # 3 puzzles x 2 solver moves
    assert np.isnan(np.load(out / "wins.npy")).all()
    bench = Bench(str(out / "bench.npz"))
    assert bench.n_puzzles == 2 and len(bench.tokens) == 3

    class Picks(torch.nn.Module):
        """Stand-in model that always prefers the given vocab indices, in order."""

        def __init__(self, prefs):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.prefs = prefs

        def forward(self, tok):
            logits = torch.zeros(len(tok), NUM_MOVES)
            for rank, idx in enumerate(self.prefs):
                logits[:, idx] = 100.0 - rank
            return logits, torch.zeros(len(tok))

    sol = [encode_move(u, b) for u, b in (("a1a8", False), ("f8d8", True), ("f6d8", True))]
    correct, solved = solve(Picks(sol), bench)
    assert correct.all() and solved.all()
    # The other mate is accepted too; missing the second move fails puzzle 2 only.
    alt = [encode_move("e1e8", False), sol[1]]
    correct, solved = solve(Picks(alt), bench)
    assert solved.tolist() == [True, False]


def test_losses_skip_missing_win_probabilities():
    torch = pytest.importorskip("torch")
    from train import policy_loss, value_loss

    logits = torch.randn(4, NUM_MOVES)
    moves = torch.tensor([[5, 7], [3, -1], [9, -1], [11, -1]])
    wins = torch.tensor([[0.6, 0.4], [0.5, float("nan")],
                         [float("nan"), float("nan")], [float("nan"), float("nan")]])
    for target in ("best", "soft"):
        assert torch.isfinite(policy_loss(logits, moves, wins, target, 0.03, 0.0))
    v = value_loss(torch.zeros(4), wins)
    # Only the 2 rows with a win probability count; at logit 0 each BCE is ln 2.
    assert torch.isclose(v, torch.tensor(0.6931), atol=1e-3)
