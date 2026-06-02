"""UCI protocol unit tests (fast, deterministic -- no subprocess/threads)."""

from chess_bote.search import MATE
from chess_bote.uci import UCIEngine, _format_score


def test_format_score_cp():
    assert _format_score(35) == "cp 35"
    assert _format_score(-120) == "cp -120"


def test_format_score_mate():
    # Mate in 1 is delivered on our move: score = MATE - 1 (one ply).
    assert _format_score(MATE - 1) == "mate 1"
    assert _format_score(MATE - 3) == "mate 2"  # 3 plies -> 2 moves
    # Getting mated.
    assert _format_score(-(MATE - 2)) == "mate -1"


def test_position_startpos_with_moves():
    eng = UCIEngine()
    eng.dispatch("position startpos moves e2e4 e7e5 g1f3")
    assert eng.board.fen().startswith(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b"
    )


def test_position_fen():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    eng = UCIEngine()
    eng.dispatch(f"position fen {fen}")
    assert eng.board.fen() == fen


def test_position_illegal_move_stops_applying():
    eng = UCIEngine()
    # e2e5 is illegal; the engine should ignore it and stop there.
    eng.dispatch("position startpos moves e2e4 e2e5")
    # Only e2e4 applied -> black to move.
    assert " b " in eng.board.fen()


def test_go_depth_runs_and_emits_bestmove():
    eng = UCIEngine()
    sent: list[str] = []
    eng._send = lambda text: sent.append(text)  # type: ignore[method-assign]
    eng.dispatch("position startpos")
    eng.dispatch("go depth 3")
    eng._stop_search()  # join the worker thread
    assert any(line.startswith("bestmove ") for line in sent)
    assert any(line.startswith("info depth ") for line in sent)
