"""Bridge to the compiled ``_core`` extension.

On import this loads the evaluation parameters from :mod:`chess_bote.eval` into
the native core, so the C++ search and the Python search evaluate positions
identically (single source of truth -- tune weights in ``eval.py`` and they
flow into the fast search with no recompile).

``HAS_NATIVE`` is False if the extension was not built, so callers can fall
back to the pure-Python search.
"""

from __future__ import annotations

import os
from typing import Callable

from .board import Board
from .eval import EG_PST, EG_VALUE, MG_PST, MG_VALUE, PHASE_WEIGHT
from .movegen import parse_move
from .search import SearchResult

try:
    from . import _core

    HAS_NATIVE = True
    _core.set_eval_params(MG_VALUE, EG_VALUE, MG_PST, EG_PST, PHASE_WEIGHT)
    # Optionally switch to NNUE eval if a net path is provided via the
    # environment (used by the NNUE launcher / A-B matches).
    _nnue_path = os.environ.get("CHESSBOTE_NNUE")
    if _nnue_path and not _core.load_nnue(_nnue_path):
        import sys

        print(f"info string failed to load NNUE: {_nnue_path}", file=sys.stderr)
except ImportError:  # pragma: no cover - extension always built in this project
    _core = None  # type: ignore[assignment]
    HAS_NATIVE = False


def native_search(
    board: Board,
    depth: int = 64,
    movetime: float | None = None,
    on_info: Callable[[dict], None] | None = None,
    nodes: int = 0,
) -> SearchResult:
    """Run the native search and adapt the result to :class:`SearchResult`.

    ``nodes`` (0 = unlimited) caps the searched node count.
    ``on_info``, if given, is called once per completed depth with a dict
    ``{depth, score, nodes, time, pv}`` (``pv`` is a list of UCI move strings).
    """
    movetime_ms = int(movetime * 1000) if movetime else 0
    r = _core.search(board.fen(), depth, movetime_ms, nodes, on_info)
    uci = r["bestmove"]
    move = None if uci == "0000" else parse_move(board, uci)
    return SearchResult(
        best_move=move,
        score=r["score"],
        depth=r["depth"],
        nodes=r["nodes"],
        seconds=r["time"],
        pv=[move] if move else [],
    )


def load_nnue(path: str) -> bool:
    """Load a quantized NNUE and switch eval to it. Returns success."""
    return bool(HAS_NATIVE and _core.load_nnue(path))


def set_nnue(on: bool) -> None:
    """Toggle between NNUE (True) and hand-crafted (False) evaluation."""
    if HAS_NATIVE:
        _core.use_nnue(on)


def selfplay_game(
    game_id: int,
    depth: int = 64,
    nodes: int = 0,
    random_plies: int = 10,
    sample_per_game: int = 20,
    max_plies: int = 300,
) -> list[tuple[str, float, int]]:
    """Play one native self-play game; see ``_core.selfplay_game``.

    Returns sampled quiet positions as ``(fen, wdl, cp)`` tuples, where both
    the WDL game outcome and the search-score cp label are White-relative.
    """
    return _core.selfplay_game(game_id, depth, nodes, random_plies,
                               sample_per_game, max_plies)


def stop() -> None:
    """Ask the running native search to stop (thread-safe)."""
    if HAS_NATIVE:
        _core.stop()


def new_game() -> None:
    """Clear the native transposition table."""
    if HAS_NATIVE:
        _core.new_game()
