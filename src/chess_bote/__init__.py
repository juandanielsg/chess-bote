"""chess-bote: a chess engine."""

from .board import Board
from .eval import evaluate
from .move import Move
from .movegen import generate_legal, in_check
from .search import SearchResult, Searcher
from .search import find_best_move as find_best_move_python
from .native import HAS_NATIVE, native_search
from .types import Color, PieceType

# The public best-move entry point uses the native search when available.
find_best_move = native_search if HAS_NATIVE else find_best_move_python

__all__ = [
    "Board",
    "Move",
    "Color",
    "PieceType",
    "evaluate",
    "generate_legal",
    "in_check",
    "find_best_move",
    "find_best_move_python",
    "Searcher",
    "SearchResult",
    "HAS_NATIVE",
]


def main() -> None:
    board = Board.startpos()
    print(board)
    print()
    print(board.fen())
