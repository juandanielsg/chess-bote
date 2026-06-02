"""Core type definitions and constants.

Square indexing is Little-Endian Rank-File (LERF):

    square = rank * 8 + file

so a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
The least significant bit of a bitboard corresponds to a1.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #


class Color(IntEnum):
    WHITE = 0
    BLACK = 1

    def opposite(self) -> "Color":
        return Color(self ^ 1)


# --------------------------------------------------------------------------- #
# Piece types
# --------------------------------------------------------------------------- #


class PieceType(IntEnum):
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5
    NONE = 6  # sentinel for an empty mailbox square


# Single-character symbols, indexed by PieceType. Uppercase = white when
# combined with color elsewhere; used for FEN and debug output.
PIECE_SYMBOL: Final = ["p", "n", "b", "r", "q", "k"]


def piece_symbol(color: Color, pt: PieceType) -> str:
    """Return the FEN character for a colored piece (e.g. white knight -> 'N')."""
    s = PIECE_SYMBOL[pt]
    return s.upper() if color == Color.WHITE else s


# --------------------------------------------------------------------------- #
# Squares
# --------------------------------------------------------------------------- #

# Generate A1..H8 as module-level constants (LERF order).
# A1 = 0, B1 = 1, ..., H8 = 63.
(
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
) = range(64)

NO_SQUARE: Final = -1

NUM_SQUARES: Final = 64
NUM_FILES: Final = 8
NUM_RANKS: Final = 8


def square(file: int, rank: int) -> int:
    """Build a square index from file (0=a..7=h) and rank (0=rank1..7=rank8)."""
    return rank * 8 + file


def file_of(sq: int) -> int:
    """File index of a square (0 = a-file ... 7 = h-file)."""
    return sq & 7


def rank_of(sq: int) -> int:
    """Rank index of a square (0 = rank 1 ... 7 = rank 8)."""
    return sq >> 3


def square_name(sq: int) -> str:
    """Algebraic name of a square, e.g. 0 -> 'a1', 63 -> 'h8'."""
    return f"{'abcdefgh'[file_of(sq)]}{rank_of(sq) + 1}"


def parse_square(name: str) -> int:
    """Parse an algebraic square name (e.g. 'e4') into a square index."""
    file = ord(name[0]) - ord("a")
    rank = int(name[1]) - 1
    return square(file, rank)
