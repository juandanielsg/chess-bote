"""Bitboard primitives.

A bitboard is a 64-bit integer where bit ``i`` (LSB = bit 0) is set when
square ``i`` is occupied, using LERF indexing (see :mod:`chess_bote.types`).

Python ints are arbitrary precision, so any operation that can shift bits past
the 64-bit boundary (notably left shifts) must be masked with :data:`BB_ALL`.
"""

from __future__ import annotations

from typing import Iterator

# --------------------------------------------------------------------------- #
# Fundamental masks
# --------------------------------------------------------------------------- #

BB_EMPTY: int = 0
BB_ALL: int = 0xFFFF_FFFF_FFFF_FFFF


def square_bb(sq: int) -> int:
    """The bitboard with only ``sq`` set."""
    return 1 << sq


# File masks (a..h)
FILE_A: int = 0x0101_0101_0101_0101
FILE_B: int = FILE_A << 1
FILE_C: int = FILE_A << 2
FILE_D: int = FILE_A << 3
FILE_E: int = FILE_A << 4
FILE_F: int = FILE_A << 5
FILE_G: int = FILE_A << 6
FILE_H: int = FILE_A << 7
FILES: list[int] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H]

# Rank masks (1..8)
RANK_1: int = 0x0000_0000_0000_00FF
RANK_2: int = RANK_1 << 8
RANK_3: int = RANK_1 << 16
RANK_4: int = RANK_1 << 24
RANK_5: int = RANK_1 << 32
RANK_6: int = RANK_1 << 40
RANK_7: int = RANK_1 << 48
RANK_8: int = RANK_1 << 56
RANKS: list[int] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8]


# --------------------------------------------------------------------------- #
# Single-bit access
# --------------------------------------------------------------------------- #


def get_bit(bb: int, sq: int) -> bool:
    return bool((bb >> sq) & 1)


def set_bit(bb: int, sq: int) -> int:
    return bb | (1 << sq)


def clear_bit(bb: int, sq: int) -> int:
    return bb & ~(1 << sq)


# --------------------------------------------------------------------------- #
# Population / scanning
# --------------------------------------------------------------------------- #


def popcount(bb: int) -> int:
    """Number of set bits. Uses the native ``int.bit_count`` (Python 3.10+)."""
    return bb.bit_count()


def lsb(bb: int) -> int:
    """Index of the least-significant set bit. ``bb`` must be non-zero."""
    return (bb & -bb).bit_length() - 1


def msb(bb: int) -> int:
    """Index of the most-significant set bit. ``bb`` must be non-zero."""
    return bb.bit_length() - 1


def pop_lsb(bb: int) -> tuple[int, int]:
    """Return ``(square, bb_with_lsb_cleared)`` for the least-significant bit."""
    sq = (bb & -bb).bit_length() - 1
    return sq, bb & (bb - 1)


def scan_forward(bb: int) -> Iterator[int]:
    """Yield set-bit square indices from least- to most-significant."""
    while bb:
        sq = (bb & -bb).bit_length() - 1
        yield sq
        bb &= bb - 1


# --------------------------------------------------------------------------- #
# Directional shifts (masked to stay within 64 bits and avoid file wrap)
# --------------------------------------------------------------------------- #
#
# Directions are named from White's perspective: "north" = toward rank 8.


def north(bb: int) -> int:
    return (bb << 8) & BB_ALL


def south(bb: int) -> int:
    return bb >> 8


def east(bb: int) -> int:
    return (bb << 1) & ~FILE_A & BB_ALL


def west(bb: int) -> int:
    return (bb >> 1) & ~FILE_H


def north_east(bb: int) -> int:
    return (bb << 9) & ~FILE_A & BB_ALL


def north_west(bb: int) -> int:
    return (bb << 7) & ~FILE_H & BB_ALL


def south_east(bb: int) -> int:
    return (bb >> 7) & ~FILE_A


def south_west(bb: int) -> int:
    return (bb >> 9) & ~FILE_H


# --------------------------------------------------------------------------- #
# Debug
# --------------------------------------------------------------------------- #


def to_string(bb: int) -> str:
    """Render a bitboard as 8 rows (rank 8 at top) of '.' and 'x'."""
    rows = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = rank * 8 + file
            row.append("x" if get_bit(bb, sq) else ".")
        rows.append(" ".join(row))
    return "\n".join(rows)
