"""Move representation.

A move is packed into a 16-bit integer (the classic encoding):

    bits  0- 5 : destination square
    bits  6-11 : origin square
    bits 12-15 : flags (see below)

The flag nibble distinguishes quiet moves, captures, castles, the double pawn
push, en passant, and the eight promotion variants. Bit 14 of the flags marks
a capture; bit 15 marks a promotion -- this lets us test those cheaply.
"""

from __future__ import annotations

from .types import PieceType, square_name

# Flag codes (the low nibble of the packed move).
FLAG_QUIET = 0b0000
FLAG_DOUBLE_PAWN = 0b0001
FLAG_KING_CASTLE = 0b0010
FLAG_QUEEN_CASTLE = 0b0011
FLAG_CAPTURE = 0b0100
FLAG_EN_PASSANT = 0b0101
FLAG_PROMO_N = 0b1000
FLAG_PROMO_B = 0b1001
FLAG_PROMO_R = 0b1010
FLAG_PROMO_Q = 0b1011
FLAG_PROMO_N_CAP = 0b1100
FLAG_PROMO_B_CAP = 0b1101
FLAG_PROMO_R_CAP = 0b1110
FLAG_PROMO_Q_CAP = 0b1111

_CAPTURE_BIT = 0b0100
_PROMO_BIT = 0b1000

# Promotion flag (low 2 bits) -> resulting piece type.
_PROMO_PIECE = {
    0b00: PieceType.KNIGHT,
    0b01: PieceType.BISHOP,
    0b10: PieceType.ROOK,
    0b11: PieceType.QUEEN,
}


class Move:
    """A 16-bit packed chess move.

    Instances are lightweight wrappers around an ``int`` so they can be stored
    cheaply in move lists. The null move is represented by an all-zero value
    (from == to == 0), distinguishable via :meth:`is_null`.
    """

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def create(cls, frm: int, to: int, flag: int = FLAG_QUIET) -> "Move":
        return cls((flag << 12) | (frm << 6) | to)

    @classmethod
    def null(cls) -> "Move":
        return cls(0)

    # -- accessors -------------------------------------------------------- #

    @property
    def to_sq(self) -> int:
        return self.value & 0x3F

    @property
    def from_sq(self) -> int:
        return (self.value >> 6) & 0x3F

    @property
    def flag(self) -> int:
        return (self.value >> 12) & 0xF

    def is_null(self) -> bool:
        return self.value == 0

    def is_capture(self) -> bool:
        return bool(self.flag & _CAPTURE_BIT)

    def is_promotion(self) -> bool:
        return bool(self.flag & _PROMO_BIT)

    def is_en_passant(self) -> bool:
        return self.flag == FLAG_EN_PASSANT

    def is_castle(self) -> bool:
        return self.flag in (FLAG_KING_CASTLE, FLAG_QUEEN_CASTLE)

    def is_double_pawn_push(self) -> bool:
        return self.flag == FLAG_DOUBLE_PAWN

    def promotion_piece(self) -> PieceType | None:
        """The piece type a pawn promotes to, or ``None`` for non-promotions."""
        if not self.is_promotion():
            return None
        return _PROMO_PIECE[self.flag & 0b11]

    # -- dunder ----------------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Move) and other.value == self.value

    def __hash__(self) -> int:
        return self.value

    def uci(self) -> str:
        """Long algebraic / UCI notation, e.g. 'e2e4', 'e7e8q', '0000' (null)."""
        if self.is_null():
            return "0000"
        s = square_name(self.from_sq) + square_name(self.to_sq)
        promo = self.promotion_piece()
        if promo is not None:
            s += "pnbrqk"[promo]
        return s

    def __repr__(self) -> str:
        return f"Move({self.uci()})"
