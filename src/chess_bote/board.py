"""The board: a hybrid bitboard + mailbox representation.

Representation (6 + 2 bitboards):
    * ``piece_bb[pt]``  -- one bitboard per piece type, both colors combined.
    * ``color_bb[c]``   -- occupancy bitboard per color.
    * ``occupied``      -- all pieces (cached union of the color boards).
    * ``mailbox[sq]``   -- piece type on each square (``PieceType.NONE`` if empty);
                           the color is read from the color bitboards.

Incremental state carried with the position: side to move, castling rights,
en-passant target square, the halfmove clock, the fullmove number, and the
Zobrist hash. Moves are applied with :meth:`make_move` / :meth:`unmake_move`,
which push/pop an undo record so the change is reversible without copying.
"""

from __future__ import annotations

from . import zobrist
from .bitboard import scan_forward
from .move import (
    FLAG_EN_PASSANT,
    FLAG_KING_CASTLE,
    FLAG_QUEEN_CASTLE,
    Move,
)
from .types import (
    A1, A8, C1, C8, D1, D8, E1, E8, F1, F8, G1, G8, H1, H8,
    Color,
    NO_SQUARE,
    PieceType,
    file_of,
    parse_square,
    piece_symbol,
    rank_of,
    square_name,
)

# Castling-rights bit flags.
WHITE_OO = 1   # White kingside  (O-O)
WHITE_OOO = 2  # White queenside (O-O-O)
BLACK_OO = 4
BLACK_OOO = 8

# Symbol -> piece type for FEN parsing.
_SYMBOL_TO_PT = {s: PieceType(i) for i, s in enumerate("pnbrqk")}

# Per-square castling mask: ANDed into the rights whenever a square is the
# origin or destination of a move. Moving a king/rook off its home square, or
# capturing a rook on its home square, clears the matching right.
_CASTLE_MASK = [0b1111] * 64
_CASTLE_MASK[E1] = 0b1111 & ~(WHITE_OO | WHITE_OOO)
_CASTLE_MASK[H1] = 0b1111 & ~WHITE_OO
_CASTLE_MASK[A1] = 0b1111 & ~WHITE_OOO
_CASTLE_MASK[E8] = 0b1111 & ~(BLACK_OO | BLACK_OOO)
_CASTLE_MASK[H8] = 0b1111 & ~BLACK_OO
_CASTLE_MASK[A8] = 0b1111 & ~BLACK_OOO

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class _Undo:
    """Per-move snapshot of the irreversible state, for :meth:`unmake_move`."""

    __slots__ = ("captured", "castling", "ep_square", "halfmove", "key")

    def __init__(
        self, captured: int, castling: int, ep_square: int, halfmove: int, key: int
    ) -> None:
        self.captured = captured
        self.castling = castling
        self.ep_square = ep_square
        self.halfmove = halfmove
        self.key = key


class Board:
    __slots__ = (
        "piece_bb",
        "color_bb",
        "occupied",
        "mailbox",
        "side_to_move",
        "castling_rights",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
        "zobrist_key",
        "_history",
    )

    def __init__(self, fen: str = START_FEN) -> None:
        self.set_fen(fen)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _clear(self) -> None:
        self.piece_bb = [0, 0, 0, 0, 0, 0]
        self.color_bb = [0, 0]
        self.occupied = 0
        self.mailbox = [PieceType.NONE] * 64
        self.side_to_move = Color.WHITE
        self.castling_rights = 0
        self.ep_square = NO_SQUARE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.zobrist_key = 0
        self._history: list[_Undo] = []

    def set_fen(self, fen: str) -> None:
        self._clear()
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid FEN (need at least 4 fields): {fen!r}")
        placement, active, castling, ep = parts[:4]
        halfmove = parts[4] if len(parts) > 4 else "0"
        fullmove = parts[5] if len(parts) > 5 else "1"

        rank = 7
        file = 0
        for ch in placement:
            if ch == "/":
                rank -= 1
                file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                color = Color.WHITE if ch.isupper() else Color.BLACK
                pt = _SYMBOL_TO_PT[ch.lower()]
                self._put_piece(color, pt, rank * 8 + file)
                file += 1

        self.side_to_move = Color.WHITE if active == "w" else Color.BLACK

        rights = 0
        if "K" in castling:
            rights |= WHITE_OO
        if "Q" in castling:
            rights |= WHITE_OOO
        if "k" in castling:
            rights |= BLACK_OO
        if "q" in castling:
            rights |= BLACK_OOO
        self.castling_rights = rights

        self.ep_square = NO_SQUARE if ep == "-" else parse_square(ep)
        self.halfmove_clock = int(halfmove)
        self.fullmove_number = int(fullmove)

        self._recompute_hash()

    @classmethod
    def startpos(cls) -> "Board":
        return cls(START_FEN)

    # ------------------------------------------------------------------ #
    # Piece manipulation (keep bitboards, mailbox, and hash in lockstep)
    # ------------------------------------------------------------------ #

    def _put_piece(self, color: int, pt: int, sq: int) -> None:
        bb = 1 << sq
        self.piece_bb[pt] |= bb
        self.color_bb[color] |= bb
        self.occupied |= bb
        self.mailbox[sq] = pt
        self.zobrist_key ^= zobrist.PIECE_KEYS[color][pt][sq]

    def _remove_piece(self, color: int, pt: int, sq: int) -> None:
        mask = ~(1 << sq)
        self.piece_bb[pt] &= mask
        self.color_bb[color] &= mask
        self.occupied &= mask
        self.mailbox[sq] = PieceType.NONE
        self.zobrist_key ^= zobrist.PIECE_KEYS[color][pt][sq]

    def _move_piece(self, color: int, pt: int, frm: int, to: int) -> None:
        self._remove_piece(color, pt, frm)
        self._put_piece(color, pt, to)

    def _recompute_hash(self) -> None:
        key = 0
        for color in (Color.WHITE, Color.BLACK):
            color_mask = self.color_bb[color]
            for pt in range(6):
                for sq in scan_forward(self.piece_bb[pt] & color_mask):
                    key ^= zobrist.PIECE_KEYS[color][pt][sq]
        key ^= zobrist.CASTLING_KEYS[self.castling_rights]
        if self.ep_square != NO_SQUARE:
            key ^= zobrist.EP_FILE_KEYS[file_of(self.ep_square)]
        if self.side_to_move == Color.BLACK:
            key ^= zobrist.SIDE_KEY
        self.zobrist_key = key

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def piece_at(self, sq: int) -> tuple[Color, PieceType] | None:
        """Return ``(color, piece_type)`` on ``sq``, or ``None`` if empty."""
        pt = self.mailbox[sq]
        if pt == PieceType.NONE:
            return None
        color = Color.WHITE if (self.color_bb[Color.WHITE] >> sq) & 1 else Color.BLACK
        return color, PieceType(pt)

    def color_at(self, sq: int) -> Color | None:
        if not (self.occupied >> sq) & 1:
            return None
        return Color.WHITE if (self.color_bb[Color.WHITE] >> sq) & 1 else Color.BLACK

    def king_square(self, color: int) -> int:
        bb = self.piece_bb[PieceType.KING] & self.color_bb[color]
        return (bb & -bb).bit_length() - 1

    def pieces(self, color: int, pt: int) -> int:
        """Bitboard of ``color``'s pieces of type ``pt``."""
        return self.piece_bb[pt] & self.color_bb[color]

    # ------------------------------------------------------------------ #
    # Make / unmake
    # ------------------------------------------------------------------ #

    def make_move(self, move: Move) -> None:
        frm = move.from_sq
        to = move.to_sq
        flag = move.flag
        us = self.side_to_move
        them = us ^ 1
        moving_pt = self.mailbox[frm]

        if flag == FLAG_EN_PASSANT:
            captured = PieceType.PAWN
        elif move.is_capture():
            captured = self.mailbox[to]
        else:
            captured = PieceType.NONE

        self._history.append(
            _Undo(
                captured,
                self.castling_rights,
                self.ep_square,
                self.halfmove_clock,
                self.zobrist_key,
            )
        )

        # Clear any existing en-passant target (from the hash too).
        if self.ep_square != NO_SQUARE:
            self.zobrist_key ^= zobrist.EP_FILE_KEYS[file_of(self.ep_square)]
            self.ep_square = NO_SQUARE

        # Remove a captured piece.
        if flag == FLAG_EN_PASSANT:
            cap_sq = to - 8 if us == Color.WHITE else to + 8
            self._remove_piece(them, PieceType.PAWN, cap_sq)
        elif captured != PieceType.NONE:
            self._remove_piece(them, captured, to)

        # Move (or promote) the piece.
        promo = move.promotion_piece()
        if promo is not None:
            self._remove_piece(us, PieceType.PAWN, frm)
            self._put_piece(us, promo, to)
        else:
            self._move_piece(us, moving_pt, frm, to)

        # Castling: relocate the rook.
        if flag == FLAG_KING_CASTLE:
            if us == Color.WHITE:
                self._move_piece(us, PieceType.ROOK, H1, F1)
            else:
                self._move_piece(us, PieceType.ROOK, H8, F8)
        elif flag == FLAG_QUEEN_CASTLE:
            if us == Color.WHITE:
                self._move_piece(us, PieceType.ROOK, A1, D1)
            else:
                self._move_piece(us, PieceType.ROOK, A8, D8)

        # Set a new en-passant target on a double pawn push.
        if move.is_double_pawn_push():
            ep = (frm + to) // 2
            self.ep_square = ep
            self.zobrist_key ^= zobrist.EP_FILE_KEYS[file_of(ep)]

        # Update castling rights.
        new_castling = self.castling_rights & _CASTLE_MASK[frm] & _CASTLE_MASK[to]
        if new_castling != self.castling_rights:
            self.zobrist_key ^= zobrist.CASTLING_KEYS[self.castling_rights]
            self.zobrist_key ^= zobrist.CASTLING_KEYS[new_castling]
            self.castling_rights = new_castling

        # Halfmove clock: reset on pawn moves and captures.
        if moving_pt == PieceType.PAWN or captured != PieceType.NONE:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if us == Color.BLACK:
            self.fullmove_number += 1

        self.side_to_move = them
        self.zobrist_key ^= zobrist.SIDE_KEY

    def unmake_move(self, move: Move) -> None:
        us = self.side_to_move ^ 1  # the side that made the move
        them = self.side_to_move
        self.side_to_move = us

        frm = move.from_sq
        to = move.to_sq
        flag = move.flag

        # Reverse the piece movement (or promotion).
        if move.is_promotion():
            self._remove_piece(us, move.promotion_piece(), to)
            self._put_piece(us, PieceType.PAWN, frm)
        else:
            self._move_piece(us, self.mailbox[to], to, frm)

        # Reverse the rook for castling.
        if flag == FLAG_KING_CASTLE:
            if us == Color.WHITE:
                self._move_piece(us, PieceType.ROOK, F1, H1)
            else:
                self._move_piece(us, PieceType.ROOK, F8, H8)
        elif flag == FLAG_QUEEN_CASTLE:
            if us == Color.WHITE:
                self._move_piece(us, PieceType.ROOK, D1, A1)
            else:
                self._move_piece(us, PieceType.ROOK, D8, A8)

        undo = self._history.pop()

        # Restore a captured piece.
        if flag == FLAG_EN_PASSANT:
            cap_sq = to - 8 if us == Color.WHITE else to + 8
            self._put_piece(them, PieceType.PAWN, cap_sq)
        elif undo.captured != PieceType.NONE:
            self._put_piece(them, undo.captured, to)

        # Restore irreversible state (and the exact prior hash).
        self.castling_rights = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove
        self.zobrist_key = undo.key
        if us == Color.BLACK:
            self.fullmove_number -= 1

    def make_null_move(self) -> None:
        """Pass the turn to the opponent (used by null-move pruning).

        Only the side to move and the en-passant target change; no piece moves.
        Must not be called while in check.
        """
        us = self.side_to_move
        self._history.append(
            _Undo(
                PieceType.NONE,
                self.castling_rights,
                self.ep_square,
                self.halfmove_clock,
                self.zobrist_key,
            )
        )
        if self.ep_square != NO_SQUARE:
            self.zobrist_key ^= zobrist.EP_FILE_KEYS[file_of(self.ep_square)]
            self.ep_square = NO_SQUARE
        self.side_to_move = us ^ 1
        self.zobrist_key ^= zobrist.SIDE_KEY
        self.halfmove_clock += 1
        if us == Color.BLACK:
            self.fullmove_number += 1

    def unmake_null_move(self) -> None:
        us = self.side_to_move ^ 1
        self.side_to_move = us
        undo = self._history.pop()
        self.castling_rights = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove
        self.zobrist_key = undo.key
        if us == Color.BLACK:
            self.fullmove_number -= 1

    def has_non_pawn_material(self, color: int) -> bool:
        """Does ``color`` have any piece other than pawns and the king?"""
        non_pawn = self.color_bb[color] & ~(
            self.piece_bb[PieceType.PAWN] | self.piece_bb[PieceType.KING]
        )
        return non_pawn != 0

    # ------------------------------------------------------------------ #
    # FEN / display
    # ------------------------------------------------------------------ #

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                piece = self.piece_at(rank * 8 + file)
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece_symbol(piece[0], piece[1])
            if empty:
                row += str(empty)
            rows.append(row)
        placement = "/".join(rows)

        active = "w" if self.side_to_move == Color.WHITE else "b"

        rights = ""
        if self.castling_rights & WHITE_OO:
            rights += "K"
        if self.castling_rights & WHITE_OOO:
            rights += "Q"
        if self.castling_rights & BLACK_OO:
            rights += "k"
        if self.castling_rights & BLACK_OOO:
            rights += "q"
        rights = rights or "-"

        ep = "-" if self.ep_square == NO_SQUARE else square_name(self.ep_square)

        return f"{placement} {active} {rights} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def __str__(self) -> str:
        lines = []
        for rank in range(7, -1, -1):
            cells = []
            for file in range(8):
                piece = self.piece_at(rank * 8 + file)
                cells.append("." if piece is None else piece_symbol(piece[0], piece[1]))
            lines.append(f"{rank + 1}  " + " ".join(cells))
        lines.append("\n   a b c d e f g h")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"
