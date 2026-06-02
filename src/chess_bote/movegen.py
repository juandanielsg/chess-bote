"""Legal move generation.

Moves are generated *strictly legally* in one pass -- no make/unmake filter.
The method is the standard one:

1.  Find the pieces giving check (``checkers``) and the squares the enemy
    attacks with our king removed (``king_danger``) -- the king may not move
    into those.
2.  Double check  -> only king moves are legal.
3.  Single check  -> non-king moves must land on the ``check_mask``: capture the
    checker, or (for a sliding checker) block on a square between it and the king.
4.  Pinned pieces -> may only move along the line through the king and the pin
    (``LINE[king][piece]``).
5.  En passant has its own discovered-check test, because the capture removes two
    pawns from a rank at once, which the pin logic above cannot see.

``is_square_attacked`` / ``in_check`` remain for callers that need a one-off
attack query (the search uses ``in_check``; tests and UCI use the rest).
"""

from __future__ import annotations

from .attacks import (
    BETWEEN,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    LINE,
    PAWN_ATTACKS,
    bishop_attacks,
    rook_attacks,
)
from .bitboard import BB_ALL, lsb, scan_forward
from .board import (
    BLACK_OO,
    BLACK_OOO,
    WHITE_OO,
    WHITE_OOO,
    Board,
)
from .move import (
    FLAG_CAPTURE,
    FLAG_DOUBLE_PAWN,
    FLAG_EN_PASSANT,
    FLAG_KING_CASTLE,
    FLAG_PROMO_B,
    FLAG_PROMO_B_CAP,
    FLAG_PROMO_N,
    FLAG_PROMO_N_CAP,
    FLAG_PROMO_Q,
    FLAG_PROMO_Q_CAP,
    FLAG_PROMO_R,
    FLAG_PROMO_R_CAP,
    FLAG_QUEEN_CASTLE,
    FLAG_QUIET,
    Move,
)
from .types import (
    B1, B8, C1, C8, D1, D8, E1, E8, F1, F8, G1, G8,
    Color,
    PieceType,
)

_QUIET_PROMOS = (FLAG_PROMO_Q, FLAG_PROMO_R, FLAG_PROMO_B, FLAG_PROMO_N)
_CAP_PROMOS = (FLAG_PROMO_Q_CAP, FLAG_PROMO_R_CAP, FLAG_PROMO_B_CAP, FLAG_PROMO_N_CAP)
_SLIDERS = (PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN)


def is_square_attacked(board: Board, sq: int, by_color: int) -> bool:
    """Is ``sq`` attacked by any piece of ``by_color``?"""
    occ = board.occupied
    if PAWN_ATTACKS[by_color ^ 1][sq] & board.pieces(by_color, PieceType.PAWN):
        return True
    if KNIGHT_ATTACKS[sq] & board.pieces(by_color, PieceType.KNIGHT):
        return True
    if KING_ATTACKS[sq] & board.pieces(by_color, PieceType.KING):
        return True
    queens = board.pieces(by_color, PieceType.QUEEN)
    if bishop_attacks(sq, occ) & (board.pieces(by_color, PieceType.BISHOP) | queens):
        return True
    if rook_attacks(sq, occ) & (board.pieces(by_color, PieceType.ROOK) | queens):
        return True
    return False


def in_check(board: Board, color: int | None = None) -> bool:
    """Is ``color`` (default: side to move) currently in check?"""
    if color is None:
        color = board.side_to_move
    return is_square_attacked(board, board.king_square(color), color ^ 1)


def generate_legal(board: Board) -> list[Move]:
    us = board.side_to_move
    them = us ^ 1
    moves: list[Move] = []

    king_sq = board.king_square(us)
    occ = board.occupied
    own = board.color_bb[us]
    enemy = board.color_bb[them]

    their_pawns = board.pieces(them, PieceType.PAWN)
    their_knights = board.pieces(them, PieceType.KNIGHT)
    bq = board.piece_bb[PieceType.BISHOP] | board.piece_bb[PieceType.QUEEN]
    rq = board.piece_bb[PieceType.ROOK] | board.piece_bb[PieceType.QUEEN]
    their_bq = bq & enemy
    their_rq = rq & enemy

    # 1. Checkers and king-danger squares (sliders see through our king).
    checkers = (
        (KNIGHT_ATTACKS[king_sq] & their_knights)
        | (PAWN_ATTACKS[us][king_sq] & their_pawns)
        | (bishop_attacks(king_sq, occ) & their_bq)
        | (rook_attacks(king_sq, occ) & their_rq)
    )

    occ_no_king = occ ^ (1 << king_sq)
    danger = 0
    for sq in scan_forward(their_pawns):
        danger |= PAWN_ATTACKS[them][sq]
    for sq in scan_forward(their_knights):
        danger |= KNIGHT_ATTACKS[sq]
    danger |= KING_ATTACKS[board.king_square(them)]
    for sq in scan_forward(their_bq):
        danger |= bishop_attacks(sq, occ_no_king)
    for sq in scan_forward(their_rq):
        danger |= rook_attacks(sq, occ_no_king)

    # King moves (never onto a danger square).
    for to in scan_forward(KING_ATTACKS[king_sq] & ~own & ~danger):
        moves.append(Move.create(king_sq, to, FLAG_CAPTURE if (enemy >> to) & 1 else FLAG_QUIET))

    num_checkers = checkers.bit_count()
    if num_checkers >= 2:
        return moves  # double check -> only the king may move

    if num_checkers == 1:
        checker_sq = lsb(checkers)
        if board.mailbox[checker_sq] in _SLIDERS:
            check_mask = BETWEEN[king_sq][checker_sq] | checkers
        else:
            check_mask = checkers  # must capture the (leaper) checker
    else:
        check_mask = BB_ALL
        _gen_castling(board, us, danger, occ, moves)

    # 4. Pinned pieces: an enemy slider with exactly one of our pieces between
    #    it and the king pins that piece to the king-slider line.
    pinned = 0
    snipers = (rook_attacks(king_sq, 0) & their_rq) | (
        bishop_attacks(king_sq, 0) & their_bq
    )
    for sniper in scan_forward(snipers):
        blockers = BETWEEN[king_sq][sniper] & occ
        if blockers and (blockers & (blockers - 1)) == 0 and (blockers & own):
            pinned |= blockers

    _gen_pawns(board, us, king_sq, pinned, check_mask, occ, enemy, their_rq, their_bq, moves)
    _gen_knights(board, us, own, enemy, pinned, check_mask, moves)
    _gen_sliders(board, us, king_sq, own, enemy, occ, pinned, check_mask, moves)
    return moves


def _gen_castling(board: Board, us: int, danger: int, occ: int, moves: list[Move]) -> None:
    rights = board.castling_rights
    if us == Color.WHITE:
        if (
            rights & WHITE_OO
            and not (occ >> F1) & 1 and not (occ >> G1) & 1
            and not (danger >> E1) & 1 and not (danger >> F1) & 1 and not (danger >> G1) & 1
        ):
            moves.append(Move.create(E1, G1, FLAG_KING_CASTLE))
        if (
            rights & WHITE_OOO
            and not (occ >> D1) & 1 and not (occ >> C1) & 1 and not (occ >> B1) & 1
            and not (danger >> E1) & 1 and not (danger >> D1) & 1 and not (danger >> C1) & 1
        ):
            moves.append(Move.create(E1, C1, FLAG_QUEEN_CASTLE))
    else:
        if (
            rights & BLACK_OO
            and not (occ >> F8) & 1 and not (occ >> G8) & 1
            and not (danger >> E8) & 1 and not (danger >> F8) & 1 and not (danger >> G8) & 1
        ):
            moves.append(Move.create(E8, G8, FLAG_KING_CASTLE))
        if (
            rights & BLACK_OOO
            and not (occ >> D8) & 1 and not (occ >> C8) & 1 and not (occ >> B8) & 1
            and not (danger >> E8) & 1 and not (danger >> D8) & 1 and not (danger >> C8) & 1
        ):
            moves.append(Move.create(E8, C8, FLAG_QUEEN_CASTLE))


def _gen_knights(
    board: Board, us: int, own: int, enemy: int, pinned: int, check_mask: int, moves: list[Move]
) -> None:
    for frm in scan_forward(board.pieces(us, PieceType.KNIGHT)):
        if (pinned >> frm) & 1:
            continue  # a pinned knight can never move
        for to in scan_forward(KNIGHT_ATTACKS[frm] & ~own & check_mask):
            moves.append(Move.create(frm, to, FLAG_CAPTURE if (enemy >> to) & 1 else FLAG_QUIET))


def _gen_sliders(
    board: Board, us: int, king_sq: int, own: int, enemy: int, occ: int,
    pinned: int, check_mask: int, moves: list[Move],
) -> None:
    for pt, attack in (
        (PieceType.BISHOP, bishop_attacks),
        (PieceType.ROOK, rook_attacks),
        (PieceType.QUEEN, _queen_attacks),
    ):
        for frm in scan_forward(board.pieces(us, pt)):
            targets = attack(frm, occ) & ~own & check_mask
            if (pinned >> frm) & 1:
                targets &= LINE[king_sq][frm]
            for to in scan_forward(targets):
                moves.append(Move.create(frm, to, FLAG_CAPTURE if (enemy >> to) & 1 else FLAG_QUIET))


def _queen_attacks(sq: int, occ: int) -> int:
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)


def _gen_pawns(
    board: Board, us: int, king_sq: int, pinned: int, check_mask: int,
    occ: int, enemy: int, their_rq: int, their_bq: int, moves: list[Move],
) -> None:
    if us == Color.WHITE:
        forward, start_rank, promo_rank = 8, 1, 7
    else:
        forward, start_rank, promo_rank = -8, 6, 0

    for frm in scan_forward(board.pieces(us, PieceType.PAWN)):
        line = LINE[king_sq][frm] if (pinned >> frm) & 1 else BB_ALL

        # Pushes.
        to = frm + forward
        if not (occ >> to) & 1 and (line >> to) & 1:
            if (check_mask >> to) & 1:
                _add_pawn(frm, to, promo_rank, False, moves)
            if (frm >> 3) == start_rank:
                to2 = to + forward
                if not (occ >> to2) & 1 and (line >> to2) & 1 and (check_mask >> to2) & 1:
                    moves.append(Move.create(frm, to2, FLAG_DOUBLE_PAWN))

        # Captures and en passant.
        for cap in scan_forward(PAWN_ATTACKS[us][frm]):
            if (enemy >> cap) & 1:
                if (line >> cap) & 1 and (check_mask >> cap) & 1:
                    _add_pawn(frm, cap, promo_rank, True, moves)
            elif cap == board.ep_square:
                captured = cap - 8 if us == Color.WHITE else cap + 8
                # Simulate the capture and confirm the king is not exposed.
                occ2 = (occ ^ (1 << frm) ^ (1 << captured)) | (1 << cap)
                if not (rook_attacks(king_sq, occ2) & their_rq) and not (
                    bishop_attacks(king_sq, occ2) & their_bq
                ):
                    moves.append(Move.create(frm, cap, FLAG_EN_PASSANT))


def _add_pawn(frm: int, to: int, promo_rank: int, capture: bool, moves: list[Move]) -> None:
    if (to >> 3) == promo_rank:
        for flag in (_CAP_PROMOS if capture else _QUIET_PROMOS):
            moves.append(Move.create(frm, to, flag))
    else:
        moves.append(Move.create(frm, to, FLAG_CAPTURE if capture else FLAG_QUIET))


def parse_move(board: Board, uci: str) -> Move | None:
    """Resolve a UCI move string (e.g. 'e2e4', 'e7e8q', 'e1g1') to a legal Move."""
    for move in generate_legal(board):
        if move.uci() == uci:
            return move
    return None
