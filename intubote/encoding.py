"""IntuBote board / move encoding.

Everything is expressed from the **side to move's** point of view: when Black is
to move the board is mirrored vertically (rank r -> 7 - r) and the colors are
swapped, so the network only ever sees "us" moving up the board. This halves
what it has to learn. Moves are mirrored the same way (square ^ 56), and
``decode_move`` maps them back.

Position tokens (``SEQ_LEN`` = 70, ids into a ``TOKEN_VOCAB``-sized table):

    [CLS] sq0 .. sq63  us_OO us_OOO them_OO them_OOO  ep

    CLS                 24
    square (a1..h8)     0 empty, 1-6 our P N B R Q K, 7-12 their P N B R Q K
    castling flag       13 off, 14 on
    en-passant file     15 none, 16-23 file a-h

A learned positional embedding tells the model which slot is which, so the
four castling slots can share the same two ids.

Move vocabulary (``NUM_MOVES`` = 1880), also side-to-move relative:
every (from, to) pair reachable by a queen or a knight on an empty board
(1792; covers all piece moves including castling as e1g1/e1c1), plus pawn
promotions from rank 7 to rank 8 (22 from->to pairs x 4 pieces = 88).
"""

from __future__ import annotations

import numpy as np

SEQ_LEN = 70
TOKEN_VOCAB = 25
CLS = 24
_CASTLE_OFF, _CASTLE_ON = 13, 14
_EP_NONE = 15

_PIECE_INDEX = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
_PROMOS = "nbrq"


def _build_move_vocab() -> list[str]:
    moves: list[str] = []
    knight = {(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)}
    for frm in range(64):
        ff, fr = frm % 8, frm // 8
        for to in range(64):
            if to == frm:
                continue
            tf, tr = to % 8, to // 8
            df, dr = tf - ff, tr - fr
            queen_line = df == 0 or dr == 0 or abs(df) == abs(dr)
            if queen_line or (df, dr) in knight:
                moves.append(sq_name(frm) + sq_name(to))
    for ff in range(8):
        for tf in (ff - 1, ff, ff + 1):
            if 0 <= tf < 8:
                for p in _PROMOS:
                    moves.append(f"{'abcdefgh'[ff]}7{'abcdefgh'[tf]}8{p}")
    return moves


def sq_name(sq: int) -> str:
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


def _sq_index(name: str) -> int:
    return (ord(name[0]) - 97) + 8 * (int(name[1]) - 1)


MOVE_VOCAB: list[str] = _build_move_vocab()
MOVE_INDEX: dict[str, int] = {m: i for i, m in enumerate(MOVE_VOCAB)}
NUM_MOVES = len(MOVE_VOCAB)


def flip_move(uci: str) -> str:
    """Mirror a UCI move vertically (a2a4 <-> a7a5); promotion suffix kept."""
    f, t = _sq_index(uci[0:2]) ^ 56, _sq_index(uci[2:4]) ^ 56
    return sq_name(f) + sq_name(t) + uci[4:]


def black_to_move(fen: str) -> bool:
    return fen.split()[1] == "b"


def encode_move(uci: str, black: bool) -> int:
    """Real-board UCI move -> vocab index (side-to-move frame). -1 if unknown."""
    return MOVE_INDEX.get(flip_move(uci) if black else uci, -1)


def decode_move(index: int, black: bool) -> str:
    """Vocab index -> real-board UCI move."""
    m = MOVE_VOCAB[index]
    return flip_move(m) if black else m


def encode_fen(fen: str, out: np.ndarray | None = None) -> np.ndarray:
    """FEN (4 or 6 fields) -> uint8 token array of length SEQ_LEN."""
    parts = fen.split()
    placement, side, castling = parts[0], parts[1], parts[2]
    ep = parts[3] if len(parts) > 3 else "-"
    black = side == "b"

    tok = out if out is not None else np.empty(SEQ_LEN, dtype=np.uint8)
    tok[0] = CLS
    tok[1:65] = 0
    rank, file = 7, 0
    for c in placement:
        if c == "/":
            rank -= 1
            file = 0
        elif c.isdigit():
            file += int(c)
        else:
            is_white = c.isupper()
            ours = is_white != black
            sq = rank * 8 + file
            if black:
                sq ^= 56
            tok[1 + sq] = 1 + _PIECE_INDEX[c.lower()] + (0 if ours else 6)
            file += 1

    us_oo, us_ooo, them_oo, them_ooo = ("k", "q", "K", "Q") if black else ("K", "Q", "k", "q")
    for i, flag in enumerate((us_oo, us_ooo, them_oo, them_ooo)):
        tok[65 + i] = _CASTLE_ON if flag in castling else _CASTLE_OFF
    tok[69] = _EP_NONE if ep == "-" else 16 + (ord(ep[0]) - 97)
    return tok


# Lichess's cp -> win-probability curve (side-to-move relative), used as the
# value target: https://lichess.org/page/accuracy
_WIN_K = 0.00368208


def cp_to_win(cp: float) -> float:
    return 1.0 / (1.0 + np.exp(-_WIN_K * cp))


def win_to_cp(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)) / _WIN_K)
