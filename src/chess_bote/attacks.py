"""Attack tables and sliding-piece attack generation.

Leaper attacks (knight, king, pawn) are fully precomputed lookup tables.

Slider attacks (bishop, rook, queen) are computed with the classical
*ray-attack* method: for each square and direction we precompute the ray
bitboard, then mask it off at the first blocker. This is simple and provably
correct -- the ideal reference for the pure-Python phase.

``rook_attacks(sq, occ)`` and ``bishop_attacks(sq, occ)`` are the seam where a
faster implementation (magic bitboards, or a C++ extension) can later drop in
without movegen needing to change.
"""

from __future__ import annotations

from .bitboard import BB_ALL
from .types import Color, NUM_SQUARES

# --------------------------------------------------------------------------- #
# Leaper attack tables
# --------------------------------------------------------------------------- #


def _in_bounds(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def _build_knight() -> list[int]:
    deltas = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    table = [0] * NUM_SQUARES
    for sq in range(NUM_SQUARES):
        f, r = sq & 7, sq >> 3
        bb = 0
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if _in_bounds(nf, nr):
                bb |= 1 << (nr * 8 + nf)
        table[sq] = bb
    return table


def _build_king() -> list[int]:
    deltas = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    table = [0] * NUM_SQUARES
    for sq in range(NUM_SQUARES):
        f, r = sq & 7, sq >> 3
        bb = 0
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if _in_bounds(nf, nr):
                bb |= 1 << (nr * 8 + nf)
        table[sq] = bb
    return table


def _build_pawn() -> list[list[int]]:
    # PAWN_ATTACKS[color][sq] = squares a pawn of `color` on `sq` attacks.
    table = [[0] * NUM_SQUARES, [0] * NUM_SQUARES]
    for sq in range(NUM_SQUARES):
        f, r = sq & 7, sq >> 3
        # White captures toward higher ranks.
        wbb = 0
        if _in_bounds(f - 1, r + 1):
            wbb |= 1 << ((r + 1) * 8 + (f - 1))
        if _in_bounds(f + 1, r + 1):
            wbb |= 1 << ((r + 1) * 8 + (f + 1))
        table[Color.WHITE][sq] = wbb
        # Black captures toward lower ranks.
        bbb = 0
        if _in_bounds(f - 1, r - 1):
            bbb |= 1 << ((r - 1) * 8 + (f - 1))
        if _in_bounds(f + 1, r - 1):
            bbb |= 1 << ((r - 1) * 8 + (f + 1))
        table[Color.BLACK][sq] = bbb
    return table


KNIGHT_ATTACKS: list[int] = _build_knight()
KING_ATTACKS: list[int] = _build_king()
PAWN_ATTACKS: list[list[int]] = _build_pawn()


# --------------------------------------------------------------------------- #
# Slider rays
# --------------------------------------------------------------------------- #
#
# Direction indices and whether stepping in that direction increases the square
# index (so we mask at the *lowest* blocker) or decreases it (mask at the
# *highest* blocker).

_DIRS = {
    "N": (0, 1, True),
    "E": (1, 0, True),
    "NE": (1, 1, True),
    "NW": (-1, 1, True),
    "S": (0, -1, False),
    "W": (-1, 0, False),
    "SE": (1, -1, False),
    "SW": (-1, -1, False),
}

_ROOK_DIRS = ("N", "E", "S", "W")
_BISHOP_DIRS = ("NE", "SE", "SW", "NW")


def _build_rays() -> dict[str, list[int]]:
    rays: dict[str, list[int]] = {}
    for name, (df, dr, _positive) in _DIRS.items():
        table = [0] * NUM_SQUARES
        for sq in range(NUM_SQUARES):
            f, r = sq & 7, sq >> 3
            bb = 0
            nf, nr = f + df, r + dr
            while _in_bounds(nf, nr):
                bb |= 1 << (nr * 8 + nf)
                nf += df
                nr += dr
            table[sq] = bb
        rays[name] = table
    return rays


_RAYS = _build_rays()


def _ray_attacks(sq: int, occ: int, dirs: tuple[str, ...]) -> int:
    attacks = 0
    for name in dirs:
        ray = _RAYS[name][sq]
        blockers = ray & occ
        if blockers:
            if _DIRS[name][2]:  # positive direction -> first blocker is the lsb
                first = (blockers & -blockers).bit_length() - 1
            else:  # negative direction -> first blocker is the msb
                first = blockers.bit_length() - 1
            ray ^= _RAYS[name][first]  # trim everything beyond the blocker
        attacks |= ray
    return attacks


# --------------------------------------------------------------------------- #
# Magic bitboards
# --------------------------------------------------------------------------- #
#
# For each square we precompute a "relevant occupancy" mask (the squares a
# slider could be blocked on, excluding board edges, which can never block
# further travel). A magic multiplier hashes any occupancy subset of that mask
# into a dense index, so a slider lookup becomes:
#
#     occ &= mask; index = (occ * magic) >> shift; return table[index]
#
# The magics are searched for once at import time with a fixed RNG seed, so the
# result is deterministic. The slow ray attacks above are the reference used to
# fill the tables (and a perfect cross-check in the tests).

_BB_ALL = (1 << 64) - 1

# Magic multipliers, found once with a deterministic search (random seed
# 0xC0FFEE) and frozen here so import does not pay for the search. Each is
# verified against the slow ray attacks for all occupancies in the tests.
_ROOK_MAGIC = [
    0x1080009080400A20, 0x0A40021000200044, 0x20800A2002100080, 0x4280100008008084,
    0x0E00040200081020, 0x0280010200800400, 0x0080020000800100, 0x0200002040840112,
    0x0A00800090204000, 0x0800804000200080, 0x1000802000801008, 0x2028801001080080,
    0x0C10800400808800, 0x0842802400800200, 0x0216005804420001, 0x2554800041000080,
    0x8D10208000400080, 0x1253030020400082, 0x2020848010042000, 0x0200828008001000,
    0x0004808008000402, 0x9228808004000200, 0x0240040008020110, 0x00C0020001006084,
    0x1110800080204000, 0x04C0400140201000, 0x0420008080100020, 0x0004230100100209,
    0x2302080280040080, 0x2082020080040080, 0x1005000100040200, 0x820AD04200208104,
    0x0000400080800020, 0x880040A001401000, 0x0020801000802000, 0x0100801000800801,
    0x0001800402800800, 0x0160400408012010, 0x010810020400A821, 0x0922004682000914,
    0x0080004420024000, 0x0040008020008040, 0x2402002040820011, 0x1004201042020008,
    0x0004000800808004, 0x4004040002008080, 0x0000020001008080, 0x0520040880420009,
    0x0000522085020200, 0x0090400906208100, 0xB000100020068480, 0x0480100180080280,
    0x280C818400080080, 0x6000200440100801, 0x0000814810020400, 0x4010004104288600,
    0x0140120100204082, 0x030A001100402086, 0x1000208040081202, 0x0003001000200409,
    0x020200201014882A, 0x4081000400080203, 0x0000410810009204, 0x0000008402C11026,
]
_BISHOP_MAGIC = [
    0x0040810202020022, 0x6020920AD3010004, 0x00100C1448400011, 0x0008208024522800,
    0x1004504000800C00, 0x10020904A1100602, 0x080900A820092000, 0x000922080C040200,
    0x000210D010010040, 0x2410611214210020, 0x08200418008102E0, 0x0008044042800080,
    0x80040D1040000001, 0x8000024820140000, 0x0386008424208480, 0x2A80210100900400,
    0x0004000890040800, 0x0420A0120C140080, 0x0090000104068010, 0x4084000A01220000,
    0x0000803400A04000, 0x0802001100620240, 0x4404018101015080, 0xC800808C40484800,
    0x0484046140082800, 0x1091600014080208, 0x000C020001080505, 0x0008080100220020,
    0x000A001022005002, 0x2480490042008209, 0x1090A20008921008, 0x00010141030400A0,
    0x940104910C202000, 0x8222420220101000, 0x8014009040020400, 0x0028020080080080,
    0x0008020010040900, 0x2230020200112580, 0x200204A608340218, 0x0008008020011100,
    0x0181019820004110, 0x104C00C410008460, 0x0002020201080204, 0x0083184200800800,
    0x0440020202000410, 0x2040208081008184, 0x0104080081200410, 0x0090024E00204241,
    0x0002015002100101, 0x1402088884100210, 0x0010262201102442, 0x4110000084040000,
    0x4004009022120080, 0x00202004A1020000, 0x0084099004008040, 0x8044480800408400,
    0xA008440209302200, 0x44200D0488010900, 0x1240410600443200, 0x8541900221084810,
    0x1000080908210100, 0x402000081001820A, 0x0040220805182484, 0x1402380808208021,
]


def _rook_mask(sq: int) -> int:
    f, r = sq & 7, sq >> 3
    m = 0
    for nr in range(r + 1, 7):
        m |= 1 << (nr * 8 + f)
    for nr in range(r - 1, 0, -1):
        m |= 1 << (nr * 8 + f)
    for nf in range(f + 1, 7):
        m |= 1 << (r * 8 + nf)
    for nf in range(f - 1, 0, -1):
        m |= 1 << (r * 8 + nf)
    return m


def _bishop_mask(sq: int) -> int:
    f, r = sq & 7, sq >> 3
    m = 0
    for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        nf, nr = f + df, r + dr
        while 0 < nf < 7 and 0 < nr < 7:
            m |= 1 << (nr * 8 + nf)
            nf += df
            nr += dr
    return m


def _occupancy_subsets(mask: int):
    """Yield every subset of the set bits of ``mask`` (carry-rippler trick)."""
    sub = 0
    while True:
        yield sub
        sub = (sub - mask) & mask
        if sub == 0:
            break


def _build_tables(mask_fn, magics: list[int], dirs: tuple[str, ...]):
    """Fill per-square attack tables from the frozen magics (no search)."""
    masks = [0] * NUM_SQUARES
    shifts = [0] * NUM_SQUARES
    tables: list[list[int]] = [[] for _ in range(NUM_SQUARES)]
    for sq in range(NUM_SQUARES):
        mask = mask_fn(sq)
        shift = 64 - mask.bit_count()
        table: list[int] = [0] * (1 << mask.bit_count())
        for occ in _occupancy_subsets(mask):
            idx = ((occ * magics[sq]) & _BB_ALL) >> shift
            table[idx] = _ray_attacks(sq, occ, dirs)
        masks[sq] = mask
        shifts[sq] = shift
        tables[sq] = table
    return masks, shifts, tables


_ROOK_MASK, _ROOK_SHIFT, _ROOK_TABLE = _build_tables(_rook_mask, _ROOK_MAGIC, _ROOK_DIRS)
_BISHOP_MASK, _BISHOP_SHIFT, _BISHOP_TABLE = _build_tables(
    _bishop_mask, _BISHOP_MAGIC, _BISHOP_DIRS
)


def rook_attacks(sq: int, occ: int) -> int:
    """Squares a rook on ``sq`` attacks given occupancy ``occ`` (blockers included)."""
    occ &= _ROOK_MASK[sq]
    return _ROOK_TABLE[sq][((occ * _ROOK_MAGIC[sq]) & _BB_ALL) >> _ROOK_SHIFT[sq]]


def bishop_attacks(sq: int, occ: int) -> int:
    occ &= _BISHOP_MASK[sq]
    return _BISHOP_TABLE[sq][((occ * _BISHOP_MAGIC[sq]) & _BB_ALL) >> _BISHOP_SHIFT[sq]]


def queen_attacks(sq: int, occ: int) -> int:
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


# --------------------------------------------------------------------------- #
# Geometry: BETWEEN and LINE tables (for pins and check evasion)
# --------------------------------------------------------------------------- #
#
# BETWEEN[a][b] -- squares strictly between a and b along a shared rank/file/
#                  diagonal (0 if they are not aligned). Used for blocking a
#                  sliding check and for detecting the blocker on a pin ray.
# LINE[a][b]    -- the full board line through a and b (0 if not aligned). A
#                  pinned piece may only move along LINE[king][piece].


def _build_between_and_line():
    between = [[0] * NUM_SQUARES for _ in range(NUM_SQUARES)]
    line = [[0] * NUM_SQUARES for _ in range(NUM_SQUARES)]
    for a in range(NUM_SQUARES):
        fa, ra = a & 7, a >> 3
        for b in range(NUM_SQUARES):
            if a == b:
                continue
            fb, rb = b & 7, b >> 3
            same_rank = ra == rb
            same_file = fa == fb
            same_diag = abs(fa - fb) == abs(ra - rb)
            if not (same_rank or same_file or same_diag):
                continue
            df = (fb > fa) - (fb < fa)
            dr = (rb > ra) - (rb < ra)
            # Squares strictly between a and b.
            bb = 0
            nf, nr = fa + df, ra + dr
            while (nf, nr) != (fb, rb):
                bb |= 1 << (nr * 8 + nf)
                nf += df
                nr += dr
            between[a][b] = bb
            # Full line through a and b (both directions to the edges).
            ln = 1 << a
            for sgn in (1, -1):
                nf, nr = fa + df * sgn, ra + dr * sgn
                while 0 <= nf < 8 and 0 <= nr < 8:
                    ln |= 1 << (nr * 8 + nf)
                    nf += df * sgn
                    nr += dr * sgn
            line[a][b] = ln
    return between, line


BETWEEN, LINE = _build_between_and_line()
