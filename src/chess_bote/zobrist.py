"""Zobrist hashing keys.

The position hash is maintained incrementally on the board: each make/unmake
XORs in/out only the keys for what changed (pieces moved, castling rights,
en-passant file, side to move).

Keys are generated once from a fixed seed so hashes are reproducible across
runs (handy for debugging and for opening-book / transposition-table sharing).
"""

from __future__ import annotations

import random

from .types import NUM_SQUARES

_SEED = 0x1A2B3C4D5E6F7081

_rng = random.Random(_SEED)


def _rand64() -> int:
    return _rng.getrandbits(64)


# piece_keys[color][piece_type][square]
PIECE_KEYS: list[list[list[int]]] = [
    [[_rand64() for _ in range(NUM_SQUARES)] for _ in range(6)] for _ in range(2)
]

# One key per castling-rights combination (4 bits -> 16 entries).
CASTLING_KEYS: list[int] = [_rand64() for _ in range(16)]

# One key per en-passant file (only the file matters for repetition).
EP_FILE_KEYS: list[int] = [_rand64() for _ in range(8)]

# XOR'd in whenever it is Black to move.
SIDE_KEY: int = _rand64()
