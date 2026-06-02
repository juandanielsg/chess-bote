"""Transposition table.

Caches the result of searching a position, keyed by its Zobrist hash, so that
positions reached by different move orders (transpositions) are not re-searched,
and so the best move from a shallower search can order the next iteration.

Each entry records a *bound*, not just a score, because alpha-beta only proves
an exact value for some nodes:

    EXACT  -- the score is the true value (search stayed inside the window).
    LOWER  -- a fail-high: the true value is at least ``score`` (beta cutoff).
    UPPER  -- a fail-low: the true value is at most ``score``.

The table is a fixed-size array indexed by the low bits of the key, with a
depth-preferred replacement policy. Entries are plain tuples to keep memory and
allocation overhead low:

    (key, depth, score, flag, move_value)
"""

from __future__ import annotations

EXACT = 0
LOWER = 1
UPPER = 2


class TranspositionTable:
    def __init__(self, size: int = 1 << 20) -> None:
        # size must be a power of two so ``key & mask`` selects a slot.
        if size & (size - 1) != 0:
            raise ValueError("transposition table size must be a power of two")
        self._size = size
        self._mask = size - 1
        self._table: list[tuple[int, int, int, int, int] | None] = [None] * size

    def clear(self) -> None:
        self._table = [None] * self._size

    def probe(self, key: int) -> tuple[int, int, int, int, int] | None:
        """Return the entry for ``key`` if present (and not a hash collision)."""
        entry = self._table[key & self._mask]
        if entry is not None and entry[0] == key:
            return entry
        return None

    def store(
        self, key: int, depth: int, score: int, flag: int, move_value: int
    ) -> None:
        idx = key & self._mask
        existing = self._table[idx]
        # Depth-preferred: keep the deeper result, but always refresh the same
        # position (same key) so its best move/bound stay current.
        if existing is None or existing[0] != key or depth >= existing[1]:
            self._table[idx] = (key, depth, score, flag, move_value)

    def __len__(self) -> int:
        return sum(1 for e in self._table if e is not None)
