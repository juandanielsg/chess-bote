"""Magic-bitboard slider attacks must match the slow ray reference exactly."""

import random

import pytest

from chess_bote import attacks as A


@pytest.mark.parametrize("sq", range(64))
def test_rook_magic_matches_rays(sq):
    rng = random.Random(sq)
    for _ in range(64):
        occ = rng.getrandbits(64)
        assert A.rook_attacks(sq, occ) == A._ray_attacks(sq, occ, A._ROOK_DIRS)


@pytest.mark.parametrize("sq", range(64))
def test_bishop_magic_matches_rays(sq):
    rng = random.Random(1000 + sq)
    for _ in range(64):
        occ = rng.getrandbits(64)
        assert A.bishop_attacks(sq, occ) == A._ray_attacks(sq, occ, A._BISHOP_DIRS)


def test_queen_is_rook_or_bishop():
    rng = random.Random(7)
    for _ in range(2000):
        sq = rng.randrange(64)
        occ = rng.getrandbits(64)
        assert A.queen_attacks(sq, occ) == A.rook_attacks(sq, occ) | A.bishop_attacks(
            sq, occ
        )
