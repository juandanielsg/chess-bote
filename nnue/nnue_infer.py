"""Reference integer inference for the quantized NNUE.

This mirrors exactly what the C++ engine will do, so it serves two purposes:
a sanity check during export, and the oracle for the C++ cross-check test.

Quantization scheme (must match the C++ side):
    accumulator weights/bias scaled by QA (clipped-ReLU range becomes [0, QA])
    output weights scaled by QB
    cp = trunc((sum(out_w_q * acc_clamped) + out_b_q) / (QA * QB))

Binary format (little-endian):
    int32  hidden, num_features, QA, QB
    int16  ft_w[num_features * hidden]   (row-major: feature, then hidden)
    int16  ft_b[hidden]
    int16  out_w[hidden]
    int64  out_b   (already scaled by QA*QB)
"""

from __future__ import annotations

import struct

import numpy as np

from chess_bote.board import Board

QA = 255
QB = 64
HIDDEN = 256
NUM_FEATURES = 768


def fen_features(fen: str) -> list[int]:
    b = Board(fen)
    feats: list[int] = []
    for color in (0, 1):
        base_c = color * 384
        for pt in range(6):
            bb = b.piece_bb[pt] & b.color_bb[color]
            base = base_c + pt * 64
            while bb:
                sq = (bb & -bb).bit_length() - 1
                bb &= bb - 1
                feats.append(base + sq)
    return feats


def load_net(path: str) -> dict:
    with open(path, "rb") as f:
        hidden, nf, qa, qb = struct.unpack("<4i", f.read(16))
        ft_w = np.frombuffer(f.read(nf * hidden * 2), dtype="<i2").reshape(nf, hidden)
        ft_b = np.frombuffer(f.read(hidden * 2), dtype="<i2")
        out_w = np.frombuffer(f.read(hidden * 2), dtype="<i2")
        (out_b,) = struct.unpack("<q", f.read(8))
    return {
        "hidden": hidden, "nf": nf, "qa": qa, "qb": qb,
        "ft_w": ft_w.astype(np.int64),
        "ft_b": ft_b.astype(np.int64),
        "out_w": out_w.astype(np.int64),
        "out_b": out_b,
    }


def _trunc_div(num: int, den: int) -> int:
    q = abs(num) // den
    return q if num >= 0 else -q


def eval_white_cp(net: dict, fen: str) -> int:
    """Quantized integer evaluation, in centipawns from White's perspective."""
    acc = net["ft_b"].copy()  # [hidden]
    for fi in fen_features(fen):
        acc += net["ft_w"][fi]
    acc = np.clip(acc, 0, net["qa"])
    out_i = int((net["out_w"] * acc).sum())
    return _trunc_div(out_i + int(net["out_b"]), net["qa"] * net["qb"])
