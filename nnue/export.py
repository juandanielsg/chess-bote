"""Quantize a trained NNUE checkpoint and export it to the C++ binary format.

Usage:
    uv run --group train python nnue/export.py --in nnue/net.pt --out nnue/net.bin

After writing the binary, it validates the quantization by comparing the float
model's eval against the integer (deployed) eval on a few positions.
"""

from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

from nnue_infer import HIDDEN, NUM_FEATURES, QA, QB, eval_white_cp, fen_features, load_net
from train import WDL_SCALE, NNUE

_SAMPLE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize + export the NNUE")
    ap.add_argument("--in", dest="inp", default="nnue/net.pt")
    ap.add_argument("--out", default="nnue/net.bin")
    args = ap.parse_args()

    model = NNUE()
    model.load_state_dict(torch.load(args.inp, map_location="cpu"))
    model.eval()

    ft_w = model.ft.weight.detach().numpy()      # [NF, H]
    ft_b = model.ft_bias.detach().numpy()         # [H]
    out_w = model.out.weight.detach().numpy().reshape(-1)  # [H]
    out_b = float(model.out.bias.detach().numpy()[0])      # cp

    ft_w_q = np.round(ft_w * QA).astype(np.int16)
    ft_b_q = np.round(ft_b * QA).astype(np.int16)
    out_w_q = np.round(out_w * QB).astype(np.int16)
    out_b_q = int(round(out_b * QA * QB))

    with open(args.out, "wb") as f:
        f.write(struct.pack("<4i", HIDDEN, NUM_FEATURES, QA, QB))
        f.write(ft_w_q.tobytes())
        f.write(ft_b_q.tobytes())
        f.write(out_w_q.tobytes())
        f.write(struct.pack("<q", out_b_q))
    print(f"wrote {args.out}")

    # Validate: float model vs deployed integer eval.
    net = load_net(args.out)
    print(f"{'position':<10} {'float cp':>9} {'int cp':>8}")
    for fen in _SAMPLE_FENS:
        feats = fen_features(fen)
        idx = torch.tensor(feats, dtype=torch.long)
        off = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            float_cp = float(model(idx, off).item())
        int_cp = eval_white_cp(net, fen)
        print(f"{fen.split()[0][:10]:<10} {float_cp:>9.1f} {int_cp:>8d}")


if __name__ == "__main__":
    main()
