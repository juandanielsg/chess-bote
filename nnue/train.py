"""Train a simple NNUE for chess-bote.

Architecture (single, White-relative perspective -- the simplest real NNUE):

    768 binary features  ->  256 accumulator (feature transformer)
                         ->  clipped ReLU [0, 1]
                         ->  1 output (centipawns, White's perspective)

The 768 features are (color, piece_type, square): index = color*384 + pt*64 + sq.
The feature transformer is an ``EmbeddingBag`` (sum mode): it sums the weight
columns of the active features -- exactly the "accumulator" the C++ engine will
maintain incrementally across make/unmake.

Trained in win/draw/loss space: target = sigmoid(cp / 400), the standard NNUE
objective, which down-weights huge evals and focuses on decisive differences.

Usage:
    uv run --group train python nnue/train.py --data nnue/data/positions.txt \
        --epochs 40 --out nnue/net.pt
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from chess_bote.board import Board

NUM_FEATURES = 768
HIDDEN = 256
WDL_SCALE = 400.0  # cp -> win probability scaling


def fen_features(fen: str) -> list[int]:
    """Active 768-feature indices for a FEN (color*384 + piece_type*64 + square)."""
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


class PositionDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.samples: list[tuple[list[int], float]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fen, cp = line.rsplit("|", 1)
                self.samples.append((fen_features(fen), float(cp)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]


def collate(batch):
    indices: list[int] = []
    offsets: list[int] = []
    targets: list[float] = []
    off = 0
    for feats, cp in batch:
        offsets.append(off)
        indices.extend(feats)
        off += len(feats)
        targets.append(cp)
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
    )


class NNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ft = nn.EmbeddingBag(NUM_FEATURES, HIDDEN, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        acc = self.ft(indices, offsets) + self.ft_bias
        hidden = torch.clamp(acc, 0.0, 1.0)
        return self.out(hidden).squeeze(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the chess-bote NNUE")
    ap.add_argument("--data", default="nnue/data/positions.txt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="nnue/net.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading {args.data} ...", file=sys.stderr)
    ds = PositionDataset(args.data)
    print(f"  {len(ds)} positions", file=sys.stderr)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    model = NNUE()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for indices, offsets, cp in dl:
            pred = model(indices, offsets)
            target_wdl = torch.sigmoid(cp / WDL_SCALE)
            pred_wdl = torch.sigmoid(pred / WDL_SCALE)
            loss = loss_fn(pred_wdl, target_wdl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(cp)
            n += len(cp)
        print(f"epoch {epoch:3d}  wdl-mse {total / n:.5f}", file=sys.stderr)

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
