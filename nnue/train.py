"""Train a simple NNUE for chess-bote.

Architecture (single, White-relative perspective -- the simplest real NNUE):

    768 binary features  ->  256 accumulator (feature transformer)
                         ->  clipped ReLU [0, 1]
                         ->  1 output (centipawns, White's perspective)

The 768 features are (color, piece_type, square): index = color*384 + pt*64 + sq.
The feature transformer is an ``EmbeddingBag`` (sum mode): it sums the weight
columns of the active features -- exactly the "accumulator" the C++ engine
maintains incrementally across make/unmake.

Trained in win/draw/loss space: target = sigmoid(cp / 400), the standard NNUE
objective, which down-weights huge evals and focuses on decisive differences.

Parsed features are cached next to the data file (``<data>.cache.npz``) so
repeated runs over the same data (e.g. the self-improvement loop's replay
window) skip FEN parsing entirely. A held-out validation split is scored every
epoch and the checkpoint with the best validation loss is the one saved.

Usage:
    uv run --group train python nnue/train.py --data nnue/data/positions.txt \
        --epochs 40 --out nnue/net.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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


class PositionData:
    """All samples as flat numpy arrays (features, bounds, labels).

    Sample i's feature indices are ``feats[bounds[i]:bounds[i+1]]``.
    Parsing FENs dominates load time, so the parsed arrays are cached in
    ``<path>.cache.npz`` and reused while the cache is newer than the data.
    """

    def __init__(self, path: str) -> None:
        cache = path + ".cache.npz"
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            z = np.load(cache)
            self.feats = z["feats"]
            self.bounds = z["bounds"]
            self.primary = z["primary"]
            self.cp = z["cp"] if "cp" in z else None
            return

        feats: list[int] = []
        bounds: list[int] = [0]
        primary: list[float] = []
        cp: list[float] = []
        has_cp = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format v1: "fen|cp"      -- primary is cp, no second label
                # Format v2: "fen|wdl|cp"  -- primary is wdl, cp is the search
                #            score (or HCE in older data), White-relative
                parts = line.split("|")
                if has_cp is None:
                    has_cp = len(parts) >= 3
                feats.extend(fen_features(parts[0]))
                bounds.append(len(feats))
                primary.append(float(parts[1]))
                if has_cp:
                    cp.append(float(parts[2]))

        self.feats = np.asarray(feats, dtype=np.int32)
        self.bounds = np.asarray(bounds, dtype=np.int64)
        self.primary = np.asarray(primary, dtype=np.float32)
        self.cp = np.asarray(cp, dtype=np.float32) if has_cp else None

        arrays = {"feats": self.feats, "bounds": self.bounds, "primary": self.primary}
        if self.cp is not None:
            arrays["cp"] = self.cp
        np.savez(cache, **arrays)

    def __len__(self) -> int:
        return len(self.primary)

    def batch(self, ids: np.ndarray):
        """Build (indices, offsets, primary, cp) tensors for the given samples."""
        lengths = self.bounds[ids + 1] - self.bounds[ids]
        offsets = np.zeros(len(ids), dtype=np.int64)
        np.cumsum(lengths[:-1], out=offsets[1:])
        indices = np.concatenate(
            [self.feats[self.bounds[i]:self.bounds[i + 1]] for i in ids]
        )
        cp = (
            torch.from_numpy(self.cp[ids])
            if self.cp is not None
            else None
        )
        return (
            torch.from_numpy(indices.astype(np.int64)),
            torch.from_numpy(offsets),
            torch.from_numpy(self.primary[ids]),
            cp,
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


def wdl_target(args, primary: torch.Tensor, cp: torch.Tensor | None) -> torch.Tensor:
    if cp is not None:
        # Mix the WDL game outcome with the search-score cp signal.
        return args.cp_mix * torch.sigmoid(cp / WDL_SCALE) + (1 - args.cp_mix) * primary
    if args.label == "wdl":
        return primary
    return torch.sigmoid(primary / WDL_SCALE)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the chess-bote NNUE")
    ap.add_argument("--data", default="nnue/data/positions.txt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="nnue/net.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--label",
        choices=["cp", "wdl"],
        default="cp",
        help="label type: 'cp' (engine eval) or 'wdl' (self-play outcome 0/0.5/1)",
    )
    ap.add_argument(
        "--cp-mix", type=float, default=0.5,
        help="weight of the search-score cp signal vs the WDL outcome when both "
             "are present (0=pure WDL, 1=pure cp)",
    )
    ap.add_argument(
        "--val-frac", type=float, default=0.05,
        help="held-out validation fraction (0 disables; best-val model is saved)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading {args.data} ...", file=sys.stderr)
    ds = PositionData(args.data)
    print(f"  {len(ds):,} positions", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))
    n_val = int(len(ds) * args.val_frac)
    val_ids, train_ids = perm[:n_val], perm[n_val:]

    model = NNUE()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    def val_loss() -> float:
        total = 0.0
        with torch.no_grad():
            for s in range(0, n_val, args.batch):
                ids = val_ids[s:s + args.batch]
                indices, offsets, primary, cp = ds.batch(ids)
                pred_wdl = torch.sigmoid(model(indices, offsets) / WDL_SCALE)
                target = wdl_target(args, primary, cp)
                total += loss_fn(pred_wdl, target).item() * len(ids)
        return total / n_val

    best_val = float("inf")
    best_state = None
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="training", unit="epoch",
                     file=sys.stderr)
    for epoch in epoch_bar:
        order = rng.permutation(train_ids)
        total = 0.0
        n = 0
        batch_starts = range(0, len(order), args.batch)
        batch_bar = tqdm(batch_starts, desc=f"  epoch {epoch:2d}", unit="batch",
                         leave=False, file=sys.stderr)
        for s in batch_bar:
            ids = order[s:s + args.batch]
            indices, offsets, primary, cp = ds.batch(ids)
            pred_wdl = torch.sigmoid(model(indices, offsets) / WDL_SCALE)
            target = wdl_target(args, primary, cp)
            loss = loss_fn(pred_wdl, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(ids)
            n += len(ids)
            batch_bar.set_postfix(loss=f"{total / n:.5f}")
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        postfix = {"loss": f"{total / n:.5f}", "lr": f"{lr:.1e}"}
        if n_val:
            v = val_loss()
            postfix["val"] = f"{v:.5f}"
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        epoch_bar.set_postfix(**postfix)
        print(f"epoch {epoch:3d}  train {total / n:.5f}"
              + (f"  val {postfix['val']}" if n_val else ""), file=sys.stderr)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"best val loss {best_val:.5f}", file=sys.stderr)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
