"""Freeze a model: copy a checkpoint's weights to a separate, write-once file.

Keeps only what playing and fine-tuning need (weights, config, the step it was
trained to, the training args) and drops the optimizer state, so the file is
~1/3 the size of a resumable checkpoint. Refuses to overwrite an existing file
unless ``--force``, so a frozen base model can't be clobbered by accident.

Usage:
    python intubote/export_model.py --ckpt intubote/intubote.pt --out models/intubote_base.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze IntuBote weights to a separate file")
    ap.add_argument("--ckpt", required=True, help="training checkpoint (or snapshot)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} already exists; not overwriting it (use --force to replace it)")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    frozen = {"model": ck["model"], "config": ck["config"], "step": ck.get("step"),
              "args": ck.get("args"), "frozen_from": os.path.abspath(args.ckpt)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save(frozen, tmp)
    os.replace(tmp, args.out)
    print(f"froze {args.ckpt} (step {ck.get('step')}) -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
