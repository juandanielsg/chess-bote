"""Train IntuBote on data from ``prepare_data.py``.

Losses (summed):
  * policy -- cross-entropy against either
      - ``--policy-target best``: Stockfish's best move (behavioral cloning), or
      - ``--policy-target soft``: a distribution over the stored candidate moves,
        softmax(win / --soft-temp), i.e. "better moves get more mass" -- a cheap
        step toward action-value training.
  * value  -- binary cross-entropy against the best move's win probability,
              weighted by ``--value-weight``.

Built for Colab: the whole dataset is moved to the GPU when it fits, training
uses mixed precision, and a resumable checkpoint (model + optimizer + step) is
written every ``--save-every`` steps, so a disconnect costs minutes, not hours.
Re-running the same command resumes automatically from ``--out``.

Usage:
    python intubote/train.py --data intubote/data/lichess2m --out intubote/intubote.pt \
        --steps 50000 --batch 1024
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Config, IntuBote, num_params  # noqa: E402


def load_data(path: str, device: torch.device, max_gpu_bytes: float):
    tokens = np.load(os.path.join(path, "tokens.npy"))
    moves = np.load(os.path.join(path, "moves.npy"))
    wins = np.load(os.path.join(path, "wins.npy")).astype(np.float32)
    nbytes = tokens.nbytes + moves.nbytes + wins.nbytes
    where = device if (device.type == "cuda" and nbytes <= max_gpu_bytes) else torch.device("cpu")
    t = torch.from_numpy(tokens).to(where)
    m = torch.from_numpy(moves.astype(np.int64)).to(where)
    w = torch.from_numpy(wins).to(where)
    print(f"data: {len(t):,} positions, {nbytes / 1e6:.0f} MB, kept on {where}", file=sys.stderr)
    return t, m, w


def policy_loss(logits, moves, wins, target: str, temp: float, smoothing: float):
    if target == "best":
        return F.cross_entropy(logits, moves[:, 0], label_smoothing=smoothing)
    valid = moves >= 0
    scores = torch.where(valid, wins / temp, torch.full_like(wins, -1e9))
    probs = torch.softmax(scores, dim=1)  # over the K candidates
    logp = F.log_softmax(logits.float(), dim=1)
    picked = torch.gather(logp, 1, moves.clamp(min=0))
    return -(probs * picked * valid).sum(1).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train IntuBote")
    ap.add_argument("--data", required=True, help="directory from prepare_data.py")
    ap.add_argument("--out", default="intubote/intubote.pt")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--policy-target", choices=["best", "soft"], default="best")
    ap.add_argument("--soft-temp", type=float, default=0.03,
                    help="win-probability temperature for --policy-target soft")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-gpu-data-gb", type=float, default=4.0,
                    help="move the dataset to the GPU if smaller than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing --out checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed)
    print(f"device: {device}", file=sys.stderr)

    tokens, moves, wins = load_data(args.data, device, args.max_gpu_data_gb * 1e9)
    n = len(tokens)
    perm = torch.from_numpy(np.random.default_rng(args.seed).permutation(n))
    n_val = max(1, int(n * args.val_frac))
    val_ids, train_ids = perm[:n_val], perm[n_val:]

    # Resume, or start fresh.
    ck = None
    if os.path.exists(args.out) and not args.no_resume:
        ck = torch.load(args.out, map_location="cpu", weights_only=False)
        cfg = Config(**ck["config"])
        print(f"resuming {args.out} at step {ck['step']}", file=sys.stderr)
    else:
        cfg = Config(d_model=args.d_model, layers=args.layers, heads=args.heads)
    model = IntuBote(cfg).to(device)
    print(f"model: {cfg.to_dict()}  {num_params(model) / 1e6:.1f}M params", file=sys.stderr)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step = 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        step = ck["step"]

    def batch(ids: torch.Tensor):
        ids_d = ids.to(tokens.device)
        return (tokens[ids_d].to(device, non_blocking=True),
                moves[ids_d].to(device, non_blocking=True),
                wins[ids_d].to(device, non_blocking=True))

    def losses(tok, mv, wn):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, vlogit = model(tok)
        logits, vlogit = logits.float(), vlogit.float()
        lp = policy_loss(logits, mv, wn, args.policy_target, args.soft_temp, args.label_smoothing)
        lv = F.binary_cross_entropy_with_logits(vlogit, wn[:, 0])
        return lp, lv, logits, vlogit

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        tot = {"policy": 0.0, "value": 0.0, "top1": 0.0, "top3": 0.0, "value_mae": 0.0}
        for s in range(0, n_val, 4096):
            tok, mv, wn = batch(val_ids[s:s + 4096])
            lp, lv, logits, vlogit = losses(tok, mv, wn)
            b = len(tok)
            top3 = logits.topk(3, dim=1).indices
            tot["policy"] += lp.item() * b
            tot["value"] += lv.item() * b
            tot["top1"] += (top3[:, 0] == mv[:, 0]).float().sum().item()
            tot["top3"] += (top3 == mv[:, :1]).any(1).float().sum().item()
            tot["value_mae"] += (torch.sigmoid(vlogit) - wn[:, 0]).abs().sum().item()
        model.train()
        return {k: v / n_val for k, v in tot.items()}

    def save() -> None:
        tmp = args.out + ".tmp"
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": cfg.to_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step,
                    "args": vars(args)}, tmp)
        os.replace(tmp, args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Metrics log for plotting (plot_metrics.py): one JSON object per eval.
    # A fresh run starts it over; a resumed run drops entries newer than the
    # checkpoint (logged after the last save, before a disconnect).
    metrics_path = args.out + ".metrics.jsonl"
    kept_lines = []
    if ck is not None and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            kept_lines = [ln for ln in f if ln.strip() and json.loads(ln)["step"] <= step]
    with open(metrics_path, "w") as f:
        f.writelines(kept_lines)

    model.train()
    g = torch.Generator().manual_seed(args.seed + step)
    t0, run_p, run_v, run_n = time.time(), 0.0, 0.0, 0
    while step < args.steps:
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        ids = train_ids[torch.randint(len(train_ids), (args.batch,), generator=g)]
        tok, mv, wn = batch(ids)
        lp, lv, _, _ = losses(tok, mv, wn)
        loss = lp + args.value_weight * lv
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        run_p, run_v, run_n = run_p + lp.item(), run_v + lv.item(), run_n + 1

        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            rate = run_n * args.batch / (time.time() - t0)
            print(f"step {step:7d}  lr {lr_at(step):.2e}  train policy {run_p / run_n:.3f} "
                  f"value {run_v / run_n:.4f} | val policy {v['policy']:.3f} "
                  f"top1 {v['top1']:.1%} top3 {v['top3']:.1%} value_mae {v['value_mae']:.3f} "
                  f"| {rate:,.0f} pos/s", file=sys.stderr, flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "lr": lr_at(step), "positions": step * args.batch,
                    "train_policy": run_p / run_n, "train_value": run_v / run_n,
                    "val_policy": v["policy"], "val_value": v["value"],
                    "top1": v["top1"], "top3": v["top3"], "value_mae": v["value_mae"],
                    "pos_per_s": rate, "time": time.time(),
                }) + "\n")
            t0, run_p, run_v, run_n = time.time(), 0.0, 0.0, 0
        if step % args.save_every == 0 or step == args.steps:
            save()

    print(f"saved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
