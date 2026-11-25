"""Play with a trained IntuBote checkpoint.

The network proposes a move from the position alone; chess-bote's legal move
generator masks out illegal moves, so IntuBote never plays one. One optional
pragmatic exception to "depth 0": ``mate_in_one`` plays a mating move if one
exists (pure policy nets are known to dither in won positions).

CLI:
    python intubote/engine.py --ckpt intubote/intubote.pt --fen "<FEN>" --top 5
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding import black_to_move, encode_fen, encode_move, win_to_cp  # noqa: E402
from model import load_checkpoint  # noqa: E402

from chess_bote.board import Board  # noqa: E402
from chess_bote.movegen import generate_legal, in_check  # noqa: E402


class Engine:
    def __init__(self, ckpt: str, device: str = "cpu", mate_in_one: bool = True) -> None:
        self.device = torch.device(device)
        self.model = load_checkpoint(ckpt, self.device)
        self.mate_in_one = mate_in_one

    @torch.no_grad()
    def analyse(self, board: Board) -> tuple[list[tuple[str, float]], float]:
        """Legal moves with probabilities (best first) and the side-to-move win prob."""
        fen = board.fen()
        black = black_to_move(fen)
        legal = [m.uci() for m in generate_legal(board)]
        if not legal:
            return [], 0.0
        tok = torch.from_numpy(encode_fen(fen)).unsqueeze(0).to(self.device)
        logits, vlogit = self.model(tok)
        logits = logits[0].float()
        idx = torch.tensor([encode_move(u, black) for u in legal], device=self.device)
        probs = torch.softmax(logits[idx], dim=0).tolist()
        ranked = sorted(zip(legal, probs), key=lambda x: -x[1])
        return ranked, torch.sigmoid(vlogit[0]).item()

    def _mating_move(self, board: Board) -> str | None:
        for m in generate_legal(board):
            board.make_move(m)
            mate = not generate_legal(board) and in_check(board)
            board.unmake_move(m)
            if mate:
                return m.uci()
        return None

    def best_move(self, board: Board) -> tuple[str | None, float]:
        """Return (uci move or None if no legal moves, side-to-move win prob)."""
        ranked, win = self.analyse(board)
        if not ranked:
            return None, win
        if self.mate_in_one:
            mate = self._mating_move(board)
            if mate:
                return mate, 1.0
        return ranked[0][0], win


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask IntuBote for a move")
    ap.add_argument("--ckpt", default="intubote/intubote.pt")
    ap.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    eng = Engine(args.ckpt)
    board = Board(args.fen)
    ranked, win = eng.analyse(board)
    print(f"win prob (side to move) {win:.3f}  ~{win_to_cp(win):+.0f} cp")
    for uci, p in ranked[: args.top]:
        print(f"  {uci:6s} {p:6.1%}")


if __name__ == "__main__":
    main()
