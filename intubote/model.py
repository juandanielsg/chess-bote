"""IntuBote: a depth-0 transformer that maps a position straight to a move.

Input: the 70 side-to-move-relative tokens from ``encoding.encode_fen``.
Body:  a pre-norm transformer encoder over those tokens (no causal mask --
       every square attends to every other square).
Heads: read from the CLS token
         * policy: logits over the NUM_MOVES move vocabulary
         * value:  logit of the side to move's win probability
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from encoding import NUM_MOVES, SEQ_LEN, TOKEN_VOCAB


@dataclass
class Config:
    d_model: int = 256
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class IntuBote(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(TOKEN_VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, cfg.heads, d * cfg.ff_mult, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.policy = nn.Linear(d, NUM_MOVES)
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: [B, SEQ_LEN] integer -> (policy logits [B, NUM_MOVES], value logit [B])."""
        x = self.tok(tokens.long()) + self.pos
        x = self.norm(self.body(x))
        cls = x[:, 0]
        return self.policy(cls), self.value(cls).squeeze(-1)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> IntuBote:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = IntuBote(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
