# IntuBote: depth-0 transformer

IntuBote picks a move by "intuition": a transformer looks at the position once and outputs a move, **with no search**. It learns from Stockfish evaluations in the [Lichess evaluation database](https://database.lichess.org/#evals). The idea follows DeepMind's *Grandmaster-Level Chess Without Search* (Ruoss et al., 2024), at Colab scale.

It lives next to the NNUE engine (`nnue/`) and shares chess-bote's board and legal move generator. Illegal moves are masked out, so IntuBote always plays a legal move.

## Pipeline

| Step | File | Output |
|---|---|---|
| Stream and encode data | `prepare_data.py` | `tokens.npy`, `moves.npy`, `wins.npy` |
| Train (GPU, resumable) | `train.py` | `intubote.pt` checkpoint + `intubote.pt.metrics.jsonl` |
| Chart training progress | `plot_metrics.py` | accuracy / policy loss / value loss (/ puzzles) panels, live in the notebook |
| Freeze a model | `export_model.py` | write-once weights file, e.g. `models/intubote_base.pt` |
| Puzzle data + benchmark | `prepare_puzzles.py` | fine-tuning set + held-out `bench.npz` |
| Score puzzles | `puzzle_bench.py` | solve rate by rating band and theme, models side by side |
| Fine-tune | `train.py --init --extra` | e.g. `intubote_puzzles.pt` |
| Ask for a move | `engine.py` | top moves + win probability |
| Play in a GUI | `uci.py` | UCI engine |
| Do it all on Colab | `colab_intubote.ipynb` | |

## Encoding (`encoding.py`)

Everything is **side-to-move relative**: when Black is to move, the board is mirrored and the colors swapped, so the network always plays "up the board".

- **Position:** 70 tokens: `[CLS]`, then 64 squares (empty / our 6 pieces / their 6 pieces), 4 castling flags (ours O-O and O-O-O, then theirs), and the en-passant file.
- **Moves:** a fixed vocabulary of 1,880 moves: every queen-line or knight jump between two squares (this also covers castling as `e1g1`/`e1c1`), plus pawn promotions to the 8th rank.
- **Value:** win probability for the side to move, using Lichess's cp→win curve.

## Model (`model.py`)

A pre-norm transformer encoder in which every square attends to every other. Two heads read the CLS token:
- **policy:** logits over the 1,880 moves
- **value:** win-probability logit

| Size | d_model / layers / heads | Params |
|---|---|---|
| tiny | 128 / 4 / 4 | ~1.1M |
| small | 256 / 8 / 8 | ~7M |
| medium | 512 / 8 / 8 | ~26M |

## Training targets (`train.py --policy-target`)

- `best`: cross-entropy against Stockfish's best move (behavioral cloning). Start here.
- `soft`: spreads the target across the stored candidate moves (up to 5 Stockfish lines per position), weighted by `softmax(win / --soft-temp)`. This is a cheap step toward action-value training.

The value head is always trained against the best line's win probability.

## Puzzle benchmark and fine-tuning

`prepare_puzzles.py` streams the [Lichess puzzle database](https://database.lichess.org/#puzzles). Each solver move in a puzzle becomes a training position. The value target is left empty (NaN), so puzzles only train the policy.

- **Held out:** 1% of puzzles, chosen by a hash of the puzzle id, are never trained on. Up to 2,000 well-established puzzles per rating band (<1000 ... 2600+) from that pool form the benchmark. Reading continues until every band is full, so the benchmark is the same whatever `--count` is.
- **Scoring:** a puzzle is solved only if the model's top legal move is right at every solver move, with no search. On the last move any mate is accepted, as on Lichess. Because a puzzle fails at its first wrong move, scoring each move after the correct line is equivalent to playing it through, so the whole benchmark runs as one batched pass. `train.py --puzzle-bench` therefore scores it at every evaluation.
- **Fine-tuning:** `train.py --init base.pt --extra <puzzles> --extra-frac 0.5` starts from the frozen base and mixes puzzles with base data in every batch. The validation split still comes from `--data` with the same seed, i.e. the base run's held-out positions, so the validation numbers show whether the model is forgetting normal play.

```bash
uv run python intubote/export_model.py --ckpt intubote/intubote.pt --out intubote/models/base.pt
uv run python intubote/prepare_puzzles.py --count 200000 --out intubote/data/puzzles
uv run python intubote/puzzle_bench.py --bench intubote/data/puzzles/bench.npz --ckpt intubote/models/base.pt
uv run python intubote/train.py --data intubote/data/lichess200k --init intubote/models/base.pt     --extra intubote/data/puzzles --puzzle-bench intubote/data/puzzles/bench.npz     --out intubote/intubote_puzzles.pt --steps 5000 --batch 256 --lr 1e-4 --warmup 500
uv run python intubote/puzzle_bench.py --bench intubote/data/puzzles/bench.npz     --ckpt intubote/models/base.pt intubote/intubote_puzzles.pt --themes
```

The benchmark measures the network alone, so it doesn't include `engine.py`'s mate-in-one rule.

## Local usage

```bash
uv sync --group intubote
uv run python intubote/prepare_data.py --count 200000 --out intubote/data/lichess200k
uv run python intubote/train.py --data intubote/data/lichess200k --out intubote/intubote.pt \
    --steps 5000 --batch 256 --d-model 128 --layers 4 --heads 4
uv run python intubote/engine.py --ckpt intubote/intubote.pt --fen "<FEN>"
uv run python intubote/uci.py --ckpt intubote/intubote.pt
```

Real training belongs on a GPU. See `colab_intubote.ipynb`.

## Known limitations and next steps

- **Tactics:** with no search, it can miss short combinations.
- **Converting wins:** in won positions many moves look equally good, so a pure policy can dither. `engine.py` therefore plays a mate in one when one exists (UCI option `MateInOne`, on by default).
- **No history:** the position tokens don't include repetitions or the 50-move counter.
- **Next:** learn from its own games (Stockfish labels the positions IntuBote reaches in play, DAgger-style), then self-play with search (MCTS) once the network is strong enough.
- **Ideas:** full action-value training (score every legal move); history tokens; using IntuBote's policy for move ordering in chess-bote's alpha-beta search.
