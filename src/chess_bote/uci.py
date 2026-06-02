"""UCI (Universal Chess Interface) protocol handler.

Implements enough of UCI to play in GUIs (CuteChess, Arena, BanksiaGUI) and
with lichess-bot:

    uci, isready, ucinewgame, position, go, stop, quit

The search runs on a background thread so ``stop`` and ``quit`` stay responsive
and ``go infinite`` is interruptible. ``info`` lines are emitted after each
completed iterative-deepening depth, followed by ``bestmove`` when the search
finishes.
"""

from __future__ import annotations

import sys
import threading

from . import native
from .board import START_FEN, Board
from .movegen import parse_move
from .search import MATE, MATE_THRESHOLD, SearchResult, Searcher
from .types import Color

ENGINE_NAME = "chess-bote"
ENGINE_AUTHOR = "juandanielsg"


def _format_score(score: int) -> str:
    """UCI score token: 'cp <centipawns>' or 'mate <moves>'."""
    if score >= MATE_THRESHOLD:
        return f"mate {(MATE - score + 1) // 2}"
    if score <= -MATE_THRESHOLD:
        return f"mate {-((MATE + score + 1) // 2)}"
    return f"cp {score}"


class UCIEngine:
    def __init__(self) -> None:
        self.board = Board()
        self.use_native = native.HAS_NATIVE
        self.searcher = Searcher()  # pure-Python fallback
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if not self.dispatch(line):
                break

    def dispatch(self, line: str) -> bool:
        """Handle one command line. Returns False to terminate the loop."""
        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "isready":
            self._send("readyok")
        elif cmd == "ucinewgame":
            self._stop_search()
            self.board = Board()
            if self.use_native:
                native.new_game()
            else:
                self.searcher.tt.clear()
        elif cmd == "setoption":
            self._cmd_setoption(tokens)
        elif cmd == "position":
            self._stop_search()
            self._cmd_position(tokens)
        elif cmd == "go":
            self._cmd_go(tokens)
        elif cmd == "stop":
            self._stop_search()
        elif cmd in ("quit", "exit"):
            self._stop_search()
            return False
        # Unknown commands are ignored, per the UCI spec.
        return True

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def _cmd_uci(self) -> None:
        self._send(f"id name {ENGINE_NAME}")
        self._send(f"id author {ENGINE_AUTHOR}")
        # NNUE net path; "handcrafted" (or empty) uses the hand-crafted eval.
        self._send("option name EvalFile type string default handcrafted")
        self._send("uciok")

    def _cmd_setoption(self, tokens: list[str]) -> None:
        if "name" not in tokens:
            return
        ni = tokens.index("name")
        if "value" in tokens:
            vi = tokens.index("value")
            name = " ".join(tokens[ni + 1 : vi])
            value = " ".join(tokens[vi + 1 :])
        else:
            name, value = " ".join(tokens[ni + 1 :]), ""

        if name.lower() == "evalfile":
            v = value.strip()
            if v == "" or v.lower() in ("handcrafted", "none", "<empty>"):
                native.set_nnue(False)
            elif not native.load_nnue(v):
                self._send(f"info string failed to load EvalFile: {v}")

    def _cmd_position(self, tokens: list[str]) -> None:
        idx = 1
        if idx < len(tokens) and tokens[idx] == "startpos":
            self.board = Board(START_FEN)
            idx += 1
        elif idx < len(tokens) and tokens[idx] == "fen":
            fen = " ".join(tokens[idx + 1 : idx + 7])
            self.board = Board(fen)
            idx += 7
        else:
            return

        if idx < len(tokens) and tokens[idx] == "moves":
            for uci in tokens[idx + 1 :]:
                move = parse_move(self.board, uci)
                if move is None:
                    self._send(f"info string illegal move ignored: {uci}")
                    break
                self.board.make_move(move)

    def _cmd_go(self, tokens: list[str]) -> None:
        self._stop_search()

        max_depth = 64
        movetime: float | None = None
        infinite = False
        wtime = btime = winc = binc = None
        movestogo: int | None = None

        i = 1
        while i < len(tokens):
            t = tokens[i]
            if t == "depth":
                i += 1
                max_depth = int(tokens[i])
            elif t == "movetime":
                i += 1
                movetime = int(tokens[i]) / 1000.0
            elif t == "wtime":
                i += 1
                wtime = int(tokens[i])
            elif t == "btime":
                i += 1
                btime = int(tokens[i])
            elif t == "winc":
                i += 1
                winc = int(tokens[i])
            elif t == "binc":
                i += 1
                binc = int(tokens[i])
            elif t == "movestogo":
                i += 1
                movestogo = int(tokens[i])
            elif t == "infinite":
                infinite = True
            i += 1

        # Derive a time budget from the clock if no explicit limit was given.
        if movetime is None and not infinite:
            movetime = self._budget(wtime, btime, winc, binc, movestogo)
        if infinite:
            movetime = None
        # Plain "go" with no limits at all: a sane default so we don't run away.
        if movetime is None and not infinite and max_depth == 64:
            movetime = 3.0

        self._start_search(max_depth, movetime)

    # ------------------------------------------------------------------ #
    # Time management
    # ------------------------------------------------------------------ #

    def _budget(
        self,
        wtime: int | None,
        btime: int | None,
        winc: int | None,
        binc: int | None,
        movestogo: int | None,
    ) -> float | None:
        if self.board.side_to_move == Color.WHITE:
            remaining, inc = wtime, winc or 0
        else:
            remaining, inc = btime, binc or 0
        if remaining is None:
            return None
        # Spread the clock over the expected remaining moves, add most of the
        # increment, and keep a safety margin so we never flag.
        moves = movestogo if movestogo else 25
        budget = remaining / moves + inc * 0.8
        budget = min(budget, remaining * 0.9)
        return max(budget, 10) / 1000.0  # ms -> s, at least 10ms

    # ------------------------------------------------------------------ #
    # Search thread plumbing
    # ------------------------------------------------------------------ #

    def _start_search(self, max_depth: int, movetime: float | None) -> None:
        # Search a copy so the engine state can't be mutated mid-search.
        position = Board(self.board.fen())

        def worker() -> None:
            if self.use_native:
                result = native.native_search(
                    position, max_depth, movetime, on_info=self._send_native_info
                )
            else:
                result = self.searcher.search(
                    position, max_depth, movetime, info_callback=self._send_info
                )
            self._send_bestmove(result)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _stop_search(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if self.use_native:
                native.stop()
            else:
                self.searcher.stop()
            self._thread.join()
        self._thread = None

    def _send_info(self, result: SearchResult) -> None:
        ms = int(result.seconds * 1000)
        nps = int(result.nps)
        pv = " ".join(m.uci() for m in result.pv)
        self._send(
            f"info depth {result.depth} score {_format_score(result.score)} "
            f"nodes {result.nodes} nps {nps} time {ms} pv {pv}"
        )

    def _send_native_info(self, d: dict) -> None:
        """Emit a UCI info line from the native search's per-depth callback."""
        seconds = d["time"]
        ms = int(seconds * 1000)
        nps = int(d["nodes"] / seconds) if seconds > 0 else 0
        pv = " ".join(d["pv"])
        self._send(
            f"info depth {d['depth']} score {_format_score(d['score'])} "
            f"nodes {d['nodes']} nps {nps} time {ms} pv {pv}"
        )

    def _send_bestmove(self, result: SearchResult) -> None:
        move = result.best_move
        self._send(f"bestmove {move.uci() if move else '0000'}")

    @staticmethod
    def _send(text: str) -> None:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def main() -> None:
    UCIEngine().run()


if __name__ == "__main__":
    main()
