"""Search: negamax with alpha-beta pruning.

Features:
    * Negamax framing (one routine, scores always relative to side to move).
    * Alpha-beta pruning.
    * Quiescence search at the leaves (captures + promotions only) to avoid the
      horizon effect from stopping in the middle of a capture sequence.
    * MVV-LVA capture ordering, with the previous iteration's best root move
      tried first.
    * Iterative deepening with an optional move-time budget.

Mate scores are stored as ``MATE - ply`` so shorter mates are preferred, and so
the score is independent of the depth at which the mate was found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .board import Board
from .eval import PIECE_VALUE, evaluate
from .move import Move
from .movegen import generate_legal, in_check
from .tt import EXACT, LOWER, UPPER, TranspositionTable
from .types import PieceType

INF = 1_000_000_000
MATE = 1_000_000
MATE_THRESHOLD = MATE - 1000  # scores beyond this are forced mates

MAX_PLY = 256
_NULL_MOVE_R = 2  # depth reduction for the null-move search


class _Timeout(Exception):
    """Raised internally to abort a search that has run out of time."""


def _score_to_tt(score: int, ply: int) -> int:
    """Make a mate score relative to the current node before storing it."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    """Re-root a mate score read back from the table to the current node."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def _mvv_lva(board: Board, move: Move) -> int:
    """Capture/promotion ordering score: Most Valuable Victim - Least Valuable Attacker."""
    score = 0
    if move.is_capture():
        victim = PieceType.PAWN if move.is_en_passant() else board.mailbox[move.to_sq]
        attacker = board.mailbox[move.from_sq]
        score = PIECE_VALUE[victim] * 16 - PIECE_VALUE[attacker]
    promo = move.promotion_piece()
    if promo is not None:
        score += PIECE_VALUE[promo]
    return score


@dataclass
class SearchResult:
    best_move: Move | None
    score: int
    depth: int
    nodes: int
    seconds: float
    pv: list[Move] = field(default_factory=list)

    @property
    def nps(self) -> float:
        return self.nodes / self.seconds if self.seconds > 0 else 0.0


class Searcher:
    """A reusable searcher. One instance per search keeps node/time state tidy."""

    def __init__(self, tt_size: int = 1 << 20) -> None:
        self.nodes = 0
        self.stopped = False
        self.tt = TranspositionTable(tt_size)
        self._deadline: float | None = None
        self._best_root_move: Move | None = None
        # Two killer (quiet cutoff) moves per ply, and a [side][from][to]
        # history table. Both are ordering heuristics, reset per search.
        self._killers: list[list[Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self._history: list[list[list[int]]] = [
            [[0] * 64 for _ in range(64)] for _ in range(2)
        ]

    def stop(self) -> None:
        """Request the running search to abort as soon as possible."""
        self.stopped = True

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def search(
        self,
        board: Board,
        max_depth: int = 64,
        movetime: float | None = None,
        info_callback: Callable[[SearchResult], None] | None = None,
    ) -> SearchResult:
        """Iteratively deepen until ``max_depth`` or the ``movetime`` budget (s).

        ``info_callback`` (if given) is called with a :class:`SearchResult`
        after each completed depth -- used by the UCI layer to emit ``info``.
        """
        self.nodes = 0
        self.stopped = False
        self._deadline = None if movetime is None else time.perf_counter() + movetime
        self._best_root_move = None
        self._killers = [[None, None] for _ in range(MAX_PLY)]
        self._history = [[[0] * 64 for _ in range(64)] for _ in range(2)]

        start = time.perf_counter()
        best_move: Move | None = None
        best_score = 0
        completed_depth = 0

        for depth in range(1, max_depth + 1):
            try:
                score, move = self._search_root(board, depth)
            except _Timeout:
                break
            best_move, best_score, completed_depth = move, score, depth
            self._best_root_move = move
            if info_callback is not None:
                info_callback(
                    SearchResult(
                        best_move=best_move,
                        score=best_score,
                        depth=completed_depth,
                        nodes=self.nodes,
                        seconds=time.perf_counter() - start,
                        pv=[best_move] if best_move else [],
                    )
                )
            # Stop early on a proven mate or once out of time.
            if abs(best_score) >= MATE_THRESHOLD:
                break
            if self._deadline is not None and time.perf_counter() >= self._deadline:
                break

        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=completed_depth,
            nodes=self.nodes,
            seconds=time.perf_counter() - start,
            pv=[best_move] if best_move else [],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _checkpoint(self) -> None:
        self.nodes += 1
        if (self.nodes & 2047) == 0:
            if self.stopped:
                raise _Timeout
            if self._deadline is not None and time.perf_counter() >= self._deadline:
                raise _Timeout

    def _search_root(self, board: Board, depth: int) -> tuple[int, Move | None]:
        moves = generate_legal(board)
        if not moves:
            # Terminal at the root: mate or stalemate.
            return (-MATE if in_check(board) else 0), None

        # Prefer the table's best move, falling back to last iteration's.
        entry = self.tt.probe(board.zobrist_key)
        tt_move = Move(entry[4]) if entry and entry[4] else self._best_root_move
        self._order_moves(board, moves, 0, tt_move)

        alpha = -INF
        best_move = moves[0]
        for i, move in enumerate(moves):
            board.make_move(move)
            if i == 0:
                score = -self._negamax(board, depth - 1, -INF, -alpha, 1)
            else:
                # PVS: scout with a null window, re-search if it beats alpha.
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if score > alpha:
                    score = -self._negamax(board, depth - 1, -INF, -alpha, 1)
            board.unmake_move(move)
            if score > alpha:
                alpha = score
                best_move = move

        # The root is searched with a full window, so the result is exact.
        self.tt.store(board.zobrist_key, depth, _score_to_tt(alpha, 0), EXACT, best_move.value)
        return alpha, best_move

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._checkpoint()
        alpha_orig = alpha
        key = board.zobrist_key

        # Transposition table probe.
        tt_move: Move | None = None
        entry = self.tt.probe(key)
        if entry is not None:
            _, e_depth, e_score, e_flag, e_move = entry
            if e_move:
                tt_move = Move(e_move)
            if ply > 0 and e_depth >= depth:
                s = _score_from_tt(e_score, ply)
                if e_flag == EXACT:
                    return s
                if e_flag == LOWER and s >= beta:
                    return s
                if e_flag == UPPER and s <= alpha:
                    return s

        if depth == 0:
            return self._quiescence(board, alpha, beta)

        checked = in_check(board)

        # Null-move pruning: if giving the opponent a free move still fails high,
        # this node is so good we can prune. Skipped when in check, in shallow /
        # near-mate windows, or with only pawns left (zugzwang risk).
        is_pv = beta - alpha > 1
        if (
            not is_pv
            and not checked
            and depth >= 3
            and beta < MATE_THRESHOLD
            and board.has_non_pawn_material(board.side_to_move)
        ):
            board.make_null_move()
            null_score = -self._negamax(
                board, depth - 1 - _NULL_MOVE_R, -beta, -beta + 1, ply + 1
            )
            board.unmake_null_move()
            if null_score >= beta:
                return beta

        moves = generate_legal(board)
        if not moves:
            return -MATE + ply if checked else 0

        self._order_moves(board, moves, ply, tt_move)
        side = board.side_to_move

        best = -INF
        best_move = moves[0]
        for i, move in enumerate(moves):
            board.make_move(move)
            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.unmake_move(move)
            if score > best:
                best = score
                best_move = move
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    self._on_cutoff(move, side, depth, ply)
                    break  # beta cutoff

        # Record the result with the bound it represents.
        if best <= alpha_orig:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt.store(key, depth, _score_to_tt(best, ply), flag, best_move.value)
        return best

    def _quiescence(self, board: Board, alpha: int, beta: int) -> int:
        self._checkpoint()

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves = [
            m for m in generate_legal(board) if m.is_capture() or m.is_promotion()
        ]
        moves.sort(key=lambda m: _mvv_lva(board, m), reverse=True)

        for move in moves:
            board.make_move(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.unmake_move(move)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _on_cutoff(self, move: Move, side: int, depth: int, ply: int) -> None:
        """Reward a quiet move that caused a beta cutoff (killers + history)."""
        if move.is_capture() or move.is_promotion():
            return
        killers = self._killers[ply]
        if killers[0] != move:
            killers[1] = killers[0]
            killers[0] = move
        self._history[side][move.from_sq][move.to_sq] += depth * depth

    def _order_moves(
        self, board: Board, moves: list[Move], ply: int, first: Move | None
    ) -> None:
        """Sort moves in place: hash move, captures (MVV-LVA), killers, history."""
        killers = self._killers[ply]
        history = self._history[board.side_to_move]

        def key(move: Move) -> int:
            if first is not None and move == first:
                return 1 << 30
            if move.is_capture() or move.is_promotion():
                return 1_000_000 + _mvv_lva(board, move)
            if move == killers[0]:
                return 900_000
            if move == killers[1]:
                return 800_000
            return history[move.from_sq][move.to_sq]

        moves.sort(key=key, reverse=True)


def find_best_move(
    board: Board, depth: int = 4, movetime: float | None = None
) -> SearchResult:
    """Convenience wrapper: search ``board`` and return the result."""
    return Searcher().search(board, max_depth=depth, movetime=movetime)
