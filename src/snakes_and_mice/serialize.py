"""Reading and writing match results — the tournament results file (§6).

A **tournament** is simply *any set of matches* (§6), joined only by a shared
results file. This module holds a stable JSON encoding of
:class:`~snakes_and_mice.result.MatchResult`, one per line, appended to a
JSON-Lines file. That encoding is the contract between the runners that write it
and the tally that reads it; it round-trips a ``MatchResult``.

Enums are encoded by their string value, a :class:`~snakes_and_mice.core.Move` by
its cell labels, and the :class:`~snakes_and_mice.core.Side`-keyed names dict by the
sides' string values — so a line is human-readable and stable. Keeping this out of
:mod:`~snakes_and_mice.result` leaves the result types pure data, with no JSON or
filesystem dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import Move, Side, TurnOutcome
from .faults import IllegalMove, PlayerFaultReason, TournamentError
from .result import GameResult, MatchResult, PlayerFaultDetail, Termination


def encode_match_result(result: MatchResult) -> dict[str, object]:
    """Encode a :class:`MatchResult` to a JSON-able dict (the results-line schema)."""
    return {
        "names": {side.value: name for side, name in result.names.items()},
        "num_games": result.num_games,
        "mouse_wins": result.mouse_wins,
        "snake_wins": result.snake_wins,
        "cats_games": result.cats_games,
        "mouse_faults": result.mouse_faults,
        "snake_faults": result.snake_faults,
        "faults": [_encode_game_result(g) for g in result.faults],
        "aborted": result.aborted,
    }


def decode_match_result(data: Any) -> MatchResult:
    """Rebuild a :class:`MatchResult` from :func:`encode_match_result`'s output.

    Raises :class:`TournamentError` (via :func:`load_match_result`) if the data does
    not match the schema; here it lets the underlying error surface for wrapping.
    """
    names: dict[Side, str] = {Side(k): v for k, v in data["names"].items()}
    return MatchResult(
        names=names,
        num_games=data["num_games"],
        mouse_wins=data["mouse_wins"],
        snake_wins=data["snake_wins"],
        cats_games=data["cats_games"],
        mouse_faults=data["mouse_faults"],
        snake_faults=data["snake_faults"],
        faults=[_decode_game_result(g) for g in data["faults"]],
        aborted=data["aborted"],
    )


def dump_match_result(result: MatchResult) -> str:
    """Serialize a :class:`MatchResult` to a single-line JSON string (no newline)."""
    return json.dumps(encode_match_result(result), separators=(",", ":"))


def load_match_result(line: str) -> MatchResult:
    """Parse one results-file line back into a :class:`MatchResult`.

    Raises :class:`TournamentError` for any malformed line.
    """
    try:
        return decode_match_result(json.loads(line))
    except (json.JSONDecodeError, ValueError, TypeError, KeyError, IllegalMove) as exc:
        raise TournamentError(f"malformed match result: {exc}") from exc


def append_match_result(result: MatchResult, path: Path) -> None:
    """Append one :class:`MatchResult` as a line to the JSON-Lines file at ``path``.

    Creates any missing parent directories. Appending (never rewriting) keeps a
    crashed run's file valid up to the last completed match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dump_match_result(result) + "\n")


def read_match_results(path: Path) -> list[MatchResult]:
    """Read every :class:`MatchResult` from the JSON-Lines file at ``path``.

    Blank lines are ignored. Raises :class:`TournamentError` for a missing file or a
    malformed line (with the offending line number).
    """
    if not path.exists():
        raise TournamentError(f"results file {path} not found")
    results: list[MatchResult] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line: str = raw.strip()
            if not line:
                continue
            try:
                results.append(load_match_result(line))
            except TournamentError as exc:
                raise TournamentError(f"{path}:{lineno}: {exc}") from exc
    return results


def _encode_game_result(result: GameResult) -> dict[str, object]:
    return {
        "termination": result.termination.value,
        "winner": result.winner.value if result.winner is not None else None,
        "fault": _encode_fault(result.fault) if result.fault is not None else None,
        "error": result.error,
    }


def _decode_game_result(data: Any) -> GameResult:
    winner: Any = data["winner"]
    fault: Any = data["fault"]
    return GameResult(
        termination=Termination(data["termination"]),
        winner=Side(winner) if winner is not None else None,
        fault=_decode_fault(fault) if fault is not None else None,
        error=data["error"],
    )


def _encode_fault(fault: PlayerFaultDetail) -> dict[str, object]:
    move: Move | None = fault.attempted_move
    claimed: TurnOutcome | None = fault.claimed_outcome
    actual: TurnOutcome | None = fault.actual_outcome
    return {
        "offender": fault.offender.value,
        "reason": fault.reason.value,
        "attempted_move": _encode_move(move) if move is not None else None,
        "claimed_outcome": claimed.value if claimed is not None else None,
        "actual_outcome": actual.value if actual is not None else None,
    }


def _decode_fault(data: Any) -> PlayerFaultDetail:
    move: Any = data["attempted_move"]
    claimed: Any = data["claimed_outcome"]
    actual: Any = data["actual_outcome"]
    return PlayerFaultDetail(
        offender=Side(data["offender"]),
        reason=PlayerFaultReason(data["reason"]),
        attempted_move=_decode_move(move) if move is not None else None,
        claimed_outcome=TurnOutcome(claimed) if claimed is not None else None,
        actual_outcome=TurnOutcome(actual) if actual is not None else None,
    )


def _encode_move(move: Move) -> list[str]:
    return [cell.label for cell in move.cells]


def _decode_move(labels: Any) -> Move:
    return Move.from_labels(*labels)
