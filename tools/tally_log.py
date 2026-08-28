"""What happened in the games recorded in a `--log-llm` message thread?

`tally-tournament` reads exact `MatchResult` records (§6), but a match only writes one
when asked with `--tournament-results`. A `--log-llm` dump is always written, and it
does contain every game's outcome — the LLM player tells its model how each game ended
(§4, "The message thread"), so the thread carries a prose record of the whole match.

This reads that record back. It does not guess at the prose: it inverts the exact
constants the player writes (`players/prompts.py`), so a reworded message changes both
sides at once. Hand-written substrings are the trap here — the fault advice for
`WRONG_PIECE_COUNT` and `WRONG_OUTCOME_CLAIM` both mention a cat's game, so a reader
that tests for "cat" before testing for a fault silently scores those faults as draws.

    uv run python tools/tally_log.py llm-logs/*.json

**It reports one game fewer than the match played.** The player composes each game's
outcome message but only *enqueues* it, to be flushed with the next game's first
request (§4) — so the final game's outcome is never sent and never reaches the log.
That is a property of the deferred-feedback design, not of this reader. Where the exact
count matters, run the match with `--tournament-results` and use `tally-tournament`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Final

from snakes_and_mice.faults import PlayerFaultReason
from snakes_and_mice.players.prompts import (
    FAULT_ADVICE,
    GAME_ABORTED,
    GAME_DRAWN,
    GAME_LOST,
    GAME_WON,
    OPPONENT_FAULT_PREFIX,
    OWN_FAULT_PREFIX,
)

# Outcomes as this reader reports them, from the logged player's point of view.
WON: Final[str] = "won"
LOST: Final[str] = "lost"
DREW: Final[str] = "drew"
ABORTED: Final[str] = "aborted"
OPPONENT_FAULTED: Final[str] = "opponent faulted"
UNRECOGNIZED: Final[str] = "unrecognized"


def classify(block: str) -> tuple[str, PlayerFaultReason | None]:
    """One game-over message → an outcome, and the fault reason where there is one.

    Faults are tested **before** the plain outcomes: a fault message embeds advice
    text that can mention any of them, so the other order misreads it.
    """
    text: str = block.strip()
    if text.startswith(GAME_ABORTED[:40]):
        return ABORTED, None
    if text.startswith(OWN_FAULT_PREFIX):
        for reason, advice in FAULT_ADVICE.items():
            if advice in text:
                return "faulted", reason
        return "faulted", None
    if text.startswith(OPPONENT_FAULT_PREFIX):
        return OPPONENT_FAULTED, None
    if text.startswith(GAME_WON):
        return WON, None
    if text.startswith(GAME_LOST):
        return LOST, None
    if text.startswith(GAME_DRAWN):
        return DREW, None
    return UNRECOGNIZED, None


def outcomes(path: Path) -> list[tuple[str, PlayerFaultReason | None]]:
    """Every game outcome recorded in one thread, in order.

    The player joins its queued messages into one user turn with blank lines between
    them, so each message is recovered by splitting the turn back on those.
    """
    thread: list[dict[str, object]] = json.loads(path.read_text())
    found: list[tuple[str, PlayerFaultReason | None]] = []
    for message in thread:
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("part_kind") != "user-prompt":
                continue
            for block in str(part.get("content", "")).split("\n\n"):
                outcome, reason = classify(block)
                if outcome is not UNRECOGNIZED:
                    found.append((outcome, reason))
    return found


def report(path: Path) -> None:
    """Print one thread's tally."""
    found = outcomes(path)
    if not found:
        print(f"{path.name}: no completed games recorded")
        return
    tally: Counter[str] = Counter(outcome for outcome, _ in found)
    reasons: Counter[str] = Counter(
        reason.name.lower() for _, reason in found if reason is not None
    )
    n: int = len(found)
    scored: int = n - tally[ABORTED]

    def pct(count: int) -> str:
        return f"{count / scored:.0%}" if scored else "—"

    print(f"{path.name}  —  {n} games recorded (the match played one more)")
    for label in (WON, DREW, LOST, "faulted", OPPONENT_FAULTED):
        if tally[label]:
            print(f"    {label:16s} {tally[label]:3d}  {pct(tally[label])}")
    if tally[ABORTED]:
        print(f"    {'aborted':16s} {tally[ABORTED]:3d}  (no contest, unscored)")
    if reasons:
        detail: str = ", ".join(f"{k} ×{v}" for k, v in sorted(reasons.items()))
        print(f"    own faults: {detail}")


def main() -> None:
    paths: list[Path] = [Path(a) for a in sys.argv[1:]]
    if not paths:
        raise SystemExit(
            "usage: tally_log.py LOG.json [LOG.json ...]\n"
            "  reads --log-llm message threads and reports each match's outcomes"
        )
    for path in paths:
        if not path.exists():
            print(f"{path}: not found")
            continue
        report(path)


if __name__ == "__main__":
    main()
