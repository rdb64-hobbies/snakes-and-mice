"""Text CLI: board rendering, a result summary, and a playable/watchable game."""

from __future__ import annotations

import argparse
from pathlib import Path

from .board import Board
from .config import (
    ConfigError,
    Roster,
    load_environment,
    load_roster,
    make_llm_player,
)
from .core import BOARD_SIZE, Cell, Move, Side, TurnOutcome
from .match import play_match
from .observer import ObservationLevel, Observer
from .players import HumanPlayer, Player, RandomPlayer
from .result import GameResult, MatchResult, PlayerFaultDetail, Termination

# The piece glyphs are emoji, which occupy TWO display columns in a terminal.
# Every rendered cell is therefore normalized to a two-column token so the board
# lines up: emoji are left as-is, and the single-column empty marker is padded
# with a trailing space. Header labels and the row-label margin are sized to
# match (two columns each, joined by a single-space gutter).
GLYPH: dict[Side, str] = {Side.MOUSE: "🐭", Side.SNAKE: "🐍"}
EMPTY: str = "·"
_CELL_WIDTH: int = 2


def _cell_token(board: Board, cell: Cell) -> str:
    """A two-display-column token for one cell: the piece emoji, or a padded dot."""
    occupant: Side | None = board.occupant(cell)
    if occupant is not None:
        return GLYPH[occupant]
    return EMPTY.ljust(_CELL_WIDTH)


def render_board(board: Board) -> str:
    """Render the board as text, row ``A`` at the top and column ``1`` at left."""
    margin: str = " " * _CELL_WIDTH  # sits above the "<row> " row-label column
    header: str = margin + " ".join(
        str(c + 1).ljust(_CELL_WIDTH) for c in range(BOARD_SIZE)
    )
    rows: list[str] = [header]
    for r in range(BOARD_SIZE):
        row_label: str = chr(ord("A") + r)
        cells: list[str] = [
            _cell_token(board, Cell(r, c)) for c in range(BOARD_SIZE)
        ]
        rows.append(f"{row_label} " + " ".join(cells))
    return "\n".join(rows)


def describe_result(result: GameResult, players: dict[Side, str]) -> str:
    """A one-line human-readable summary of a game result."""
    if result.termination is Termination.LINE_COMPLETED:
        assert result.winner is not None
        return f"{players[result.winner]} ({result.winner.value}) wins."
    if result.termination is Termination.CATS_GAME:
        return "Cat's game — a draw."
    fault: PlayerFaultDetail | None = result.fault
    assert fault is not None
    detail: str = f"{players[fault.offender]} ({fault.offender.value}) faulted: {fault.reason.value}"
    if fault.attempted_move is not None:
        detail += f" on {fault.attempted_move}"
    if fault.claimed_outcome is not None and fault.actual_outcome is not None:
        detail += (
            f" (claimed {fault.claimed_outcome.value}, "
            f"actually {fault.actual_outcome.value})"
        )
    return detail


def describe_match_result(result: MatchResult) -> str:
    """A multi-line human-readable summary of a match's tallies."""
    names: dict[Side, str] = result.names
    lines: list[str] = [
        f"Match complete — {result.num_games} "
        f"{'game' if result.num_games == 1 else 'games'}",
        f"  🐭 {names[Side.MOUSE]} (mouse): {result.mouse_wins} "
        f"{'win' if result.mouse_wins == 1 else 'wins'}",
        f"  🐍 {names[Side.SNAKE]} (snake): {result.snake_wins} "
        f"{'win' if result.snake_wins == 1 else 'wins'}",
        f"  Cat's games: {result.cats_games}",
    ]
    if result.faults:
        lines.append(
            f"  Faults: {result.mouse_faults} mouse, {result.snake_faults} snake"
        )
    return "\n".join(lines)


class ConsoleObserver(Observer):
    """Renders a match to stdout, showing as much as its level asks for.

    The engine fires every hook; this observer gates its own output against the
    :class:`~snakes_and_mice.observer.ObservationLevel` it was built with —
    ``move`` shows every turn, ``game`` just each game's board and result, and
    ``match`` only the opening banner and the closing tally. Per-game
    bookkeeping (the game counter, the turn counter) is kept up to date at every
    level, so it is correct whenever a finer level does render. It never pauses:
    with a human player, that player's own input paces the game; otherwise play
    runs straight through.
    """

    def __init__(self, level: ObservationLevel = ObservationLevel.MOVE) -> None:
        super().__init__(level)
        self._names: dict[Side, str] = {}
        self._num_games: int = 1
        self._game: int = 0
        self._turn: int = 0

    def on_match_start(self, names: dict[Side, str], num_games: int) -> None:
        self._names = names
        self._num_games = num_games
        print("Snakes and Mice\n")
        print(f"🐭 Mouse: {names[Side.MOUSE]}    🐍 Snake: {names[Side.SNAKE]}")
        if num_games > 1:
            print(f"Match: {num_games} games")

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        self._names = names
        self._game += 1
        self._turn = 0
        if self.level < ObservationLevel.GAME:
            return
        if self._num_games > 1:
            print(f"\n=== Game {self._game} of {self._num_games} ===")
        # The starting board is only worth showing when the moves that follow
        # will update it — i.e. at MOVE level. At GAME level it would be a lone
        # seeded board no one watches change, so skip it.
        if self.level >= ObservationLevel.MOVE:
            print()
            print(render_board(board))

    def on_move_start(self, side: Side, board: Board) -> None:
        if self.level < ObservationLevel.MOVE:
            return
        self._turn += 1
        # Printed before the move is produced, so a slow player (e.g. an LLM)
        # visibly "thinks" here before on_move_end reports what it played.
        print(f"\nTurn {self._turn} — {self._names[side]} ({side.value}) to move…")

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        if self.level < ObservationLevel.MOVE:
            return
        print(f"  plays {move}:\n")
        print(render_board(board))

    def on_game_end(self, result: GameResult) -> None:
        if self.level < ObservationLevel.GAME:
            return
        print(f"\n{describe_result(result, self._names)}")

    def on_match_end(self, result: MatchResult) -> None:
        if self._num_games > 1:
            print(f"\n{describe_match_result(result)}")


_RANDOM_NAME: dict[Side, str] = {Side.MOUSE: "Randy", Side.SNAKE: "Ransom"}
_DEFAULT_LOG_DIR: str = "llm-logs"


def _make_player(
    kind: str, side: Side, roster: Roster | None, log_dir: Path | None
) -> Player:
    """Build the player for one side. ``kind`` is ``random``, ``human``, or an
    LLM roster name (in which case ``roster`` must be loaded)."""
    if kind == "human":
        return HumanPlayer(name="You")
    if kind == "random":
        return RandomPlayer(name=_RANDOM_NAME[side])
    assert roster is not None  # a roster is loaded whenever an LLM name is used
    return make_llm_player(kind, roster, log_dir=log_dir)


def main(argv: list[str] | None = None) -> None:
    """Play or watch a match of one or more games.

    By default two random bots play a single game at ``move`` detail. Pass
    ``--games N`` for a longer match and ``--watch match|game|move`` to choose
    how much is shown. ``--mouse`` and ``--snake`` each name who plays that side:
    ``random``, ``human``, or an LLM roster name from ``players.yaml``. A human at
    the board always forces ``move`` detail (with a note), since a human must see
    every move to play it. ``--log-llm [DIR]`` dumps each LLM player's full raw
    message thread as JSON for debugging.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="snakes-and-mice",
        description="Play or watch a match of Snakes and Mice.",
    )
    parser.add_argument(
        "--mouse", default="random", metavar="WHO",
        help="who plays Mouse: random, human, or an LLM roster name (default: random)",
    )
    parser.add_argument(
        "--snake", default="random", metavar="WHO",
        help="who plays Snake: random, human, or an LLM roster name (default: random)",
    )
    parser.add_argument(
        "--games", type=int, default=1, metavar="N",
        help="number of games in the match (default: 1)",
    )
    parser.add_argument(
        "--watch", choices=["match", "game", "move"], default="move",
        help="how much to show: match, game, or every move (default: move)",
    )
    parser.add_argument(
        "--log-llm", nargs="?", const=_DEFAULT_LOG_DIR, default=None, metavar="DIR",
        help=(
            "dump each LLM player's full message thread as JSON for debugging "
            f"(default DIR: {_DEFAULT_LOG_DIR}/); no-op for random/human"
        ),
    )
    args: argparse.Namespace = parser.parse_args(argv)

    if args.games < 1:
        parser.error("--games must be at least 1")

    log_dir: Path | None = Path(args.log_llm) if args.log_llm is not None else None
    builtin: set[str] = {"random", "human"}
    needs_roster: bool = args.mouse not in builtin or args.snake not in builtin

    roster: Roster | None = None
    try:
        if needs_roster:
            load_environment()
            roster = load_roster()
        mouse: Player = _make_player(args.mouse, Side.MOUSE, roster, log_dir)
        snake: Player = _make_player(args.snake, Side.SNAKE, roster, log_dir)
    except ConfigError as exc:
        parser.error(str(exc))

    has_human: bool = "human" in (args.mouse, args.snake)
    level: ObservationLevel = ObservationLevel[args.watch.upper()]
    if has_human and level < ObservationLevel.MOVE:
        print("(a human is playing — showing every move)\n")
        level = ObservationLevel.MOVE
    play_match(mouse, snake, args.games, ConsoleObserver(level))
