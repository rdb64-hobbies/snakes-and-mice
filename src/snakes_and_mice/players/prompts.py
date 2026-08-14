"""The natural-language contract the LLM player presents to a model.

Kept separate from the player mechanics in :mod:`.llm` because this prose churns
on its own schedule: it is the text we tune when experimenting with how to help
a model *follow the rules* (as opposed to play well). The mechanics only require
that a preamble exists and that every :class:`~snakes_and_mice.faults.PlayerFaultReason`
maps to some advice string — they are indifferent to the exact wording.

Note: the "Your response" paragraph of :data:`RULES_PREAMBLE` (including its
worked example) and the ``UNPARSEABLE_OUTPUT`` entry of :data:`FAULT_ADVICE`
both name the ``LLMMove`` fields defined in :mod:`.llm`; keep all of them in
sync when the fields change.
"""

from __future__ import annotations

from ..faults import PlayerFaultReason

# The rules preamble, enqueued once (lazily) ahead of the first move request. It
# explains the game and the response protocol; everything else the model learns
# it must infer from the opponent moves it is told about.
RULES_PREAMBLE: str = """\
You are playing Snakes and Mice, a two-player game on a 5x5 board.

The board. Rows are labeled A (top) to E (bottom); columns 1 (left) to 5 \
(right). A cell is a row letter then a column digit, e.g. C3. The valid cells \
are A1 through E5.

Sides. One player is the mouse, the other the snake. The mouse moves first. \
Before the first move the snake already occupies cell B3 (it is seeded there, \
which is a starting position, not the snake's move); every other cell starts \
empty.

A turn. On your turn you place TWO of your own pieces on two different empty \
cells — UNLESS a single piece already ends the game (completes a line for you, \
or leaves every line dead), in which case you may place just ONE. Pieces are \
placed in the order you list them and the position is checked after each, so a \
line completed by your first piece wins before the second is placed.

Winning. There are 12 lines: the 5 rows, the 5 columns, and the 2 main \
diagonals. You WIN the instant a line is fully occupied by your own pieces.

Cat's game. If every one of the 12 lines contains at least one piece from each \
side, no line can ever be completed: the game is a draw (a "cat's game").

Illegal moves lose the game immediately — there are no second chances within a \
game. A move is illegal if it: names a cell off the board; repeats a cell; \
places the wrong number of pieces (not one or two); plays on an already-occupied \
cell; plays a single piece that does not end the game; or misreports the outcome.

What you see. You are told only your opponent's moves, as they happen. You must \
track the full board yourself from the seeded snake at B3, your own moves, and \
your opponent's.

Your response. Each turn, return the structured fields: move_rationale (a short \
justification), cells (one or two labels like ["C3","D4"]), and claimed_outcome \
(exactly one of in_play, win, or cats_game — your honest assessment of the \
position after your move). For example, a valid response sets move_rationale to \
"take the center and extend toward a diagonal", cells to ["C3","D4"], and \
claimed_outcome to "in_play". Return all three fields — with exactly those \
names — every turn, and be sure to actually emit them rather than ending your \
response while still reasoning."""


# How to explain each fault back to the model, so the next game's opening can
# tell it what it did wrong and how to avoid repeating it (§11, end_game).
FAULT_ADVICE: dict[PlayerFaultReason, str] = {
    PlayerFaultReason.OFF_BOARD: (
        "you named a cell that is off the board — cells range from A1 to E5 "
        "(rows A–E, columns 1–5)."
    ),
    PlayerFaultReason.DUPLICATE_CELLS: (
        "you named the same cell twice — your two cells must be different."
    ),
    PlayerFaultReason.WRONG_PIECE_COUNT: (
        "you placed the wrong number of pieces — place exactly two cells, unless "
        "a single cell already wins or completes a cat's game."
    ),
    PlayerFaultReason.CELL_NOT_EMPTY: (
        "you played on a cell that was already occupied — only empty cells may be "
        "played, so track every piece already on the board."
    ),
    PlayerFaultReason.UNPARSEABLE_OUTPUT: (
        "your response could not be read as a move — you must emit the move as "
        "the structured output, not stop after reasoning. Return exactly the "
        "fields move_rationale, cells (one or two labels like [\"C3\",\"D4\"]), "
        "and claimed_outcome (one of in_play, win, or cats_game)."
    ),
    PlayerFaultReason.THINKING_LIMIT_EXCEEDED: (
        "you ran out of output tokens while still thinking and never produced "
        "your move — think more briefly and commit to your cells sooner, keeping "
        "your reasoning well within the limit."
    ),
    PlayerFaultReason.WRONG_OUTCOME_CLAIM: (
        "you misjudged the outcome of your own move — assess win, cats_game, or "
        "in_play carefully before committing each turn."
    ),
}
