"""Tests for the command-line entry point.

The presentation layer it drives is covered in test_console; here we check only
what is specific to ``main`` — that it wires up and runs a match without a
network, and quiets the per-request HTTP logging that would otherwise clutter
the board. (Rendering assertions live in test_console.)
"""

from __future__ import annotations

import logging

import pytest

from snakes_and_mice.cli import main


def test_main_quiets_http_request_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The per-request httpx INFO logs must not bleed into the board rendering.
    logging.getLogger("httpx").setLevel(logging.INFO)
    main(["--watch", "match"])  # two random players, no network
    assert logging.getLogger("httpx").level == logging.WARNING
