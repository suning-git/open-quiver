"""
Tkinter front-end for quiver editing and mutation (see Docs/GuiProposal.md).

Implementation is split across this package; start the UI with run_app() or main().
"""

from __future__ import annotations

__all__ = ["main", "run_app"]


def run_app() -> None:
    """Create the root window and enter the Tk main loop."""
    from gui.app import QuiverMutationApp

    app = QuiverMutationApp()
    app.run()


def main() -> int:
    """CLI-style entry: run the GUI and return an exit code."""
    run_app()
    return 0
