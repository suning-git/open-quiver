from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable
from gui.state import Mode


class Toolbar:
    def __init__(self, parent, on_mode_change: Callable[[Mode], None]) -> None:
        self.frame = ttk.Frame(parent, padding=(4, 4))
        self.frame.pack(side=tk.TOP, fill=tk.X)

        self.on_mode_change = on_mode_change

        self.mode_var = tk.StringVar(value=Mode.NORMAL.name)

        self._build()

    def _build(self) -> None:
        ttk.Radiobutton(
            self.frame,
            text="Normal",
            value=Mode.NORMAL.name,
            variable=self.mode_var,
            command=self._handle_change,
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            self.frame,
            text="Add Vertex",
            value=Mode.ADD_VERTEX.name,
            variable=self.mode_var,
            command=self._handle_change,
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            self.frame,
            text="Add Arrow",
            value=Mode.ADD_ARROW_SRC.name,
            variable=self.mode_var,
            command=self._handle_change,
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            self.frame,
            text="Delete Vertex",
            value=Mode.DELETE_VERTEX.name,
            variable=self.mode_var,
            command=self._handle_change,
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            self.frame,
            text="Freeze",
            value=Mode.FREEZE.name,
            variable=self.mode_var,
            command=self._handle_change,
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            self.frame,
            text="Mutate",
            value=Mode.MUTATE.name,
            variable=self.mode_var,
            command=self._handle_change,
        ).pack(side=tk.LEFT)

        self.mode_label = ttk.Label(self.frame, text="Mode: NORMAL")
        self.mode_label.pack(side=tk.RIGHT)

    def _handle_change(self) -> None:
        mode = Mode[self.mode_var.get()]
        self.mode_label.config(text=f"Mode: {mode.name}")
        self.on_mode_change(mode)