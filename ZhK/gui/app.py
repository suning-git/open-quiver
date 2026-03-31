"""
Main Tk application: root window, toolbar placeholder, canvas placeholder.

Wire in canvas_view, state, and controls as those modules are added.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from gui.state import GuiState, Mode
from gui.canvas_view import CanvasView
from gui.toolbar import Toolbar


class QuiverMutationApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Quiver mutation")
        self.root.minsize(640, 480)
        self.state = GuiState()

        self._build_layout()

    def _build_layout(self) -> None:
        # toolbar
        # ttk.Label(self.toolbar.frame, text="Toolbar (stub)").pack(side=tk.LEFT)
        self.toolbar = Toolbar(self.root, self._on_mode_change, self._add_framing)

        # TODO: add_vertex, add_arrow, delete_vertex, freeze, mutation mode,
        #       random quiver, random mutation, add framing, undo
        self.main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_frame = ttk.Frame(self.main_pane)
        self.sidebar = ttk.Frame(self.main_pane, width=200)
        
        self.main_pane.add(self.canvas_frame, weight=3)
        self.main_pane.add(self.sidebar, weight=1)

        # TODO: replace with gui.canvas_view -- vertices as filled circles, labels inside;
        #       draggable vertices; arrow multiplicity at midpoint when > 1
        self.canvas_view = CanvasView(
            self.canvas_frame,
            on_click=self._on_canvas_click,
            on_drag_start=self._on_drag_start,
            on_drag_move=self._on_drag_move,
            on_drag_end=self._on_drag_end,
        )

        # TODO: mutation history list + undo
        ttk.Label(self.sidebar, text="Mutation history (stub)").pack(anchor=tk.W, padx=4, pady=4)
        self.history_list = tk.Listbox(self.sidebar, height=12)
        self.history_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    
    # --- toolbar ---
    def _on_mode_change(self, mode: Mode) -> None:
        self.state.set_mode(mode)
        self._clear_selection()
        self.state.arrow_start = None
        print(f"Switched mode to {mode}")
    
    # --- canvas ---
    def _on_canvas_click(self, x: int, y: int) -> None:
        mode = self.state.mode
        data = self.state.data

        vid = data.find_vertex_at(x, y)

        # Clear selection if click on empty.
        if vid is None:
            self._cancel_arrow_mode()
            self._clear_selection()

        if mode == Mode.ADD_VERTEX:
            if vid is None: # Ignore if click on existing vertices.
                vid = data.add_vertex(x, y)
                self.canvas_view.draw_vertex(vid, x, y)
                print(f"Added vertex {vid}")
        
        elif mode == Mode.NORMAL:
            if vid is not None:
                self._select_vertex(vid)
                print(f"Clicked vertex {vid}")
        
        elif mode == Mode.ADD_ARROW_SRC:
            if vid is not None:
                self._handle_arrow_src(vid)
        
        elif mode == Mode.ADD_ARROW_DST:
            if vid is not None:
                self._handle_arrow_dst(vid)

        elif mode == Mode.DELETE_VERTEX:
            if vid is not None:
                self._delete_vertex(vid)
                print(f"Deleted vertex {vid}")

        elif mode == Mode.FREEZE:
            if vid is not None:
                self._freeze_vertex(vid)
                print(f"Froze vertex {vid}")

        elif mode == Mode.MUTATE:
            if vid is not None:
                self._mutate_vertex(vid)
                print(f"Mutated vertex {vid}")
        
        elif mode == Mode.ADD_FRAMING:
            self._add_framing()
            print("Added framing")


    def _redraw_all(self) -> None:
        # Clear canvas
        self.canvas_view.clear_all()

        # redraw vertex
        for vid, (x, y) in self.state.data.positions.items():
            self.canvas_view.draw_vertex(vid, x, y)

        # redraw arrows
        arrows = self.state.data.get_arrow_draw_data()
        self.canvas_view.redraw_arrows(arrows)

        # redraw color
        self._refresh_colors()


    def _on_drag_start(self, x: int, y: int) -> None:
        # Disable dragging in arrow mode.
        if self.state.mode in (Mode.ADD_ARROW_SRC, Mode.ADD_ARROW_DST):
            return

        vid = self.state.data.find_vertex_at(x, y)

        if vid is not None:
            self.state.dragging_vertex = vid
            self._select_vertex(vid)
    
    def _on_drag_move(self, x: int, y: int) -> None:
        vid = self.state.dragging_vertex

        if vid is None:
            return

        # Update position in state
        self.state.data.positions[vid] = (x, y)

        # Update vertices and arrows in canvas
        self.canvas_view.move_vertex(vid, x, y)
        arrows = self.state.data.get_arrow_draw_data()
        self.canvas_view.redraw_arrows(arrows)
    
    def _on_drag_end(self, x: int, y: int) -> None:
        self.state.dragging_vertex = None


    def _select_vertex(self, vid: int) -> None:
        old = self.state.selected_vertex

        if old is not None:
            self.canvas_view.clear_highlight(old)

        self.state.selected_vertex = vid
        self.canvas_view.highlight_vertex(vid)

    def _clear_selection(self) -> None:
        old = self.state.selected_vertex

        if old is not None:
            self.canvas_view.clear_highlight(old)

        self.state.selected_vertex = None

    def _handle_arrow_src(self, vid: int) -> None:
        self.state.arrow_start = vid

        self._select_vertex(vid)

        # Switch to destination mode
        self.state.set_mode(Mode.ADD_ARROW_DST)

    def _handle_arrow_dst(self, vid: int) -> None:
        start = self.state.arrow_start
        end = vid

        if start is None: # Should not happen, but be safe.
            self.state.set_mode(Mode.ADD_ARROW_SRC)
            return

        if start == end:
            return  # Loops are not allowed.

        # Add arrow
        self.state.data.add_arrow(start, end)
        arrows = self.state.data.get_arrow_draw_data()
        self.canvas_view.redraw_arrows(arrows)
        self._refresh_colors()

        # Reset to SRC mode for next arrow
        self.state.arrow_start = None
        self._clear_selection()

        self.state.set_mode(Mode.ADD_ARROW_SRC)
    
    def _cancel_arrow_mode(self) -> None:
        if self.state.mode in (Mode.ADD_ARROW_SRC, Mode.ADD_ARROW_DST):
            self.state.arrow_start = None
            self.state.set_mode(Mode.ADD_ARROW_SRC)

    
    def _delete_vertex(self, vid: int) -> None:
        # Clear states
        self._clear_selection()
        self.state.dragging_vertex = None
        self.state.arrow_start = None

        self.state.data.remove_vertex(vid)

        # Redrawing
        self._redraw_all()


    def _freeze_vertex(self, vid: int) -> None:
        self.state.arrow_start = None
        self.state.data.freeze_vertex(vid)

        self._redraw_all()
    
    def _refresh_colors(self) -> None:
        self.canvas_view.recolor_all_vertices(
            self.state.data.positions,
            self.state.data.quiver
        )
    
    def _mutate_vertex(self, vid: int) -> None:
        self._clear_selection()
        self.state.arrow_start = None

        self.state.data.mutate_vertex(vid)

        self._redraw_all()


    def run(self) -> None:
        self.root.mainloop()
    
    def _add_framing(self) -> None:
        self._clear_selection()
        self.state.arrow_start = None
        self.state.data.add_framing()

        self._redraw_all()


def main() -> int:
    QuiverMutationApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
