from __future__ import annotations

import math
import tkinter as tk
from typing import Callable

from quiver import VColor, vertex_color
from gui.constants import VERTEX_RADIUS

VCOLOR_TO_FILL = {
    VColor.DEFAULT: "white",
    VColor.GREEN: "#90EE90",
    VColor.RED: "#FF7F7F",
    VColor.YELLOW: "#EEEE7F",
    VColor.FROZEN: "#ADD8E6",
}


class CanvasView:
    def __init__(
            self,
            parent,
            on_click: Callable[[int, int], None],
            on_drag_start: Callable[[int, int], None],
            on_drag_move: Callable[[int, int], None],
            on_drag_end: Callable[[int, int], None],
        ) -> None:
        self.canvas = tk.Canvas(parent, background="#f4f4f4", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.on_click = on_click
        self.on_drag_start = on_drag_start
        self.on_drag_move = on_drag_move
        self.on_drag_end = on_drag_end

        self.canvas.bind("<Button-1>", self._handle_click)
        self.canvas.bind("<B1-Motion>", self._handle_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._handle_drag_end)


    def _handle_click(self, event) -> None:
        self.on_click(event.x, event.y)
        self.on_drag_start(event.x, event.y)

    def _handle_drag_move(self, event) -> None:
        self.on_drag_move(event.x, event.y)

    def _handle_drag_end(self, event) -> None:
        self.on_drag_end(event.x, event.y)


    def draw_vertex(self, vid: int, x: float, y: float, frozen: bool = False) -> None:
        r = VERTEX_RADIUS

        fill_color = "#ADD8E6" if frozen else "white"

        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill=fill_color,
            outline="black",
            width=2,
            tags=f"vertex_{vid}"
        )

        self.canvas.create_text(
            x, y,
            text=str(vid),
            font=("Arial", 16, "bold"), 
            tags=f"vertex_label_{vid}"
        )
    
    def highlight_vertex(self, vid: int) -> None:
        self.canvas.itemconfig(f"vertex_{vid}", outline="red", width=3)

    def clear_highlight(self, vid: int) -> None:
        self.canvas.itemconfig(f"vertex_{vid}", outline="black", width=2)

    
    def move_vertex(self, vid: int, x: float, y: float) -> None:
        """
        Move vertex graphics to new position without redrawing.
        """
        r = VERTEX_RADIUS

        # Move circle
        self.canvas.coords(
            f"vertex_{vid}",
            x - r, y - r, x + r, y + r
        )

        # Move label
        self.canvas.coords(
            f"vertex_label_{vid}",
            x, y
        )

    def _arrow_endpoints(self, x1, y1, x2, y2):
        """
        Compute endpoints so that arrow touches circle boundaries instead of centers.
        """
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        if dist == 0:
            return x1, y1, x2, y2

        ux = dx / dist
        uy = dy / dist

        r = VERTEX_RADIUS

        return (
            x1 + ux * r,
            y1 + uy * r,
            x2 - ux * r,
            y2 - uy * r,
        )
    
    def draw_arrow(self, a: int, b: int, x1: float, y1: float, x2: float, y2: float, count: int = 1) -> None:
        """
        Draw a directed arrow from vertex a to vertex b.
        """
        x1, y1, x2, y2 = self._arrow_endpoints(x1, y1, x2, y2)
        self.canvas.create_line(
            x1, y1, x2, y2,
            arrow=tk.LAST,
            width=2,
            tags=("arrow", f"arrow_{a}_{b}")
        )

        if count > 1:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            text = str(count)

            # Create text (temporary to measure bbox)
            text_id = self.canvas.create_text(
                mx, my,
                text=text,
                font=("Arial", 12, "bold"),
                tags=("arrow_label", f"arrow_label_{a}_{b}")
            )

            # Get bounding box
            bbox = self.canvas.bbox(text_id)
            x0, y0, x1b, y1b = bbox

            pad = 2

            # Draw background rectangle
            rect_id = self.canvas.create_rectangle(
                x0 - pad, y0 - pad, x1b + pad, y1b + pad,
                fill="#F4F4F4",
                outline="",
                tags=("arrow_label_bg", f"arrow_label_bg_{a}_{b}")
            )

            # Raise text above rectangle
            self.canvas.tag_raise(text_id, rect_id)
    
    def redraw_arrows(self, arrows) -> None:
        """
        arrows: iterable of (a, b, x1, y1, x2, y2, count)
        """
        self.canvas.delete("arrow")
        self.canvas.delete("arrow_label")
        self.canvas.delete("arrow_label_bg")

        for a, b, x1, y1, x2, y2, count in arrows:
            self.draw_arrow(a, b, x1, y1, x2, y2, count)
    
    def clear_all(self) -> None:
        self.canvas.delete("all")

    # def set_vertex_frozen(self, vid: int, frozen: bool) -> None:
    #     fill_color = "#ADD8E6" if frozen else "white"
    #     self.canvas.itemconfig(f"vertex_{vid}", fill=fill_color)
    def recolor_vertex(self, vid: int, vcolor: VColor) -> None:
        fill = VCOLOR_TO_FILL[vcolor]
        self.canvas.itemconfig(f"vertex_{vid}", fill=fill)
    
    def recolor_all_vertices(self, positions, quiver) -> None:
        for vid in positions.keys():
            vcolor = vertex_color(quiver, vid)
            self.recolor_vertex(vid, vcolor)