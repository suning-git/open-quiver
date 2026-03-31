"""
Application state: current ice quiver, UI mode, layout positions, mutation undo stack.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
from enum import Enum, auto

from quiver import IceQuiver, mutate_ice_quiver
from gui.constants import VERTEX_RADIUS

class Mode(Enum):
    NORMAL = auto()
    ADD_VERTEX = auto()
    ADD_ARROW_SRC = auto()
    ADD_ARROW_DST = auto()
    DELETE_VERTEX = auto()
    FREEZE = auto()
    MUTATE = auto()


class IceQuiverWithPosition:
    def __init__(self) -> None:
        # TODO: connect to IceQuiver
        self.quiver = IceQuiver.empty()

        self.positions: Dict[int, Tuple[float, float]] = {}
        self._next_vertex_id: int = 0

    # Check if the position list coincides with vertex list.
    def _check_consistency(self) -> None:
        assert set(self.positions.keys()) == set(self.quiver.vertices)

    # --- vertex manipulations ---

    def add_vertex(self, x: float, y: float) -> int:
        new_quiver, vid = self.quiver.add_vertex()
        self.quiver = new_quiver
        self.positions[vid] = (x, y)

        return vid
    
    def find_vertex_at(self, x: float, y: float) -> Optional[int]:
        for vid, (vx, vy) in self.positions.items():
            dx = x - vx
            dy = y - vy

            if dx * dx + dy * dy <= VERTEX_RADIUS * VERTEX_RADIUS:
                return vid

        return None
    
    def add_arrow(self, a: int, b: int) -> None:
        """
        Add arrow a -> b to the quiver.
        """
        self.quiver = self.quiver.add_arrow(a, b)

    def get_arrow_draw_data(self):
        """
        Prepare arrow data for rendering.
        """
        result = []

        for (a, b), count in self.quiver.arrow_counts.items():
            x1, y1 = self.positions[a]
            x2, y2 = self.positions[b]

            result.append((a, b, x1, y1, x2, y2, count))

        return result
    
    def remove_vertex(self, vid: int) -> None:
        old_positions = self.positions

        # update IceQuiver first
        # old_vertices = list(self.quiver.vertices)
        self.quiver = self.quiver.remove_vertex(vid)

        def remap(x: int) -> int:
            if x < vid:
                return x
            elif x > vid:
                return x - 1
            else:
                raise ValueError

        # new positions
        new_positions: Dict[int, Tuple[float, float]] = {}

        for old_id, pos in old_positions.items():
            if old_id == vid:
                continue
            new_positions[remap(old_id)] = pos

        self.positions = new_positions

    def freeze_vertex(self, vid: int) -> None:
        if vid in self.quiver.frozen:
            return
        self.quiver = self.quiver.freeze_vertex(vid)

    def mutate_vertex(self, vid: int) -> None:
        if vid in self.quiver.frozen:
            return
        self.quiver = mutate_ice_quiver(self.quiver, vid)


class GuiState:
    def __init__(self) -> None:
        # vertex_id -> (x, y)
        self.data = IceQuiverWithPosition()
        self.mode: Mode = Mode.NORMAL

        self.selected_vertex: Optional[int] = None
        self.dragging_vertex: Optional[int] = None
        self.arrow_start: Optional[int] = None
    
    def set_mode(self, mode: Mode) -> None:
        self.mode = mode