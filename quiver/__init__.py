"""
Ice quivers, mutation, framing, and I/O. Import from here for a flat API:

    from quiver import IceQuiver, mutate_ice_quiver, frame_quiver, load_quiver
"""

from __future__ import annotations

from .core import Arrow, ArrowCounts, IceQuiver, ice_quiver_from_exmat
from .io import (
    format_edgelist,
    format_exchange_matrix,
    load_quiver,
    parse_exchange_matrix,
    parse_quiver_edgelist,
    parse_quiver_json,
)
from .vertex_color import is_green_framed, is_red_framed, VColor, vertex_color
from .operations import (
    coframe_quiver,
    frame_quiver,
    mutate_ice_quiver,
    random_quiver,
)

__all__ = [
    "Arrow",
    "ArrowCounts",
    "IceQuiver",
    "coframe_quiver",
    "format_edgelist",
    "format_exchange_matrix",
    "frame_quiver",
    "framed_layout_n",
    "ice_quiver_from_exmat",
    "is_green_framed",
    "is_red_framed",
    "VColor",
    "vertex_color",
    "load_quiver",
    "mutate_ice_quiver",
    "parse_exchange_matrix",
    "parse_quiver_edgelist",
    "parse_quiver_json",
    "random_quiver",
]
