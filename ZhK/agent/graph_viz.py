"""Graph visualization using pyvis for the green-red game."""

import math

from pyvis.network import Network


def render_graph(state: dict, mutated_vertex: int | None = None) -> str:
    """Render the current game state as an interactive HTML graph.

    Style: white background, clean circles, black arrows.
    - Green mutable vertices: light green fill
    - Red mutable vertices: light red/pink fill
    - Frozen vertices: light blue fill

    Args:
        state: Dict from engine.get_state() or engine.mutate().
        mutated_vertex: Vertex that was just mutated (highlighted), or None.

    Returns:
        HTML string for embedding in Streamlit.
    """
    n = state["total_mutable"]
    colors = state["colors"]
    edges = state["edges"]

    net = Network(
        height="500px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black",
    )

    net.set_options("""{
        "physics": {
            "enabled": false
        },
        "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
            "color": { "color": "#333333" },
            "smooth": { "type": "curvedCW", "roundness": 0.12 },
            "width": 1.5
        },
        "nodes": {
            "font": { "size": 14, "color": "black", "face": "arial" },
            "borderWidth": 1.5
        },
        "interaction": {
            "dragNodes": true,
            "zoomView": true
        }
    }""")

    radius_inner = 160
    radius_outer = 280

    # Mutable vertices
    for i in range(1, n + 1):
        angle = 2 * math.pi * (i - 1) / n - math.pi / 2
        x = radius_inner * math.cos(angle)
        y = radius_inner * math.sin(angle)

        color = colors[i]
        if color == "red":
            bg = "#f4a0a0"  # light red/pink
        else:
            bg = "#a0d8a0"  # light green

        border = "#333333"
        border_width = 3 if i == mutated_vertex else 1.5

        net.add_node(
            i,
            label=str(i),
            x=x,
            y=y,
            color={"background": bg, "border": border},
            borderWidth=border_width,
            size=25,
            shape="circle",
        )

    # Frozen vertices
    for i in range(1, n + 1):
        fv = n + i
        angle = 2 * math.pi * (i - 1) / n - math.pi / 2
        x = radius_outer * math.cos(angle)
        y = radius_outer * math.sin(angle)

        net.add_node(
            fv,
            label=f"{n + i}",
            x=x,
            y=y,
            color={"background": "#a0c4e8", "border": "#333333"},
            borderWidth=1.5,
            size=22,
            shape="circle",
        )

    # Edges
    for src, dst, count in edges:
        label = f"x{count}" if count > 1 else ""
        width = 1.2 + count * 0.5
        net.add_edge(src, dst, label=label, width=width, color="#333333")

    return net.generate_html()
