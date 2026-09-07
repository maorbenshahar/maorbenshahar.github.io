# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Draw two connected libraries in the same elementary-construction graph.

Run: uv run assets/autoprover/connected_libraries.py

Each successful addition costs one unit and previously known nodes are retained.
The breadth-first checks compute shortest construction lengths in tiny finite
graphs. They do not measure the effort of a prover searching for those routes.
"""

from collections import deque
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch


OUTPUT = Path(__file__).resolve().parent
BLUE = "#246a93"
GRAY = "#a7b1bc"
LIGHT_GRAY = "#f4f6f8"
ORANGE = "#c76d1c"
INK = "#26323c"


def branch_graph(length):
    """Two unary chains of equal length with the shared starting node c."""
    rules = []
    for branch in ("b", "u"):
        previous = "c"
        for index in range(1, length + 1):
            node = f"{branch}{index}"
            rules.append((frozenset({previous}), node))
            previous = node
    return rules


def known_prefix(branch, length):
    return frozenset({"c"} | {f"{branch}{index}" for index in range(1, length + 1)})


def shortest_additions(initial, target, rules):
    """BFS over retained-library sets, enumerating every enabled fresh addition."""
    queue = deque([(frozenset(initial), 0)])
    visited = {frozenset(initial)}
    while queue:
        known, count = queue.popleft()
        if target in known:
            return count
        for premises, conclusion in rules:
            if premises <= known and conclusion not in known:
                following = known | {conclusion}
                if following not in visited:
                    visited.add(following)
                    queue.append((following, count + 1))
    raise ValueError("Target is not derivable from the initial library")


def is_connected(known, rules):
    """Connectivity of the induced undirected dependency graph."""
    reached = {next(iter(known))}
    while True:
        before = len(reached)
        for premises, conclusion in rules:
            endpoints = premises | {conclusion}
            if endpoints <= known and endpoints & reached:
                reached.update(endpoints)
        if len(reached) == before:
            return reached == known


def check_distances():
    cases = 0
    for length in range(1, 6):
        rules = branch_graph(length)
        target = f"b{length}"
        for prefix_length in range(length):
            near = known_prefix("b", prefix_length)
            other = known_prefix("u", prefix_length)
            assert len(near) == len(other) == prefix_length + 1
            assert is_connected(near, rules)
            assert is_connected(other, rules)
            assert shortest_additions(near, target, rules) == length - prefix_length
            assert shortest_additions(other, target, rules) == length
            cases += 2
    print(f"Passed {cases} BFS distance checks for connected prefixes (L=1..5, 0<=m<L).")


def draw_panel(ax, known, title, remaining):
    positions = {"c": (0.0, 0.0)}
    positions.update({f"b{index}": (float(index), 0.72) for index in range(1, 5)})
    positions.update({f"u{index}": (float(index), -0.72) for index in range(1, 5)})
    radius = 0.25

    for premises, conclusion in branch_graph(4):
        premise = next(iter(premises))
        start = positions[premise]
        end = positions[conclusion]
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = math.hypot(dx, dy)
        offset = (radius * dx / norm, radius * dy / norm)
        start = (start[0] + offset[0], start[1] + offset[1])
        end = (end[0] - offset[0], end[1] - offset[1])
        color = BLUE if premise in known and conclusion in known else GRAY
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.8, color=color, shrinkA=2, shrinkB=3, zorder=1,
        ))

    for node, position in positions.items():
        if node in known:
            fill, outline, text_color = BLUE, BLUE, "white"
        elif node == "b4":
            fill, outline, text_color = "white", ORANGE, ORANGE
        else:
            fill, outline, text_color = LIGHT_GRAY, GRAY, INK
        ax.add_patch(Circle(
            position, radius, facecolor=fill, edgecolor=outline,
            linewidth=2 if node == "b4" else 1.6, zorder=2,
        ))
        label = "$G$" if node == "b4" else (
            "$c$" if node == "c" else rf"${node[0]}_{node[1:]}$"
        )
        ax.text(*position, label, color=text_color, fontsize=13,
                ha="center", va="center", zorder=3)

    ax.set_title(title, fontsize=13, color=INK, pad=10)
    ax.text(2.0, -1.32,
            f"4 known nodes; shortest route: {remaining} step{'s' if remaining != 1 else ''}",
            fontsize=11.5, color=INK, ha="center", va="center")
    ax.set(xlim=(-0.5, 4.5), ylim=(-1.58, 1.18), aspect="equal")
    ax.set_axis_off()


def make_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "svg.fonttype": "none", "svg.hashsalt": "autoprover-connected-libraries",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.20, top=0.79, wspace=0.10)
    fig.suptitle("Same dependencies, different known prefixes", fontsize=16,
                 color=INK, y=0.965)
    draw_panel(axes[0], known_prefix("b", 3), r"Known: $c, b_1, b_2, b_3$", 1)
    draw_panel(axes[1], known_prefix("u", 3), r"Known: $c, u_1, u_2, u_3$", 4)

    handles = [
        Line2D([], [], marker="o", markersize=10, linestyle="none",
               markerfacecolor=BLUE, markeredgecolor=BLUE, label="Known"),
        Line2D([], [], marker="o", markersize=10, linestyle="none",
               markerfacecolor=LIGHT_GRAY, markeredgecolor=GRAY, label="Not yet known"),
        Line2D([], [], marker="o", markersize=10, linestyle="none",
               markerfacecolor="white", markeredgecolor=ORANGE,
               markeredgewidth=1.8, label=r"Target $G=b_4$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=3, frameon=False, fontsize=11, columnspacing=2.2,
               handletextpad=0.55)

    svg = OUTPUT / "connected-libraries.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    png = OUTPUT / "connected-libraries.png"
    fig.savefig(png, dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(f"Wrote {svg} and {png}")


if __name__ == "__main__":
    check_distances()
    make_figure()
