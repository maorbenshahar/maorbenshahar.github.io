# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Calculate and plot the cost of guided depth-first construction search.

Run: uv run assets/autoprover/search_efficiency.py

The full b-ary tree has a known root and a target leaf at depth L. A supplied
guide retains every edge on the target path and independently retains each
other edge with probability q. Depth-first search orders children uniformly
and stops when it constructs the target. Every step adds a new verified node.
Its unit cost models selection, local use of the supplied guide, and verification.
The known root, navigation, and acquisition of the guide are excluded.
Every constructed node has an already constructed parent, and is retained.

These are exact model calculations, not measurements of an actual prover.
The checks enumerate the generative model and actual search histories, rather
than reimplementing the formula's expected-subtree-size derivation.
"""

from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np


OUTPUT = Path(__file__).resolve().parent
INK = "#26323c"
BLUE = "#23789f"
ORANGE = "#a9740b"
RED = "#b53a4b"
GRAY = "#87939e"


def mean_work(depth, retention, branching=2):
    """Exact for rational retention; stable positive summation for floats."""
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not isinstance(branching, int) or branching < 2:
        raise ValueError("branching must be an integer of at least two")
    if not 0 <= retention <= 1:
        raise ValueError("retention must lie between zero and one")
    power = 1
    weighted_sum = 0
    for level in range(depth):
        weighted_sum += (depth - level) * power
        power *= branching * retention
    return depth + Fraction(branching - 1, 2) * retention * weighted_sum


def limiting_efficiency(retention, branching=2):
    """Limit of L/W_L at fixed q, where W_L is the mean work."""
    if not 0 <= retention <= 1:
        raise ValueError("retention must lie between zero and one")
    if branching * retention >= 1:
        return 0.0
    return 1 / (1 + (branching - 1) * retention
                / (2 * (1 - branching * retention)))


def tree_nodes(depth, branching=2):
    """Heap labels: root 0, children b*v+1 through b*v+b."""
    return (branching ** (depth + 1) - 1) // (branching - 1)


def target_path(target, branching=2):
    """Each non-root node's label also identifies its incoming edge."""
    path = set()
    while target:
        path.add(target)
        target = (target - 1) // branching
    return frozenset(path)


def marked_trees(depth, target):
    """Enumerate all binary-tree edge marks conditional on a fixed target."""
    forced = target_path(target)
    other_edges = [node for node in range(1, tree_nodes(depth))
                   if node not in forced]
    for marks in product((False, True), repeat=len(other_edges)):
        retained = forced | {edge for edge, keep in zip(other_edges, marks) if keep}
        positives = sum(marks)
        yield frozenset(retained), positives, len(other_edges) - positives


def dfs_work(depth, retained, target, reverse_order):
    """Run DFS with explicit child orders; count actual first constructions."""
    total_nodes = tree_nodes(depth)
    stack = [0]
    known = set()
    work = 0
    while stack:
        node = stack.pop()
        if node:
            assert (node - 1) // 2 in known
            assert node not in known
            work += 1
        known.add(node)
        if node == target:
            return work
        children = [child for child in (2 * node + 1, 2 * node + 2)
                    if child < total_nodes and child in retained]
        if children and reverse_order[node]:
            children.reverse()
        stack.extend(reversed(children))
    raise AssertionError("The guide must retain the entire target path")


def enumerated_dfs_mean(depth, target, retention):
    """Average every marked tree and every child order using exact weights."""
    internal_nodes = tree_nodes(depth - 1)
    orders = list(product((False, True), repeat=internal_nodes))
    expectation = Fraction(0)
    probability = Fraction(0)
    histories = 0
    for retained, positives, negatives in marked_trees(depth, target):
        marking_probability = retention ** positives * (1 - retention) ** negatives
        probability += marking_probability
        conditional_total = 0
        for order in orders:
            conditional_total += dfs_work(depth, retained, target, order)
            histories += 1
        expectation += marking_probability * Fraction(conditional_total, len(orders))
    assert probability == 1
    return expectation, histories


def reachable_leaves(depth, retained):
    """Read candidate targets directly from the observed marked graph."""
    first_leaf = tree_nodes(depth - 1)
    reached = {0}
    leaves = set()
    for node in range(1, tree_nodes(depth)):
        if node in retained and (node - 1) // 2 in reached:
            reached.add(node)
            if node >= first_leaf:
                leaves.add(node)
    return leaves


def check_target_posterior(depth, retention):
    """Generate the joint distribution, then condition on each observed graph.

    The target starts uniformly among leaves. We do not assume the posterior
    formula: joint masses are accumulated from all targets and random marks,
    then compared with candidates obtained by a separate graph traversal.
    """
    leaves = range(tree_nodes(depth - 1), tree_nodes(depth))
    joint = defaultdict(lambda: defaultdict(Fraction))
    for target in leaves:
        for retained, positives, negatives in marked_trees(depth, target):
            mass = (retention ** positives * (1 - retention) ** negatives
                    / len(leaves))
            if mass:
                joint[retained][target] += mass
    total_mass = Fraction(0)
    for retained, target_masses in joint.items():
        candidates = reachable_leaves(depth, retained)
        mass = sum(target_masses.values(), Fraction(0))
        total_mass += mass
        assert set(target_masses) == candidates
        for target in candidates:
            assert target_masses[target] / mass == Fraction(1, len(candidates))
    assert total_mass == 1
    return len(joint)


def check_calculations():
    depth = 2
    target = tree_nodes(depth - 1)  # Fixed leftmost target leaf.
    for retention in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
        actual, histories = enumerated_dfs_mean(depth, target, retention)
        assert actual == mean_work(depth, retention)
        print(f"DFS enumeration: L=2, q={retention}, {histories} histories, "
              f"W_L={actual} ({float(actual):.6f}).")

    posterior_patterns = 0
    for depth in (1, 2):
        for retention in (Fraction(0), Fraction(1, 5), Fraction(1, 2),
                          Fraction(4, 5), Fraction(1)):
            posterior_patterns += check_target_posterior(depth, retention)
    print(f"Posterior check: {posterior_patterns} positive-probability marked "
          "graphs; target uniform over reachable candidate leaves.")

    critical_cases = 0
    for branching in (2, 3, 4):
        for depth in (1, 2, 8, 32, 128):
            critical = Fraction(1, branching)
            expected = depth + Fraction(branching - 1, 4 * branching) * depth * (depth + 1)
            assert mean_work(depth, critical, branching) == expected
            assert mean_work(depth, Fraction(0), branching) == depth
            complete_tree_mean = Fraction(tree_nodes(depth, branching) - 1 + depth, 2)
            assert mean_work(depth, Fraction(1), branching) == complete_tree_mean
            critical_cases += 1
    print(f"Passed {critical_cases} exact critical-sum and endpoint checks.")

    assert limiting_efficiency(0) == 1
    assert limiting_efficiency(0.5) == limiting_efficiency(1) == 0
    for retention in (0.05, 0.25, 0.45):
        observed = 10000 / mean_work(10000, retention)
        assert math.isclose(observed, limiting_efficiency(retention), rel_tol=0.002)
    print("Finite-size efficiencies agree with the subcritical limit.")


def make_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "svg.fonttype": "none", "svg.hashsalt": "autoprover-search-efficiency",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, (efficiency, work) = plt.subplots(1, 2, figsize=(12.4, 5.25))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.77, wspace=0.29)
    fig.suptitle("A threshold in search efficiency", fontsize=17, y=0.975)
    fig.text(0.5, 0.91,
             "The same binary construction tree, with different retention of off-route moves",
             ha="center", fontsize=11.5, color="#596773")

    retentions = np.linspace(0, 1, 501)
    for depth, color in ((8, BLUE), (32, ORANGE), (128, RED)):
        efficiency.plot(retentions,
                        [depth / mean_work(depth, float(q)) for q in retentions],
                        color=color, linewidth=2.3, label=rf"$L={depth}$")
    efficiency.plot(retentions, [limiting_efficiency(float(q)) for q in retentions],
                    color=INK, linestyle="--", linewidth=2.0,
                    label=r"Limit as $L\to\infty$")
    efficiency.axvline(0.5, color=GRAY, linestyle=":", linewidth=1.3, zorder=0)
    efficiency.text(0.52, 0.96, r"$q_c=1/2$", color="#677682", fontsize=11,
                    va="top")
    efficiency.set(title="Efficiency at increasing target depth",
                   xlabel=r"Off-route retention probability, $q$",
                   ylabel=r"Efficiency, $\eta_L=L/W_L$",
                   xlim=(0, 1), ylim=(-0.015, 1.025))
    efficiency.legend(loc="upper right", bbox_to_anchor=(1.0, 0.83),
                      frameon=False, fontsize=10)
    efficiency.grid(axis="y", alpha=0.17)

    depths = range(1, 65)
    for retention, color, label in (
        (0.25, BLUE, r"$q=0.25$: linear"),
        (0.5, ORANGE, r"$q=0.50$: quadratic"),
        (0.75, RED, r"$q=0.75$: exponential"),
    ):
        work.plot(list(depths), [mean_work(depth, retention) for depth in depths],
                  color=color, linewidth=2.4, label=label)
    work.set(title="Expected work on either side of the threshold",
             xlabel=r"Target depth, $L$", ylabel=r"Mean work, $W_L=\mathbb{E}[\tau_L]$",
             xscale="log", yscale="log", xlim=(1, 64), ylim=(0.85, 2e12))
    work.xaxis.set_major_locator(LogLocator(base=2))
    work.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
    work.xaxis.set_minor_locator(NullLocator())
    work.grid(which="major", alpha=0.15)
    work.legend(loc="upper left", frameon=False, fontsize=10)
    for ax in (efficiency, work):
        ax.title.set_fontsize(11.5)
        ax.title.set_color(INK)
        ax.tick_params(labelsize=10)

    fig.text(0.5, 0.025, "Exact model calculations; acquisition of the guide is outside this experiment.",
             ha="center", fontsize=10, color="#677682")
    svg = OUTPUT / "search-efficiency.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    png = OUTPUT / "search-efficiency.png"
    fig.savefig(png, dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(f"Wrote {svg} and {png}")


if __name__ == "__main__":
    check_calculations()
    make_figure()
