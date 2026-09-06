# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Figures and numerical checks for 'An autoprover as a refrigerator'.

Run: uv run assets/autoprover/models.py
These are model calculations, not data from theorem-prover runs.
The article supplies the proofs; the finite numerical checks are independent
checks of the formula implementations used here.
"""
from itertools import combinations
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent


def seed_density(x, density):
    return 1 - (1 - x) * math.exp(density * x * x)


def least_closure(rho, density):
    """Solve on a monotone branch; avoid skipping a narrow pair of roots."""
    if rho == 0:
        return 0.0
    if density > 2:
        low_fold = (1 - math.sqrt(1 - 2 / density)) / 2
        high_fold = 1 - low_fold
        critical = seed_density(low_fold, density)
        if rho <= critical:
            lo, hi = 0.0, low_fold
        else:
            lo, hi = high_fold, 1.0
    else:
        lo, hi = 0.0, 1.0
    for _ in range(65):
        mid = (lo + hi) / 2
        if seed_density(mid, density) < rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def grouped_closure(rho):
    return rho + (1 - rho) * (3 * rho**2 - 2 * rho**3)


def completion_cost(length, p):
    """Stable geometric-sum form of the reflected-chain formula."""
    ratio = (1 - p) / p
    return sum((length - k) * ratio**k for k in range(length)) / p


def check_calculations():
    # Independently solve T = 1 + transition_matrix @ T.
    for length in (1, 2, 5, 9):
        for p in (.2, .5, .8):
            transition = np.zeros((length, length))
            for state in range(1, length + 1):
                if state > 1:
                    transition[state - 1, state - 2] += p
                transition[state - 1, min(state + 1, length) - 1] += 1 - p
            numerical = np.linalg.solve(np.eye(length) - transition, np.ones(length))[-1]
            analytic = completion_cost(length, p)
            assert math.isclose(numerical, analytic, rel_tol=2e-10), (length, p)
    # Enumerate actual closure under all twelve rules in a four-object group.
    rules = [(set(pair), head) for head in range(4)
             for pair in combinations([i for i in range(4) if i != head], 2)]
    for rho in (0.0, .098, .099, .3, .75, 1.0):
        expected = 0.0
        for mask in range(16):
            known = {i for i in range(4) if mask & (1 << i)}
            count = len(known)
            changed = True
            while changed:
                previous = len(known)
                for premises, head in rules:
                    if premises <= known:
                        known.add(head)
                changed = len(known) != previous
            expected += rho**count * (1 - rho)**(4 - count) * len(known) / 4
        assert math.isclose(expected, grouped_closure(rho), abs_tol=1e-14)
    for density in (1.5, 2.0, 3.0):
        values = [least_closure(rho, density) for rho in (.01, .098, .099, .2, .3)]
        assert values == sorted(values)
        for rho, x in zip((.01, .098, .099, .2, .3), values):
            assert math.isclose(x, 1 - (1 - rho) * math.exp(-density * x*x), abs_tol=1e-13)
    print("Finite-chain, exhaustive group-closure, and fixed-point checks passed.")


def save(fig, name):
    svg = OUTPUT / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    # Keep generated XML clean in version control; newlines remain separators.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def figures():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "autoprover-models",
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    rhos = np.linspace(0, .3, 1801)
    for density, color in ((1.5, "#23789f"), (2.0, "#a9740b"), (3.0, "#b53a4b")):
        if density == 3:
            fold = (1 - math.sqrt(1 - 2 / density)) / 2
            critical = seed_density(fold, density)
            lower = np.linspace(0, critical - 1e-9, 800)
            upper = np.linspace(critical + 1e-9, .3, 800)
            ax.plot(lower, [least_closure(r, density) for r in lower],
                    color=color, lw=2.6, label=r"$\lambda = 3$")
            ax.plot(upper, [least_closure(r, density) for r in upper], color=color, lw=2.6)
            top = least_closure(critical + 1e-9, density)
            ax.plot([critical, critical], [fold, top], color=color, lw=1.3, ls=":")
            ax.annotate("Jump near 9.83% seeds",
                        xy=(critical, .66), xytext=(.125, .60),
                        fontsize=11, arrowprops={"arrowstyle": "->", "color": "#555"})
        else:
            ax.plot(rhos, [least_closure(r, density) for r in rhos],
                    color=color, lw=2.6, label=rf"$\lambda = {density:g}$")
    ax.plot(rhos, [grouped_closure(r) for r in rhos],
            color="#5b6269", lw=2, ls="--", label=r"Four-object groups ($\lambda = 3$)")
    ax.set(xlim=(0, .3), ylim=(0, 1), xlabel=r"Initially supplied fraction, $\rho$",
           ylabel=r"Fraction in the easy closure, $x$")
    ax.set_title("Accessible knowledge in two dependency ensembles", fontsize=14, loc="left", pad=14)
    ax.grid(alpha=.15)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    save(fig, "knowledge-cascade")

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    lengths = range(1, 81)
    for ratio, color, label in (
        (2, "#23789f", r"$w/b=2$: asymptotically linear"),
        (1, "#a9740b", r"$w/b=1$: quadratic"),
        (.5, "#b53a4b", r"$w/b=1/2$: exponential"),
    ):
        p = ratio / (1 + ratio)
        ax.plot(list(lengths), [completion_cost(length, p) for length in lengths],
                lw=2.6, color=color, label=label)
    ax.set(yscale="log", xlim=(1, 80), xlabel=r"Initial progress coordinate, $L$",
           ylabel=r"Expected reasoning selections, $T_L$")
    ax.set_title("Guidance changes the scaling of search cost", fontsize=14, loc="left", pad=14)
    ax.grid(alpha=.15)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save(fig, "guided-search")


if __name__ == "__main__":
    check_calculations()
    fold = (1 - math.sqrt(1 - 2 / 3)) / 2
    print(f"Critical seed density at lambda=3: {seed_density(fold, 3):.10f}")
    for rho in (.098, .099):
        x = least_closure(rho, 3)
        print(f"rho={rho}: closure={x:.8f}, unknown-target probability={(x-rho)/(1-rho):.8f}")
    figures()
    print(f"Figures written to {OUTPUT}")
