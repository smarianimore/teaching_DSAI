"""
Animated visualization of Generate & Test on the Australia map-coloring CSP.

Reuses the CSP and the generator from generate_and_test.py, so what you see is
exactly the algorithm running: each complete assignment is generated, then
tested against every constraint.

Usage:
    python visualize_generate_and_test.py                 # open a window
    python visualize_generate_and_test.py --speed 20      # test 20 assignments per frame
    python visualize_generate_and_test.py --save gt.gif   # write a GIF instead of opening a window

Requires: matplotlib  (pip install matplotlib)
"""

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from generate_and_test import map_coloring_csp, generate, is_consistent

COLORS = {"R": "#e04a45", "G": "#3da34f", "B": "#3f6fd8"}
EMPTY = "#e7eaee"
BAD = "#c62828"

# Schematic map (same shapes as the HTML version; y grows downward there, so we flip below).
REGIONS = {
    "WA":  [(20, 60), (130, 60), (130, 250), (30, 250), (20, 180)],
    "NT":  [(130, 40), (230, 40), (230, 140), (130, 140)],
    "Q":   [(230, 40), (300, 10), (340, 120), (355, 170), (230, 170)],
    "SA":  [(130, 140), (230, 140), (230, 215), (215, 250), (130, 250)],
    "NSW": [(230, 170), (355, 170), (345, 225), (230, 225)],
    "V":   [(215, 250), (230, 225), (345, 225), (320, 257)],
    "T":   [(285, 285), (335, 285), (310, 318)],
}
NODES = {"WA": (40, 85), "NT": (140, 30), "Q": (255, 60), "SA": (140, 130),
         "NSW": (275, 150), "V": (215, 205), "T": (60, 205)}


def flip(points):
    return [(x, -y) for x, y in points]


def run_steps(csp, steps_per_frame):
    """Drive the G&T loop and yield one snapshot per animation frame.
    A frame ends early when a solution is found, so no solution is skipped."""
    gen = generate(csp)
    tested, solutions = 0, []
    while True:
        last = None
        for _ in range(steps_per_frame):
            assignment = next(gen, None)
            if assignment is None:
                break
            tested += 1
            ok = is_consistent(csp, assignment)
            if ok:
                solutions.append(assignment)
            last = (assignment, ok)
            if ok:
                break
        if last is None:
            return
        yield last[0], last[1], tested, tuple(solutions)
        if last[1]:                      # hold solutions on screen for a few frames
            for _ in range(8):
                yield last[0], last[1], tested, tuple(solutions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=int, default=5, help="assignments tested per frame (default 5)")
    parser.add_argument("--save", metavar="FILE", help="save animation to FILE (.gif) instead of showing it")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")

    csp = map_coloring_csp()
    total = 1
    for v in csp.variables:
        total *= len(csp.domains[v])
    total_solutions = sum(is_consistent(csp, assignment) for assignment in generate(csp))
    edges = [c.scope for c in csp.constraints]

    fig, (ax_map, ax_graph, ax_solutions) = plt.subplots(
        1, 3, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.1, 1, 1]}
    )
    fig.suptitle("Generate & Test: Australia map coloring", fontsize=14, fontweight="bold")

    # --- map ---
    patches = {}
    for v, pts in REGIONS.items():
        p = Polygon(flip(pts), closed=True, facecolor=EMPTY, edgecolor="white", linewidth=2)
        ax_map.add_patch(p)
        patches[v] = p
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        ax_map.text(cx, -cy, v, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax_map.set_xlim(0, 370)
    ax_map.set_ylim(-330, 0)
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # --- constraint graph ---
    lines = []
    for a, b in edges:
        (x1, y1), (x2, y2) = NODES[a], NODES[b]
        (line,) = ax_graph.plot([x1, x2], [-y1, -y2], color="#b9c0c9", linewidth=2, zorder=1)
        lines.append(line)
    dots = {}
    for v, (x, y) in NODES.items():
        (dot,) = ax_graph.plot([x], [-y], "o", markersize=34, markerfacecolor=EMPTY,
                               markeredgecolor="#b9c0c9", markeredgewidth=2, zorder=2)
        dots[v] = dot
        ax_graph.text(x, -y, v, ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=3)
    ax_graph.set_xlim(0, 320)
    ax_graph.set_ylim(-250, 0)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")
    ax_graph.set_title("Constraint graph (red = violated)", fontsize=10)

    ax_solutions.axis("off")
    ax_solutions.set_title("Solutions found: 0 / {}".format(total_solutions), fontsize=10)
    solution_header = f"{'No.':>3} " + " ".join(f"{v:>3}" for v in csp.variables)
    solution_list = ax_solutions.text(
        0.02, 0.96, solution_header, transform=ax_solutions.transAxes,
        ha="left", va="top", family="monospace", fontsize=8,
    )

    status = fig.text(0.5, 0.06, "", ha="center", fontsize=12, fontweight="bold")
    counters = fig.text(0.5, 0.02, "", ha="center", fontsize=10, color="#555")
    started_at = None

    def update(frame):
        nonlocal started_at
        if started_at is None:
            started_at = time.perf_counter()
        elapsed = time.perf_counter() - started_at
        minutes, seconds = divmod(elapsed, 60)

        assignment, ok, tested, solutions = frame
        violated = [(a, b) for a, b in edges if assignment[a] == assignment[b]]
        hit = {v for e in violated for v in e}
        for v in csp.variables:
            col = COLORS[assignment[v]]
            patches[v].set_facecolor(col)
            patches[v].set_edgecolor(BAD if v in hit else "white")
            patches[v].set_linewidth(4 if v in hit else 2)
            dots[v].set_markerfacecolor(col)
            dots[v].set_markeredgecolor(BAD if v in hit else "#b9c0c9")
        for line, e in zip(lines, edges):
            bad = e in violated
            line.set_color(BAD if bad else "#b9c0c9")
            line.set_linewidth(4 if bad else 2)
        if ok:
            status.set_text(f"Assignment {tested}: consistent. SOLUTION FOUND")
            status.set_color("#1e7a3a")
        else:
            status.set_text(f"Assignment {tested}: rejected ({len(violated)} constraint(s) violated)")
            status.set_color(BAD)
        ax_solutions.set_title(
            f"Solutions found: {len(solutions)} / {total_solutions}", fontsize=10
        )
        rows = [
            f"{i:>3} " + " ".join(f"{solution[v]:>3}" for v in csp.variables)
            for i, solution in enumerate(solutions, start=1)
        ]
        solution_list.set_text(solution_header + ("\n" + "\n".join(rows) if rows else ""))
        counters.set_text(
            f"tested {tested:,} / {total:,}    "
            f"solutions found: {len(solutions)} / {total_solutions}    "
            f"elapsed: {int(minutes):02}:{seconds:04.1f}"
        )
        return []

    anim = FuncAnimation(fig, update, frames=list(run_steps(csp, args.speed)),
                         interval=60, repeat=False, blit=False)

    if args.save:
        anim.save(args.save, writer="pillow", fps=15, dpi=80)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
