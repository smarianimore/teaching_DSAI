#!/usr/bin/env python3
"""
MINIMAX explained by two cartoon robots playing tic-tac-toe
===========================================================

Phase 1 - PLANNING ("in imagination")
    A translucent ghost of the robot whose turn it is walks down the game tree
    depth-first.  Every node of the tree is a real board position, every layer
    is one more move.  Leaves get a score (+1 MAX wins, 0 draw, -1 MIN wins),
    the scores are carried back up, and every inner node keeps the max (blue
    layers) or the min (red layers) of its children.

Phase 2 - PLAYING
    The real robots (bottom of the screen) now play the game on the real board,
    each one always picking the child with the best backed-up value.  The
    chosen path lights up in the tree.

Usage
-----
    python minimax_robots.py                       # watch it in a window
    In the window, Left/Right step backward/forward; Space toggles playback.
    python minimax_robots.py --save minimax.mp4    # needs ffmpeg
    python minimax_robots.py --save minimax.gif --dpi 60 --speed 2
    python minimax_robots.py --board "XO.OX...."   # another start position (X O .)

The start position must have <= 4 empty cells or so, otherwise the tree no
longer fits on a screen (5 empty cells already give ~200 positions).
The robot that has to move first in the start position is MAX.

Requires: numpy, matplotlib (ffmpeg for mp4, pillow for gif)
"""
import argparse
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D

DEFAULT_BOARD = "XOXX.O..."   # O to move; O must block, perfect play ends in a draw

# ----------------------------------------------------------------------------
# look & feel
# ----------------------------------------------------------------------------
BLUE, RED, GREY, GOLD, ORANGE = "#2f6fed", "#e8553d", "#8d949e", "#e0a800", "#f28c28"
BG, PANEL_BG = "#fbfaf7", "#efece3"
VCOL = {1: BLUE, 0: GREY, -1: RED}                          # colour of a score
VTINT = {1: "#dce7ff", 0: "#ececec", -1: "#ffe0da"}         # board tint of a scored node
FIG_W, FIG_H, PANEL_FRAC = 16, 9, 0.26
S, DX, BADGE = 0.88, 1.0, 0.34                              # board size, leaf spacing, score badge height
CELL = ["top-left", "top-middle", "top-right", "middle-left", "center",
        "middle-right", "bottom-left", "bottom-middle", "bottom-right"]
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def fmt(v):
    return "0" if v == 0 else f"{v:+d}".replace("-", "\u2212")


def shade(c, f):
    """f<1 -> darker, 1<f<=2 -> lighter."""
    r, g, b, _ = to_rgba(c)
    if f <= 1:
        return (r * f, g * f, b * f)
    t = f - 1
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


# ----------------------------------------------------------------------------
# 1.  tic-tac-toe + the minimax tree
# ----------------------------------------------------------------------------
def winner(b):
    for a, c, d in LINES:
        if b[a] != "." and b[a] == b[c] == b[d]:
            return b[a], (a, d)
    return None, None


def to_move(b):
    return "X" if b.count("X") == b.count("O") else "O"


class Node:
    def __init__(self, board, parent=None, move=None):
        self.board, self.parent, self.move = board, parent, move
        self.depth = 0 if parent is None else parent.depth + 1
        self.win, self.win_line = winner(board)
        self.terminal = bool(self.win) or "." not in board
        self.player = None if self.terminal else to_move(board)     # who moves HERE
        self.children, self.best = [], None
        self.value = self.dist = None
        self.hist = []                                              # (event, text, kind, value) badge updates
        self.reveal = self.decide = self.play = None                # event indices


def build_tree(board, parent=None, move=None):
    n = Node(board, parent, move)
    if not n.terminal:
        for i, ch in enumerate(board):
            if ch == ".":
                n.children.append(build_tree(board[:i] + n.player + board[i + 1:], n, i))
    return n


def solve(n, maxm):
    """Plain minimax (+1 / 0 / -1 from MAX's point of view).
    Ties are broken by 'win fast, lose slowly' so the robots never dawdle."""
    if n.terminal:
        n.value = 0 if n.win is None else (1 if n.win == maxm else -1)
        n.dist = 0
        return
    for c in n.children:
        solve(c, maxm)
    is_max = n.player == maxm
    vals = [c.value for c in n.children]
    n.value = max(vals) if is_max else min(vals)
    cands = [c for c in n.children if c.value == n.value]
    mover_wins = n.value > 0 if is_max else n.value < 0
    mover_loses = n.value < 0 if is_max else n.value > 0
    if mover_wins:
        n.best = min(cands, key=lambda c: c.dist)
    elif mover_loses:
        n.best = max(cands, key=lambda c: c.dist)
    else:
        n.best = cands[0]
    n.dist = 1 + n.best.dist


def layout_leaves(n, counter):
    if not n.children:
        n.lx = counter[0]
        counter[0] += 1
    else:
        for c in n.children:
            layout_leaves(c, counter)
        n.lx = (n.children[0].lx + n.children[-1].lx) / 2


# ----------------------------------------------------------------------------
# 2.  a cartoon robot (drawn from patches, movable through one Affine2D)
# ----------------------------------------------------------------------------
_x = np.linspace(-0.11, 0.11, 15)
_t = np.linspace(0, 2 * np.pi, 20)
MOUTH = {"smile": (_x, 0.90 + 0.06 * (_x / 0.11) ** 2),
         "sad": (_x, 0.95 - 0.05 * (_x / 0.11) ** 2),
         "flat": (_x, np.full_like(_x, 0.92)),
         "o": (0.05 * np.cos(_t), 0.91 + 0.05 * np.sin(_t))}


class Robot:
    """Feet at (0,0), about 0.7 wide and 1.5 tall in local units."""

    def __init__(self, ax, color, mark, scale=1.0, z=10, ghost=False):
        self.T, self.s, self.arts = Affine2D(), scale, []
        tr = self.T + ax.transData
        dark, light, lw = shade(color, .55), shade(color, 1.5), 1.5 * scale

        def patch(cls, *a, dz=0, **k):
            p = cls(*a, transform=tr, zorder=z + dz, **k)
            ax.add_patch(p)
            self.arts.append(p)
            return p

        def line(xs, ys, dz=0, **k):
            l = Line2D(xs, ys, transform=tr, zorder=z + dz, **k)
            ax.add_line(l)
            self.arts.append(l)
            return l

        self.arm = [line([0, 0], [0, 0], dz=-1, color=dark, lw=5 * scale, solid_capstyle="round") for _ in "LR"]
        self.hand = [patch(Circle, (0, 0), .075, fc=dark, ec=dark, dz=-1) for _ in "LR"]
        for x in (-.22, .08):                                                       # legs
            patch(Rectangle, (x, 0), .14, .2, fc=dark, ec=dark, lw=lw)
        patch(FancyBboxPatch, (-.34, .16), .68, .58, boxstyle="round,pad=0,rounding_size=0.12",
              fc=color, ec=dark, lw=lw)                                             # body
        patch(Rectangle, (-.19, .27), .38, .36, fc="white", ec=dark, lw=lw * .6, dz=1)   # chest window
        if mark == "X":
            for a, b in (((-.09, .34), (.09, .56)), ((-.09, .56), (.09, .34))):
                line([a[0], b[0]], [a[1], b[1]], dz=2, color=color, lw=2.4 * scale, solid_capstyle="round")
        else:
            patch(Circle, (0, .45), .1, fill=False, ec=color, lw=2.4 * scale, dz=2)
        patch(Rectangle, (-.06, .73), .12, .09, fc=dark, ec=dark)                   # neck
        patch(FancyBboxPatch, (-.3, .8), .6, .46, boxstyle="round,pad=0,rounding_size=0.15",
              fc=light, ec=dark, lw=lw)                                             # head
        for x in (-.35, .30):                                                       # ears
            patch(Rectangle, (x, .95), .05, .16, fc=dark, ec=dark)
        for x in (-.12, .12):                                                       # eyes
            patch(Circle, (x, 1.05), .085, fc="white", ec=dark, lw=lw * .6, dz=1)
            patch(Circle, (x, 1.05), .04, fc="black", dz=2)
        self.mouth = line([0], [0], dz=2, color=dark, lw=1.8 * scale, solid_capstyle="round")
        line([0, 0], [1.26, 1.40], color=dark, lw=1.8 * scale)                      # antenna
        self.bulb = patch(Circle, (0, 1.45), .06, ec=dark, lw=lw * .7)
        self.carry = None
        if ghost:                                                                   # thought bubbles + score label
            for (x, y, r) in ((.42, 1.22, .05), (.55, 1.36, .07)):
                patch(Circle, (x, y), r, fc="white", ec=dark, lw=lw * .6)
            self.carry = ax.text(0, 1.62, "", transform=tr, ha="center", va="bottom", fontsize=8.5,
                                 color="white", fontweight="bold", zorder=z + 6, visible=False,
                                 bbox=dict(boxstyle="round,pad=0.22", fc=GREY, ec="none"))

    def pose(self, x, y, left=0., right=0., mood="smile", light=False):
        self.T.clear().scale(self.s).translate(x, y)
        for i, (sgn, r) in enumerate(((-1, left), (1, right))):
            th = np.radians(18 + 150 * r)                        # 0 = hanging down, 1 = raised
            sx, sy = sgn * .3, .62
            hx, hy = sx + sgn * .42 * np.sin(th), sy - .42 * np.cos(th)
            self.arm[i].set_data([sx, hx], [sy, hy])
            self.hand[i].center = (hx, hy)
        self.mouth.set_data(*MOUTH[mood])
        self.bulb.set_facecolor("#ffd21f" if light else "#c8cdd4")

    def show(self, v):
        for a in self.arts:
            a.set_visible(v)
        if not v:
            self.set_carry(None)

    def set_alpha(self, a):
        for x in self.arts:
            x.set_alpha(a)

    def set_carry(self, v):
        if self.carry is None:
            return
        if v is None:
            self.carry.set_visible(False)
        else:
            self.carry.set_text(fmt(v))
            self.carry.get_bbox_patch().set_facecolor(VCOL[v])
            self.carry.set_visible(True)


class Badge:
    """The little score label below a board."""

    def __init__(self, ax, n, fs):
        w, h = .62, .27
        y = n.cy - S / 2 - BADGE + .03
        self.p = FancyBboxPatch((n.cx - w / 2, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                zorder=8, visible=False, lw=1.3)
        ax.add_patch(self.p)
        self.t = ax.text(n.cx, y + h / 2, "", ha="center", va="center", fontsize=fs,
                         fontweight="bold", zorder=9, visible=False)

    def hide(self):
        self.p.set_visible(False)
        self.t.set_visible(False)

    def show(self, txt, kind, v, a):
        col, full = VCOL[v], kind != "run"
        self.p.set_facecolor(col if full else "white")
        self.p.set_edgecolor(col)
        self.t.set_color("white" if full else col)
        self.t.set_text(txt)
        self.p.set_alpha(a)
        self.t.set_alpha(a)
        self.p.set_visible(True)
        self.t.set_visible(True)


def ease(f):
    return f * f * (3 - 2 * f)


def walk(pts, w, f):
    """Point at fraction f along a polyline whose segments take the time shares w."""
    if len(pts) == 1:
        return np.array(pts[0], float)
    cum = np.cumsum([0] + list(w))
    f = min(max(f, 0.0), 1.0)
    for i in range(len(pts) - 1):
        if f <= cum[i + 1] + 1e-9:
            t = (f - cum[i]) / (cum[i + 1] - cum[i])
            return np.array(pts[i], float) + t * (np.array(pts[i + 1], float) - np.array(pts[i], float))
    return np.array(pts[-1], float)


# ----------------------------------------------------------------------------
# 3.  build the whole animation
# ----------------------------------------------------------------------------
def build(start=DEFAULT_BOARD, speed=1.0):
    if len(start) != 9 or set(start) - set("XO.") or start.count("X") - start.count("O") not in (0, 1):
        raise ValueError("board must be 9 chars of X, O and '.', with #X == #O or #X == #O + 1")
    if winner(start)[0] or "." not in start:
        raise ValueError("the start position must not be finished already")
    maxm = to_move(start)
    minm = "O" if maxm == "X" else "X"
    root = build_tree(start)
    solve(root, maxm)
    nodes = []

    def collect(n):
        n.idx = len(nodes)
        nodes.append(n)
        for c in n.children:
            collect(c)
    collect(root)
    counter = [0]
    layout_leaves(root, counter)
    L, R = counter[0], max(n.depth for n in nodes) + 1
    if L > 40:
        print(f"warning: {L} leaves - the tree will be tiny; try a position with fewer empty cells")

    def nm(m):
        return "MAX" if m == maxm else "MIN"

    def gp(n):                                   # which robot's ghost stands at a node
        return n.player if not n.terminal else n.parent.player

    # ---- event script (planning = depth-first search, then the real game) ----
    ev = []

    def fr(k):
        return max(1, round(k / speed))

    def add(kind, node, trav=0, dwell=1, g=(None, None), say=None, carry=None, **kw):
        ev.append(dict(kind=kind, node=node, trav=fr(trav) if trav else 0, dwell=fr(dwell),
                       g=g, say=say or {}, carry=carry, **kw))
        return len(ev) - 1

    root.reveal = -1
    add("start", root, dwell=18, g=(maxm, maxm),
        say={maxm: "I'm MAX: I want the biggest score. Let me imagine every possible future, depth first..."})

    def rec(n):
        is_max, run = n.player == maxm, None
        for c in n.children:
            i = add("down", c, trav=6, dwell=2, g=(n.player, gp(c)),
                    say={n.player: f"What if I play {n.player} in the {CELL[c.move]}?"})
            c.reveal = i
            if c.terminal:
                txt = (f"{c.win} makes three in a row! Score {fmt(c.value)}." if c.win
                       else "Board full: a draw. Score 0.")
                j = add("leaf", c, dwell=9, g=(gp(c), gp(c)), carry=c.value, say={gp(c): txt})
                c.hist.append((j, fmt(c.value), "leaf", c.value))
            else:
                rec(c)
            run = c.value if run is None else (max(run, c.value) if is_max else min(run, c.value))
            i = add("up", c, trav=6, dwell=3, g=(gp(c), n.player), carry=c.value,
                    say={n.player: f"That line is worth {fmt(c.value)}. As {nm(n.player)} I keep the "
                                   f"{'biggest' if is_max else 'smallest'} so far: {fmt(run)}."})
            n.hist.append((i, ("\u2265" if is_max else "\u2264") + fmt(run), "run", run))
        vals = ", ".join(fmt(c.value) for c in n.children)
        j = add("decide", n, dwell=11, g=(n.player, n.player), carry=n.value,
                say={n.player: f"All replies explored: {'max' if is_max else 'min'}({vals}) = {fmt(n.value)}. "
                               f"Best move: {CELL[n.best.move]}."})
        n.hist.append((j, fmt(n.value), "final", n.value))
        n.decide = j

    rec(root)
    NP = len(ev)                                                   # number of planning events
    cur = root
    while not cur.terminal:
        c = cur.best
        c.play = add("play", c, dwell=26, g=(cur.player, cur.player), src=cur,
                     say={cur.player: f"I'm {nm(cur.player)}. My plan says the best I can reach is "
                                      f"{fmt(cur.value)}. I play {cur.player} in the {CELL[c.move]}!"})
        cur = c
    fin = cur
    if fin.win:
        say = {fin.win: "Three in a row - I win! Minimax works!",
               ("O" if fin.win == "X" else "X"): "Good game... I couldn't do better."}
    else:
        say = {maxm: "A draw! Neither of us can do better.", minm: "A draw! Perfect play from both sides."}
    add("end", fin, dwell=44, say=say)
    frames = [(i, k) for i, e in enumerate(ev) for k in range(e["trav"] + e["dwell"])]
    pv = set()
    p = root
    while p:
        pv.add(p)
        p = p.best

    # ---- geometry of the tree --------------------------------------------------
    aspect = FIG_W / (FIG_H * (1 - PANEL_FRAC))
    LEFT, RIGHT, TOP, BOT, DY0 = 2.3, 0.4, 1.9, 0.45, 1.75
    Wu0 = L * DX + LEFT + RIGHT
    Hu0 = (R - 1) * DY0 + S + BADGE + TOP + BOT
    Wu = max(Wu0, Hu0 * aspect)
    Hu = Wu / aspect
    DY = DY0 + (Hu - Hu0) / max(1, R - 1)
    xs0 = LEFT + (Wu - Wu0) / 2
    ppu = FIG_W * 72 / Wu                                          # points per data unit
    for n in nodes:
        n.cx = xs0 + (n.lx + .5) * DX
        n.cy = Hu - TOP - S / 2 - n.depth * DY

    def top(n):
        return (n.cx, n.cy + S / 2)

    def bot(n):
        return (n.cx, n.cy - S / 2 - BADGE)

    def cell_c(n, i):
        return (n.cx - S / 2 + (i % 3 + .5) * S / 3, n.cy + S / 2 - (i // 3 + .5) * S / 3)

    # ---- figure ------------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax = fig.add_axes([0, PANEL_FRAC, 1, 1 - PANEL_FRAC])
    pn = fig.add_axes([0, 0, 1, PANEL_FRAC])
    for a in (ax, pn):
        a.axis("off")
    ax.set_xlim(0, Wu)
    ax.set_ylim(0, Hu)
    PH = FIG_H * PANEL_FRAC
    pn.set_xlim(0, FIG_W)
    pn.set_ylim(0, PH)
    pn.add_patch(Rectangle((0, 0), FIG_W, PH, fc=PANEL_BG, ec="none", zorder=0))
    pn.plot([0, FIG_W], [PH, PH], color="#cfc9bb", lw=1.5, zorder=1)

    for d in range(R):                                              # layer bands + labels
        yc = Hu - TOP - S / 2 - d * DY
        m = maxm if d % 2 == 0 else minm
        ax.add_patch(Rectangle((0, yc - S / 2 - BADGE - .06), Wu, S + BADGE + .16,
                               fc=BLUE if m == maxm else RED, alpha=.07, ec="none", zorder=0))
        ax.text(.2, yc, f"{nm(m)}'s turn\n(plays {m})", fontsize=10.5, fontweight="bold",
                color=BLUE if m == maxm else RED, va="center", ha="left", zorder=2)
    ax.text(.2, Hu - .15, "MINIMAX on tic-tac-toe", fontsize=17, fontweight="bold", va="top", color="#222")
    ax.text(Wu / 2, Hu - .15, "Left / Right: previous / next step    Space: play / pause",
            fontsize=10, ha="center", va="top", color="#555")
    sub_t = ax.text(.2, Hu - .62, "", fontsize=11, va="top", color="#444", linespacing=1.25)
    cnt_t = ax.text(Wu - .2, Hu - .15, "", fontsize=10.5, va="top", ha="right", color="#444")
    ax.text(Wu - .2, Hu - .52, "scores are from MAX's point of view:", fontsize=9.5, va="top", ha="right", color="#666")
    for i, (v, lab) in enumerate(((1, "+1  MAX wins"), (0, "0  draw"), (-1, "\u22121  MIN wins"))):
        ax.text(Wu - .2, Hu - .82 - i * .27, lab, fontsize=10, va="top", ha="right", color=VCOL[v], fontweight="bold")

    # tree collections (rebuilt whenever the state changes)
    mlw = max(1.0, 0.12 * (S / 3) * ppu)
    hm = S / 3 * .27
    edge_lc = LineCollection([], zorder=1.5)
    node_pc = PolyCollection([], zorder=2)
    hl_pc = PolyCollection([], zorder=3, edgecolors="none")
    grid_lc = LineCollection([], zorder=4, linewidths=.6)
    x_lc = LineCollection([], zorder=5, linewidths=mlw, capstyle="round")
    win_lc = LineCollection([], zorder=6, linewidths=mlw * 1.4, capstyle="round")
    for c in (edge_lc, node_pc, hl_pc, grid_lc, x_lc, win_lc):
        ax.add_collection(c, autolim=False)
    o_sc = ax.scatter([], [], s=(2 * hm * 1.15 * ppu) ** 2, facecolors="none", linewidths=mlw, zorder=5)
    trail = Line2D([], [], color=ORANGE, lw=3, zorder=7, visible=False, solid_capstyle="round")
    ax.add_line(trail)
    badges = [Badge(ax, n, .55 * .27 * ppu) for n in nodes]

    ghosts = {maxm: Robot(ax, BLUE, maxm, .85, z=30, ghost=True), minm: Robot(ax, RED, minm, .85, z=30, ghost=True)}
    for g in ghosts.values():
        g.set_alpha(.82)
        g.show(False)

    # bottom panel: the two real robots, speech bubbles and the real board
    HX = {maxm: 1.45, minm: FIG_W - 1.45}
    home = {maxm: Robot(pn, BLUE, maxm, 1.1, z=10), minm: Robot(pn, RED, minm, 1.1, z=10)}
    for m in (maxm, minm):
        pn.text(HX[m], .13, f"{nm(m)}-bot  ({m})", ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=BLUE if m == maxm else RED)
    bub = {maxm: pn.text(2.6, 1.3, "", ha="left", va="center", fontsize=12.5, zorder=5, color="#222"),
           minm: pn.text(FIG_W - 2.6, 1.3, "", ha="right", va="center", fontsize=12.5, zorder=5, color="#222")}
    BX, BY, BC = FIG_W / 2 - .9, .3, .6
    pn.add_patch(Rectangle((BX, BY), 3 * BC, 3 * BC, fc="white", ec="#555", lw=1.5, zorder=2))
    for j in (1, 2):
        pn.plot([BX + j * BC] * 2, [BY, BY + 3 * BC], color="#555", lw=2, zorder=3)
        pn.plot([BX, BX + 3 * BC], [BY + j * BC] * 2, color="#555", lw=2, zorder=3)
    pn.text(FIG_W / 2, BY + 3 * BC + .12, "the real game", ha="center", va="bottom", fontsize=10, color="#666")
    rmarks = []
    for i in range(9):
        cx, cy, h = BX + (i % 3 + .5) * BC, BY + (2 - i // 3 + .5) * BC, BC * .26
        l1 = Line2D([cx - h, cx + h], [cy - h, cy + h], lw=5.5, solid_capstyle="round", zorder=4, visible=False)
        l2 = Line2D([cx - h, cx + h], [cy + h, cy - h], lw=5.5, solid_capstyle="round", zorder=4, visible=False)
        o = Circle((cx, cy), h * 1.1, fill=False, lw=5.5, zorder=4, visible=False)
        pn.add_line(l1)
        pn.add_line(l2)
        pn.add_patch(o)
        rmarks.append((l1, l2, o))

    st = dict(key=None, ph=None, rb=None)

    def set_real_board(b):
        for i, ch in enumerate(b):
            l1, l2, o = rmarks[i]
            col = BLUE if ch == maxm else RED
            for a in (l1, l2, o):
                a.set_visible(False)
            if ch == "X":
                for a in (l1, l2):
                    a.set_color(col)
                    a.set_visible(True)
            elif ch == "O":
                o.set_edgecolor(col)
                o.set_visible(True)

    # ---- redraw of everything that only changes between events ------------------
    def refresh(e, arrived):
        ph2 = e >= NP
        E = ev[e]
        kind = E["kind"]

        def ap_(i):
            return i is not None and (i < e or (i == e and arrived))

        active, real_set, real_cur = None, set(), None
        if ph2:
            real_set = {n for n in nodes if n is root or ap_(n.play)}
            real_cur = max(real_set, key=lambda n: n.depth)
        else:
            node = E["node"]
            active = {"down": node if arrived else node.parent,
                      "up": node.parent if arrived else node}.get(kind, node)
        P, FC, EC, LW, HL, HLc, G, X, Xc, Oc, Occ, W, Wc, ES, EC2, EW = ([] for _ in range(16))
        count = 0
        for n in nodes:
            vis = ph2 or ap_(n.reveal)
            b = badges[n.idx]
            if not vis:
                b.hide()
                continue
            count += 1
            a = 1.0 if not ph2 else (1.0 if n in pv else (.5 if n.parent in pv else .15))
            x0, y0 = n.cx - S / 2, n.cy - S / 2
            stt = None
            for i, txt, kd, val in n.hist:
                if ap_(i):
                    stt = (txt, kd, val)
            if stt:
                b.show(*stt, a)
            else:
                b.hide()
            tint = VTINT[stt[2]] if stt and stt[1] != "run" else "white"
            if n is active:
                ec, lw = ORANGE, 2.6
            elif n is real_cur:
                ec, lw = GOLD, 3.4
            elif n in real_set:
                ec, lw = GOLD, 2.0
            else:
                ec, lw = "#5b6068", .8
            P.append([(x0, y0), (x0 + S, y0), (x0 + S, y0 + S), (x0, y0 + S)])
            FC.append(to_rgba(tint, a))
            EC.append(to_rgba(ec, a))
            LW.append(lw)
            if n.move is not None:
                cx, cy = cell_c(n, n.move)
                s = S / 6
                HL.append([(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)])
                HLc.append(to_rgba("#ffe98a", a))
            for j in (1, 2):
                G.append([(x0 + j * S / 3, y0), (x0 + j * S / 3, y0 + S)])
                G.append([(x0, y0 + j * S / 3), (x0 + S, y0 + j * S / 3)])
            for i, ch in enumerate(n.board):
                if ch == ".":
                    continue
                cx, cy = cell_c(n, i)
                col = to_rgba(BLUE if ch == maxm else RED, a)
                if ch == "X":
                    X += [[(cx - hm, cy - hm), (cx + hm, cy + hm)], [(cx - hm, cy + hm), (cx + hm, cy - hm)]]
                    Xc += [col, col]
                else:
                    Oc.append((cx, cy))
                    Occ.append(col)
            if n.win:
                W.append([cell_c(n, n.win_line[0]), cell_c(n, n.win_line[1])])
                Wc.append(to_rgba(BLUE if n.win == maxm else RED, a))
            if n.parent is not None:
                gold = ph2 and n.play is not None and n.play <= e
                best = n.parent.best is n and (ph2 or ap_(n.parent.decide))
                col = GOLD if gold else ((BLUE if n.parent.player == maxm else RED) if best else "#b4b9c0")
                ES.append([bot(n.parent), top(n)])
                EC2.append(to_rgba(col, a))
                EW.append(4.0 if gold else (2.4 if best else 1.0))
        node_pc.set_verts(P)
        node_pc.set_facecolor(FC)
        node_pc.set_edgecolor(EC)
        node_pc.set_linewidth(LW)
        hl_pc.set_verts(HL)
        if HL:
            hl_pc.set_facecolor(HLc)
        grid_lc.set_segments(G)
        grid_lc.set_color("#9aa0a8")
        x_lc.set_segments(X)
        if X:
            x_lc.set_color(Xc)
        win_lc.set_segments(W)
        if W:
            win_lc.set_color(Wc)
        edge_lc.set_segments(ES)
        if ES:
            edge_lc.set_color(EC2)
            edge_lc.set_linewidth(EW)
        o_sc.set_offsets(np.array(Oc, float).reshape(-1, 2))
        if Oc:
            o_sc.set_edgecolor(Occ)
        cnt_t.set_text(f"positions explored: {count} / {len(nodes)}")
        if st["ph"] != ph2:
            st["ph"] = ph2
            sub_t.set_text(textwrap.fill(
                "Phase 2 \u00b7 PLAYING: each robot plays the move with the best backed-up value" if ph2 else
                "Phase 1 \u00b7 PLANNING: MAX imagines possible futures; the ghost shows whose turn is being simulated", 46))
            for m in (maxm, minm):
                bub[m].set_bbox(dict(boxstyle="round,pad=0.5", fc="white" if ph2 else "#fff9d6",
                                     ec=BLUE if m == maxm else RED, lw=2 if ph2 else 1.5,
                                     ls="-" if ph2 else "--"))

    # ---- one animation frame ----------------------------------------------------
    def render(fi):
        e, k = frames[fi]
        E = ev[e]
        kind, n_fr = E["kind"], E["trav"] + E["dwell"]
        u = k / max(1, n_fr - 1)
        if kind == "play":
            frac, arrived = 1.0, u >= .5
        elif E["trav"]:
            frac = min(1.0, (k + 1) / E["trav"])
            arrived = frac >= 1.0
        else:
            frac, arrived = 1.0, True
        if st["key"] != (e, arrived):
            st["key"] = (e, arrived)
            refresh(e, arrived)
        ph2 = e >= NP
        node = E["node"]

        # ghosts walking through the tree (phase 1)
        trail.set_visible(False)
        if not ph2:
            if kind == "down":
                pts, w = [top(node.parent), bot(node.parent), top(node)], [.3, .7]
            elif kind == "up":
                pts, w = [top(node), bot(node.parent), top(node.parent)], [.7, .3]
            else:
                pts, w = [top(node)], [1]
            f = ease(frac)
            gx, gy = walk(pts, w, f)
            who = E["g"][1] if arrived else E["g"][0]
            for m in (maxm, minm):
                g = ghosts[m]
                if m != who:
                    g.show(False)
                    continue
                g.show(True)
                g.pose(gx, gy + .02 + .02 * np.sin(fi * .6),
                       left=.5 if kind == "up" else 0, right=.5 if kind == "up" else 0,
                       mood={"leaf": "o", "decide": "smile"}.get(kind, "flat"))
                g.set_carry(E["carry"])
            if kind == "down" and not arrived:
                a_, b_ = np.array(bot(node.parent)), np.array(top(node))
                end = a_ + np.clip((f - .3) / .7, 0, 1) * (b_ - a_)
                trail.set_data([a_[0], end[0]], [a_[1], end[1]])
                trail.set_visible(True)
        else:
            for g in ghosts.values():
                g.show(False)

        # real board
        rb = root.board if not ph2 else (E["src"].board if kind == "play" and not arrived else node.board)
        if rb != st["rb"]:
            st["rb"] = rb
            set_real_board(rb)

        # the two real robots + speech
        for m in (maxm, minm):
            sp = m in E["say"]
            bob = .035 * np.sin(fi * .35 + (0 if m == maxm else 1.7))
            inner = outer = 0.0
            mood, light = "smile", False
            if not ph2:
                if sp:
                    outer = .6 if kind in ("start", "down") else 0.0
                    mood = "smile" if kind == "decide" else ("o" if kind == "leaf" else "flat")
                    light = (fi // 4) % 2 == 0
            elif kind == "play":
                if m == E["src"].player:
                    sw = 1 - abs(2 * u - 1)
                    inner, bob, light = sw, bob + .12 * sw, True
            else:
                if node.win is None:
                    mood = "flat"
                elif node.win == m:
                    inner = outer = 1.0
                    bob = .16 * abs(np.sin(fi * .45))
                else:
                    mood = "sad"
            left, right = (outer, inner) if m == maxm else (inner, outer)
            home[m].pose(HX[m], .45 + bob, left, right, mood, light)
            bub[m].set_visible(sp)
            if sp:
                bub[m].set_text(textwrap.fill(E["say"][m], 36))

    return fig, render, len(frames), ev


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Minimax explained by two robots playing tic-tac-toe")
    ap.add_argument("--board", default=DEFAULT_BOARD, help="start position, 9 chars of X O . (row by row)")
    ap.add_argument("--save", help="write a .mp4 (needs ffmpeg) or .gif instead of opening a window")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast (fewer frames)")
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    fig, render, n, events = build(args.board, args.speed)
    if args.save:
        anim = FuncAnimation(fig, render, frames=n, interval=1000 / args.fps, blit=False, repeat=True)
        writer = (PillowWriter(fps=args.fps) if args.save.lower().endswith(".gif")
                  else FFMpegWriter(fps=args.fps, codec="libx264", bitrate=3500,
                                    extra_args=["-pix_fmt", "yuv420p"]))
        anim.save(args.save, writer=writer, dpi=args.dpi,
                  progress_callback=lambda i, total: print(f"\rrendering {i + 1}/{total}", end="", flush=True))
        print(f"\nsaved {args.save}")
    else:
        event_starts = []
        frame = 0
        for event in events:
            event_starts.append(frame)
            frame += event["trav"] + event["dwell"]

        frame_index = 0
        playing = True
        timer = fig.canvas.new_timer(interval=1000 / args.fps)

        def tick():
            nonlocal frame_index
            frame_index = (frame_index + 1) % n
            render(frame_index)
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal frame_index, playing
            if event.key == " ":
                playing = not playing
                (timer.start if playing else timer.stop)()
                return
            if event.key not in ("left", "right"):
                return

            timer.stop()
            playing = False
            current = next(i for i, start in enumerate(event_starts)
                           if start <= frame_index < (event_starts[i + 1] if i + 1 < len(event_starts) else n))
            target = min(max(current + (1 if event.key == "right" else -1), 0), len(event_starts) - 1)
            frame_index = (event_starts[target + 1] if target + 1 < len(event_starts) else n) - 1
            render(frame_index)
            fig.canvas.draw_idle()

        timer.add_callback(tick)
        fig.canvas.mpl_connect("key_press_event", on_key)
        render(frame_index)
        timer.start()
        plt.show()


if __name__ == "__main__":
    main()
