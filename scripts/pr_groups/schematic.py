"""Schematic (rendered image, no overlapping text): photoreceptor arrangement in Drosophila (literature) vs FAAB
(measured, groups 1 and 4) in three rows: retina cross-section, axon bundle cross-section, lamina. The trapezoid is
drawn as the quadrilateral R1-R3-R4-R6 (dotted); R2 and R5 sit on its legs in the fly retina and bulge outside them
in the FAAB bundle (measured 168 deg apart). Output figures/pr_bundle_schematic.png. Network-free."""
from faab.config import REPO
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, FancyBboxPatch
from faab.config import fig

OUT = os.path.join(REPO, "figures")
C = {"R1": "#D85A30", "R6": "#D85A30", "R2": "#7F77DD", "R5": "#7F77DD", "R3": "#1D9E75", "R4": "#1D9E75",
     "R7": "#A32D2D", "R8": "#0F6E56"}


def cell(ax, x, y, name, r=0.42, label=True, dx=0.55, dy=0.0, ha="left"):
    ax.add_patch(Circle((x, y), r, color=C[name], ec="k", lw=0.5, zorder=4))
    if label:
        ax.text(x + dx, y + dy, name, fontsize=8, ha=ha, va="center", zorder=5)


def trapezoid(ax, pts):
    ax.add_patch(Polygon(pts, closed=True, fill=False, ls=(0, (1, 3)), lw=0.9, ec="#777", zorder=2))


def panel(ax, title, caption):
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10, pad=4)
    ax.text(0, -5.6, caption, fontsize=8.5, ha="center", va="top", wrap=True)


def main():
    fig, ax = plt.subplots(3, 2, figsize=(11, 15))
    fig.subplots_adjust(hspace=0.25, wspace=0.08, top=0.95, bottom=0.02)

    # ---- retina, Drosophila: open rhabdom, trapezoid R1-R3-R4-R6, R2/R5 on the legs, R7 over R8 central
    A = ax[0, 0]
    P = {"R1": (-2.2, 2.0), "R2": (-1.9, -0.2), "R3": (-1.0, -2.3), "R4": (1.0, -2.3), "R5": (1.9, -0.2), "R6": (2.2, 2.0)}
    trapezoid(A, [P["R1"], P["R3"], P["R4"], P["R6"]])
    for n, (x, y) in P.items():
        cell(A, x, y, n, dx=-0.6 if x < 0 else 0.6, ha="right" if x < 0 else "left")
    cell(A, 0, 0.7, "R7", dx=0.6); cell(A, 0, -0.5, "R8", r=0.3, dx=0.5)
    panel(A, "Drosophila retina (literature)", "open rhabdom: seven separate visual axes\nR7 stacked over R8 in the center\n"
                                                "dotted: the trapezoid R1-R3-R4-R6, R2 and R5 on its legs")
    # ---- retina, FAAB (Brammer 1970): fused rhabdom, R8 central, R7 wedged between two outer cells
    B = ax[0, 1]
    ang = {"R1": 50, "R6": -50, "R2": 110, "R5": -110, "R3": 160, "R4": -160}
    Q = {n: (2.6 * np.sin(np.radians(a)), 2.6 * np.cos(np.radians(a))) for n, a in ang.items()}
    trapezoid(B, [Q["R1"], Q["R3"], Q["R4"], Q["R6"]])
    for n, (x, y) in Q.items():
        cell(B, x, y, n, dx=-0.6 if x < 0 else 0.6, ha="right" if x < 0 else "left")
    cell(B, 0, 3.0, "R7", dx=0.6); cell(B, 0, 0, "R8", r=0.5, dx=0.7)
    panel(B, "FAAB retina (Brammer 1970)", "fused rhabdom: one visual axis\nR8 central, R7 wedged between two outer cells\n"
                                            "same trapezoid topology")
    # ---- bundle, Drosophila (Braitenberg 1967, Musca): crown of 8, order 1,8,2,3,4,5,6,7
    D = ax[1, 0]
    order = ["R1", "R8", "R2", "R3", "R4", "R5", "R6", "R7"]
    for k, n in enumerate(order):
        a = np.radians(90 + 45 * k)
        x, y = 2.6 * np.cos(a), 2.6 * np.sin(a)
        cell(D, x, y, n, r=0.42 if n != "R8" else 0.3, dx=0.9 * np.cos(a), dy=0.9 * np.sin(a), ha="center")
    pts = {n: (2.6 * np.cos(np.radians(90 + 45 * order.index(n))), 2.6 * np.sin(np.radians(90 + 45 * order.index(n))))
           for n in ("R1", "R3", "R4", "R6")}
    trapezoid(D, [pts["R1"], pts["R3"], pts["R4"], pts["R6"]])
    panel(D, "Drosophila axon bundle (Braitenberg 1967, Musca)", "crown of eight fibers, cyclic order 1, 8, 2, 3, 4, 5, 6, 7\n"
                                                                  "R7 and R8 are part of the ring\nbundle twists 180 deg before the lamina")
    # ---- bundle, FAAB measured: ring of six (angles vs R7 from groups 1 and 4), R8 central, R7 just inside the ring
    E = ax[1, 1]
    meas = {"R1": 42, "R6": -42, "R2": 100, "R5": -100, "R3": 163, "R4": -163}  # deg from R7, group 1 means
    M = {n: (2.6 * np.sin(np.radians(a)), 2.6 * np.cos(np.radians(a))) for n, a in meas.items()}
    trapezoid(E, [M["R1"], M["R3"], M["R4"], M["R6"]])
    E.plot([0, 0], [-4.2, 4.2], color="#378ADD", lw=1, ls="--", zorder=1)
    for n, (x, y) in M.items():
        cell(E, x, y, n, dx=-0.6 if x < 0 else 0.6, ha="right" if x < 0 else "left")
    cell(E, 0, 2.3, "R7", dx=0.6); cell(E, 0, 0, "R8", r=0.32, dx=0.5)
    panel(E, "FAAB axon bundle (cutout tracking, groups 1 and 4)", "ring of six in fly order; R8 at the center (0.3 um)\n"
                                                                    "R7 just inside the ring between R1 and R6; twist ~1 deg/um\n"
                                                                    "R2/R5 bulge outside the trapezoid legs (168 deg apart)")
    # ---- lamina, Drosophila: six axons from six ommatidia converge on one cartridge; compact terminals
    F = ax[2, 0]
    F.add_patch(FancyBboxPatch((-1.1, -3.2), 2.2, 6.0, boxstyle="round,pad=0.1", fc="#E6F1FB", ec="#378ADD", lw=0.6, zorder=1))
    for k, (n, (x, y)) in enumerate({"R1": (-0.45, 1.6), "R2": (0.45, 1.6), "R3": (-0.45, 0.4), "R4": (0.45, 0.4),
                                     "R5": (-0.45, -0.8), "R6": (0.45, -0.8)}.items()):
        cell(F, x, y, n, r=0.3, label=False)
    for y0 in (2.4, 0.4, -1.6):
        F.add_patch(FancyArrowPatch((-4.2, y0), (-1.4, 0.4 + (y0 - 0.4) * 0.5), arrowstyle="-|>", mutation_scale=10, color="#777", lw=0.8))
    F.text(-4.4, 0.4, "from six\nommatidia", fontsize=8, ha="right", va="center")
    F.text(1.5, 1.2, "one cartridge:\n6 R1-R6 terminals\n+ L1, L2 ...", fontsize=8, va="center")
    panel(F, "Drosophila lamina (FlyWire measurement)", "neural superposition: each axon goes to a different cartridge\n"
                                                         "terminal p90 radius ~3 um")
    # ---- lamina, FAAB: all eight together, wide overlapping terminals; R8 ends, R7 continues
    G = ax[2, 1]
    for n, (x, y, r) in {"R1": (-0.8, 0.6, 3.2), "R6": (0.9, 0.5, 3.0), "R2": (-1.4, -0.4, 3.1), "R5": (1.5, -0.3, 3.0),
                         "R3": (-0.5, -1.0, 2.9), "R4": (0.6, -1.1, 3.0)}.items():
        G.add_patch(Circle((x, y), r, color=C[n], alpha=0.10, zorder=1))
        cell(G, x, y, n, r=0.3, label=False)
    cell(G, 0, 0, "R8", r=0.3, dx=0.45)
    G.add_patch(FancyArrowPatch((0, 0.4), (0, 4.6), arrowstyle="-|>", mutation_scale=12, color=C["R7"], lw=1.2, zorder=5))
    G.text(0.3, 3.9, "R7 continues\nto the medulla", fontsize=8, va="center", color=C["R7"])
    G.add_patch(FancyArrowPatch((-3.2, -3.2), (-0.3, -0.25), arrowstyle="-|>", mutation_scale=10, color="#777", lw=0.8))
    G.text(-3.3, -3.4, "all eight from\none ommatidium", fontsize=8, ha="right", va="top")
    panel(G, "FAAB lamina (groups 1-8)", "all eight stay together; terminals 8-11 um radius\n"
                                          "spread over 3-5 cartridge spacings (pooling)\nR8 ends here and synapses onto R7 (R8 -> R7, not back)")
    fig.suptitle("Photoreceptor arrangement: Drosophila (literature) vs FAAB (measured)", fontsize=13)
    png = fig("pr_bundle_schematic.png")
    fig.savefig(png, dpi=140)
    print("saved", png)


if __name__ == "__main__":
    main()
