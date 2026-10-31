"""Aligned (rotated) overlay of the bundle cross-sections (David, 2026-10-14): every complete compact slice of the
cutout tracking (figures/pr_bundle_track_group<g>.tsv) is rotated about the bundle center so that the R7 points up
(+v), which removes the twist of the bundle and any arbitrary frame orientation; the six R1-R6 of all slices then
overlay as a fixed trapezoid. Per group: all rotated slice positions (small dots, colored by terminal-plane name) and
the circular-mean positions (large markers); the symmetry axis is vertical (through R7 and the bundle center).
Output figures/pr_ring_aligned.png with one panel per group that has a tracking table.
Usage: python faab_pr_ring_aligned.py [groups ...]"""
from faab.config import REPO
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from faab.config import fig

OUT = os.path.join(REPO, "figures")
COL = {"R2/R5": "#7F77DD", "R1/R6": "#D85A30", "R3/R4": "#1D9E75", "R7": "#A32D2D", "R8": "#0F6E56"}
RING_MAX = 2.5


def aligned(g):
    df = pd.read_csv(fig(f"pr_bundle_track_group{g}.tsv"), sep="\t")
    df = df[df.complete]
    rows = []
    for s, sub in df.groupby("s_box"):
        out = sub[~sub.cell.isin(["R7", "R8"])]
        if len(out) != 6 or out.radius.max() > RING_MAX or "R7" not in set(sub.cell):
            continue
        a7 = np.radians(float(sub[sub.cell == "R7"].angle.iloc[0]))
        rot = np.pi / 2 - a7  # R7 -> +v
        for r in sub.itertuples():
            a = np.radians(r.angle) + rot
            rows.append(dict(s=s, root=r.root, cell=r.cell, u=r.radius * np.cos(a), v=r.radius * np.sin(a)))
    return pd.DataFrame(rows)


def main():
    groups = [int(a) for a in sys.argv[1:]] or sorted(int(f.split("group")[1].split(".")[0]) for f in os.listdir(OUT)
                                                      if f.startswith("pr_bundle_track_group") and f.endswith(".tsv"))
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 6.5), squeeze=False)
    for ax, g in zip(axes[0], groups):
        A = aligned(g)
        for (root, cell), sub in A.groupby(["root", "cell"]):
            ax.scatter(sub.u, sub.v, s=10, color=COL.get(cell, "#888"), alpha=0.35)
        means = A.groupby(["root", "cell"])[["u", "v"]].mean().reset_index()
        for r in means.itertuples():
            mk = "D" if r.cell in ("R7", "R8") else "o"
            ax.scatter(r.u, r.v, s=260, marker=mk, color=COL.get(r.cell, "#888"), edgecolor="k", zorder=5)
            ax.annotate(r.cell, (r.u, r.v), xytext=(9, 6), textcoords="offset points", fontsize=9)
        ax.axvline(0, color="#378ADD", lw=1.5, label="symmetry axis (through R7 and the center)")
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("um (rotated so that R7 points up)"); ax.set_ylabel("um")
        ax.set_title(f"group {g}: {A.s.nunique()} compact slices overlaid, rotated to R7", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Bundle cross-sections aligned on R7: small dots = every slice, large = mean; colors = terminal-plane names")
    fig.tight_layout()
    png = fig("pr_ring_aligned.png")
    fig.savefig(png, dpi=130)
    print("saved", png)


if __name__ == "__main__":
    main()
