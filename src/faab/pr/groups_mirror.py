"""Mirror-pair test for the R1-R6 of each photoreceptor group, IN SITU (no re-alignment).

For each group: bundle axis = first principal component of the pooled R1-R6 skeleton
points. Candidate mirror planes all contain that axis and pass through the R1-R6 centroid;
their orientation is a single angle theta (0-180 deg, 3 deg steps). For each theta, every
R1-R6 cell is reflected across the plane and NBLASTed (normalised, mean of both
directions) against the other five unreflected cells. The plane that maximises the mean
best-match score is taken as the group's mirror plane; mutual best matches are the mirror
pairs. Also reported: whether the R7/R8 pair lies in that plane (angle between the plane
normal and the centroid->R7/R8 direction; 90 deg = in plane), and the in-place NBLAST
without reflection as a baseline. Local only, uses cached L2 skeletons.
Outputs: figures/faab_pr_groups_mirror.tsv, faab_pr_groups_mirror.txt, faab_pr_groups_mirror.png
"""
from faab.config import REPO
import os
import pickle
import time
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
G1 = os.path.join(OUT, "faab_pr_current_roots.tsv")
G28 = os.path.join(OUT, "faab_pr_groups_xyz.tsv")
THETAS = np.arange(0, 180, 3)


def load_groups():
    groups = {}
    g1 = pd.read_csv(G1, sep="\t", dtype=str)
    groups[1] = [(r["root_id_curr"], r["class"]) for _, r in g1.iterrows()]
    g = pd.read_csv(G28, sep="\t", dtype=str)
    for k, sub in g.groupby("group", sort=False):
        groups[int(k) + 1] = [(r["root_id_curr"], r["class"]) for _, r in sub.iterrows()]
    return groups


def prep(sk):
    n = navis.resample_skeleton(sk.copy(), resample_to=1000, inplace=False)
    n = n / 1000.0
    n.units = "um"
    return n


def reflect(n, c, nrm):
    m = n.copy()
    p = m.nodes[["x", "y", "z"]].values.astype(float)
    d = (p - c) @ nrm
    p2 = p - 2 * np.outer(d, nrm)
    m.nodes[["x", "y", "z"]] = p2
    return m


def main():
    t0 = time.time()
    groups = load_groups()
    sk_by = pickle.load(open(SCACHE, "rb"))
    lines, rows, panels = [], [], {}
    for g, cells in groups.items():
        outer = [r for r, c in cells if c.startswith("R1")]
        inner = [r for r, c in cells if not c.startswith("R1")]
        neu = {r: prep(sk_by[int(r)]) for r in outer + inner if int(r) in sk_by}
        pts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
        c = pts.mean(0)
        axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        u = np.cross(axis, [0, 0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(axis, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        dps = {r: navis.make_dotprops(neu[r], k=5) for r in outer}
        orig = navis.NeuronList([dps[r] for r in outer])

        best = None
        for th in THETAS:
            nrm = np.cos(np.radians(th)) * u + np.sin(np.radians(th)) * v
            refl = navis.NeuronList([navis.make_dotprops(reflect(neu[r], c, nrm), k=5) for r in outer])
            S = navis.nblast(refl, orig, scores="mean", normalized=True, n_cores=1, progress=False).values
            np.fill_diagonal(S, np.nan)
            score = np.nanmax(S, axis=1).mean()
            if best is None or score > best[0]:
                best = (score, th, S.copy(), nrm)
        score, th, S, nrm = best
        base = navis.nblast(orig, orig, scores="mean", normalized=True, n_cores=1, progress=False).values
        np.fill_diagonal(base, np.nan)

        # mutual best mirror partners
        bm = {i: int(np.nanargmax(S[i])) for i in range(len(outer))}
        mutual = sorted({tuple(sorted((i, j))) for i, j in bm.items() if bm.get(j) == i})
        # does R7/R8 lie in the plane?
        r78 = [r for r in inner if r in neu]
        ang = None
        if r78:
            c78 = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in r78]).mean(0)
            dvec = c78 - c
            dvec -= dvec @ axis * axis
            if np.linalg.norm(dvec) > 1e-6:
                ang = float(np.degrees(np.arccos(abs(dvec / np.linalg.norm(dvec) @ nrm))))
        lines.append(f"group {g}: best mirror plane theta={th} deg, mean best-match {score:.3f}; "
                     f"mutual mirror pairs {len(mutual)}/3; in-place baseline median {np.nanmedian(base):.3f}; "
                     f"R7/R8 direction vs plane normal: {ang:.0f} deg (90 = R7/R8 in plane)" if ang is not None else
                     f"group {g}: best mirror plane theta={th} deg, mean best-match {score:.3f}; "
                     f"mutual mirror pairs {len(mutual)}/3; in-place baseline median {np.nanmedian(base):.3f}")
        for i, j in mutual:
            oth = [S[i, k] for k in range(len(outer)) if k not in (i, j)] + \
                  [S[j, k] for k in range(len(outer)) if k not in (i, j)]
            lines.append(f"   {outer[i]} <-> {outer[j]}   mirror {S[i, j]:.3f} / {S[j, i]:.3f}   "
                         f"in-place {base[i, j]:.3f}   next-best mirror {max(oth):.3f}")
            rows.append([g, outer[i], outer[j], round(float(S[i, j]), 3), round(float(S[j, i]), 3),
                         round(float(base[i, j]), 3), round(float(max(oth)), 3), th])
        unpaired = [outer[i] for i in range(len(outer)) if not any(i in p for p in mutual)]
        if unpaired:
            lines.append(f"   unpaired: {', '.join(unpaired)}")
        print(lines[-1 - len(mutual) - (1 if unpaired else 0)], flush=True)
        for ln in lines[-len(mutual) - (1 if unpaired else 0):]:
            print(ln, flush=True)
        panels[g] = (outer, neu, c, axis, u, v, nrm, mutual, r78)

    pd.DataFrame(rows, columns=["group", "cell_a", "cell_b", "mirror_ab", "mirror_ba", "in_place",
                                "next_best_mirror", "theta_deg"]).to_csv(
        os.path.join(OUT, "faab_pr_groups_mirror.tsv"), sep="\t", index=False)
    open(os.path.join(OUT, "faab_pr_groups_mirror.txt"), "w").write("\n".join(lines) + "\n")

    # figure: each group viewed down its bundle axis, mirror pairs coloured, plane drawn
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = ["#d62728", "#1f77b4", "#2ca02c"]
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for ax, (g, (outer, neu, c, axis, u, v, nrm, mutual, r78)) in zip(axes.ravel(), panels.items()):
            colour = {}
            for k, (i, j) in enumerate(mutual):
                colour[i] = colour[j] = cols[k % 3]
            for i, r in enumerate(outer):
                p = neu[r].nodes[["x", "y", "z"]].values - c
                ax.scatter(p @ u, p @ v, s=2, color=colour.get(i, "#999999"), alpha=0.6,
                           label=f"{r[-6:]}" + (" (pair)" if i in colour else ""))
            for r in r78:
                p = neu[r].nodes[["x", "y", "z"]].values - c
                ax.scatter(p @ u, p @ v, s=2, color="#ff9900", alpha=0.5)
            L = 60
            t = np.array([-nrm @ v, nrm @ u])  # in-plane direction perpendicular to normal
            ax.plot([-L * t[0], L * t[0]], [-L * t[1], L * t[1]], "k--", lw=1)
            ax.set_aspect("equal")
            ax.set_title(f"group {g}: view down bundle axis; {len(mutual)}/3 mirror pairs", fontsize=10)
            ax.legend(fontsize=6, markerscale=4, loc="upper right")
            ax.set_xlabel("um")
        fig.suptitle("R1-R6 per ommatidium, projected along the axon-bundle axis; dashed = best mirror plane; "
                     "orange = R7/R8; grey = unpaired", fontsize=11)
        fig.savefig(os.path.join(OUT, "faab_pr_groups_mirror.png"), dpi=120, bbox_inches="tight")
        print("saved figure", flush=True)
    except Exception as e:
        print("figure failed:", str(e)[:80])
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
