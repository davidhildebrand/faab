"""Map the in-situ mirror pairs of each photoreceptor group onto the fly pairings.

Per group: refit the mirror plane (as in faab_pr_groups_mirror.py), then give each cell a
cross-section position: t = mean coordinate along the plane (perpendicular to the bundle
axis), d = mean signed distance from the plane. Mirror pairs have similar t and opposite d.
Self-mirror = NBLAST of a cell against its own reflection (high = the cell itself lies on
the plane). Fly-name recipe (assumption stated in the output):
   lateral pair  (largest mean |d|)                   -> R2/R5
   end pair whose t is nearest the R7/R8 axons        -> R1/R6  (R7 sits between R1 and R6)
   remaining end pair                                 -> R3/R4
Outputs figures/faab_pr_groups_flynames.tsv and prints a per-group table. Local only.
"""
import os
import pickle
import time
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis

from faab.pr.groups_mirror import load_groups, prep, reflect, THETAS, SCACHE, OUT
from faab.config import fig

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)


def table(headers, rows, aligns):
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = [" | ".join(f"{headers[i]:{aligns[i]}{w[i]}}" for i in range(len(headers))),
           "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):{aligns[i]}{w[i]}}" for i in range(len(headers))))
    return "\n".join(out)


def main():
    t0 = time.time()
    groups = load_groups()
    sk_by = pickle.load(open(SCACHE, "rb"))
    all_rows = []
    for g, cells in groups.items():
        outer = [r for r, c in cells if c.startswith("R1")]
        inner = [(r, c) for r, c in cells if not c.startswith("R1")]
        neu = {r: prep(sk_by[int(r)]) for r, _ in cells if int(r) in sk_by}
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
            Sd = S.copy()
            np.fill_diagonal(Sd, np.nan)
            score = np.nanmax(Sd, axis=1).mean()
            if best is None or score > best[0]:
                best = (score, th, S.copy(), nrm)
        score, th, S, nrm = best
        selfm = np.diag(S).copy()
        np.fill_diagonal(S, np.nan)
        that = np.cross(axis, nrm)
        that /= np.linalg.norm(that)

        def pos(r):
            p = neu[r].nodes[["x", "y", "z"]].values - c
            return float((p @ that).mean()), float((p @ nrm).mean())

        P = {r: pos(r) for r in outer}
        p78 = [pos(r) for r, _ in inner if r in neu]
        t78 = np.mean([p[0] for p in p78]) if p78 else np.nan
        d78 = np.mean([p[1] for p in p78]) if p78 else np.nan

        bm = {i: int(np.nanargmax(S[i])) for i in range(len(outer))}
        pairs = sorted({tuple(sorted((i, j))) for i, j in bm.items() if bm.get(j) == i})
        left = [i for i in range(len(outer)) if not any(i in p for p in pairs)]
        note_extra = ""
        if len(left) == 2:
            i, j = left
            on_plane = all(abs(P[outer[k]][1]) < 2.5 and selfm[k] > S[k, l] for k, l in ((i, j), (j, i)))
            if on_plane:
                note_extra = (f"cells {outer[i]} and {outer[j]} sit ON the plane (|d|<2.5 um, self-mirror "
                              f"{selfm[i]:.2f}/{selfm[j]:.2f} beats their mutual {S[i, j]:.2f}); no third pair")
            else:
                pairs.append((i, j))
                note_extra = (f"third pair {outer[i]}/{outer[j]} taken as the leftover two "
                              f"(mutual mirror {S[i, j]:.2f}, not mutual-best)")
        # fly-name recipe
        names = {}
        if len(pairs) == 3:
            lat = max(pairs, key=lambda p: (abs(P[outer[p[0]]][1]) + abs(P[outer[p[1]]][1])) / 2)
            ends = [p for p in pairs if p != lat]
            tmean = {p: (P[outer[p[0]]][0] + P[outer[p[1]]][0]) / 2 for p in ends}
            if np.isnan(t78):
                near = min(ends, key=lambda p: tmean[p])
            else:
                near = min(ends, key=lambda p: abs(tmean[p] - t78))
            far = [p for p in ends if p != near][0]
            for k in lat:
                names[k] = "R2/R5 (lateral)"
            for k in near:
                names[k] = "R1/R6 (end nearest R7/R8)"
            for k in far:
                names[k] = "R3/R4 (end away from R7/R8)"
        rows = []
        for i, r in enumerate(outer):
            partner = next((outer[p[1] if p[0] == i else p[0]] for p in pairs if i in p), "")
            ps = next((S[p[0], p[1]] for p in pairs if i in p), np.nan)
            rows.append([r, f"{P[r][0]:.1f}", f"{P[r][1]:+.1f}", f"{selfm[i]:.2f}", partner,
                         "" if np.isnan(ps) else f"{ps:.2f}", names.get(i, "unassigned")])
            all_rows.append([g] + rows[-1])
        print(f"\n=== group {g}: plane theta={th} deg, R7/R8 at t={t78:.1f} um, d={d78:+.1f} um "
              f"(d~0 means R7/R8 lie in the plane) ===")
        print(table(["cell", "t_um", "d_um", "self_mirror", "partner", "pair_score", "proposed fly pair"],
                    rows, ["<", ">", ">", ">", "<", ">", "<"]))
        if note_extra:
            print("  " + note_extra)
    pd.DataFrame(all_rows, columns=["group", "cell", "t_um", "d_um", "self_mirror", "partner",
                                    "pair_score", "proposed_fly_pair"]).to_csv(
        fig("faab_pr_groups_flynames.tsv"), sep="\t", index=False)
    print(f"\nassumptions: R2/R5 = pair farthest from the plane; R1/R6 = straddling pair nearest the R7/R8 "
          f"axons (R7 is recruited between R1 and R6 in the fly); R3/R4 = the other straddling pair.")
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
