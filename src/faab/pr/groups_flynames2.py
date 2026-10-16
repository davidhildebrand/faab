"""Terminal-zone version of the mirror-pair / fly-name analysis.

Same as faab_pr_groups_flynames.py but every cell is first cut down to its lamina
terminal: along the group's bundle axis the terminal side is the side carrying the branch
points, and the zone is the terminal-most ZONE_UM along that axis (group-wide end). The
mirror plane is then fitted on terminal points only, pairs and cross-section positions
(t along the plane, d signed distance from it) come from terminals only, and the fly-name
recipe is applied: lateral pair -> R2/R5, straddling pair nearest R7/R8 -> R1/R6,
other straddling pair -> R3/R4. Output: figures/faab_pr_groups_flynames_terminal.tsv
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
from faab.pr.groups_flynames import table

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
ZONE_UM = 35.0


def terminal_subset(n, c, axis, s_end, sign):
    p = n.nodes[["x", "y", "z"]].values - c
    s = sign * (p @ axis)
    keep = n.nodes.node_id.values[s > s_end - ZONE_UM]
    if len(keep) < 10:
        return n
    return navis.subset_neuron(n, keep, inplace=False)


def main():
    t0 = time.time()
    groups = load_groups()
    sk_by = pickle.load(open(SCACHE, "rb"))
    all_rows = []
    for g, cells in groups.items():
        outer = [r for r, cc in cells if cc.startswith("R1")]
        inner = [r for r, cc in cells if not cc.startswith("R1")]
        full = {r: prep(sk_by[int(r)]) for r, _ in cells if int(r) in sk_by}
        pts = np.vstack([full[r].nodes[["x", "y", "z"]].values for r in outer])
        c = pts.mean(0)
        axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        # terminal (lamina) side = the low-x end in this eye region (retina sits at high x)
        s_pool = (pts - c) @ axis
        x_hi = pts[s_pool > np.percentile(s_pool, 80), 0].mean()
        x_lo = pts[s_pool < np.percentile(s_pool, 20), 0].mean()
        sign = 1.0 if x_hi < x_lo else -1.0
        s_end = max((sign * ((full[r].nodes[["x", "y", "z"]].values - c) @ axis)).max() for r in outer)
        neu = {r: terminal_subset(full[r], c, axis, s_end, sign) for r in full}
        tpts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
        ct = tpts.mean(0)
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
            refl = navis.NeuronList([navis.make_dotprops(reflect(neu[r], ct, nrm), k=5) for r in outer])
            S = navis.nblast(refl, orig, scores="mean", normalized=True, n_cores=1, progress=False).values
            Sd = S.copy()
            np.fill_diagonal(Sd, np.nan)
            score = np.nanmax(Sd, axis=1).mean()
            if best is None or score > best[0]:
                best = (score, th, S.copy(), nrm)
        score, th, S, nrm = best
        selfm = np.diag(S).copy()
        np.fill_diagonal(S, np.nan)
        base = navis.nblast(orig, orig, scores="mean", normalized=True, n_cores=1, progress=False).values
        np.fill_diagonal(base, np.nan)
        that = np.cross(axis, nrm)
        that /= np.linalg.norm(that)

        def pos(r):
            p = neu[r].nodes[["x", "y", "z"]].values - ct
            return float((p @ that).mean()), float((p @ nrm).mean())

        P = {r: pos(r) for r in outer}
        p78 = [pos(r) for r in inner if r in neu]
        t78 = np.mean([p[0] for p in p78]) if p78 else np.nan
        d78 = np.mean([p[1] for p in p78]) if p78 else np.nan

        bm = {i: int(np.nanargmax(S[i])) for i in range(len(outer))}
        pairs = sorted({tuple(sorted((i, j))) for i, j in bm.items() if bm.get(j) == i})
        left = [i for i in range(len(outer)) if not any(i in p for p in pairs)]
        note = ""
        if len(left) == 2:
            i, j = left
            pairs.append((i, j))
            note = f"third pair = leftover {outer[i]}/{outer[j]} (mutual mirror {S[i, j]:.2f}, not mutual-best)"
        names, conf = {}, ""
        if len(pairs) == 3:
            dmean = {p: (abs(P[outer[p[0]]][1]) + abs(P[outer[p[1]]][1])) / 2 for p in pairs}
            lat = max(pairs, key=lambda p: dmean[p])
            ends = [p for p in pairs if p != lat]
            tmean = {p: (P[outer[p[0]]][0] + P[outer[p[1]]][0]) / 2 for p in ends}
            near = min(ends, key=lambda p: abs(tmean[p] - t78)) if not np.isnan(t78) else ends[0]
            far = [p for p in ends if p != near][0]
            for k in lat:
                names[k] = "R2/R5"
            for k in near:
                names[k] = "R1/R6"
            for k in far:
                names[k] = "R3/R4"
            second = sorted(dmean.values())[-2]
            conf = (f"lateral margin {dmean[lat] - second:.1f} um; end-pair t separation "
                    f"{abs(tmean[near] - tmean[far]):.1f} um; R7/R8 at t={t78:.1f}, d={d78:+.1f}")
        rows = []
        for i, r in enumerate(outer):
            partner = next((outer[p[1] if p[0] == i else p[0]] for p in pairs if i in p), "")
            ps = next((S[p[0], p[1]] for p in pairs if i in p), np.nan)
            nb = np.nanmax([S[i, k] for k in range(len(outer)) if k != i and not any(i in p and k in p for p in pairs)] or [np.nan])
            rows.append([r, neu[r].n_nodes, f"{P[r][0]:+.1f}", f"{P[r][1]:+.1f}", partner,
                         "" if np.isnan(ps) else f"{ps:.2f}", "" if np.isnan(nb) else f"{nb:.2f}",
                         names.get(i, "")])
            all_rows.append([g] + rows[-1])
        print(f"\n=== group {g}: terminal zone {ZONE_UM:.0f} um, plane theta={th} deg, mean best-match {score:.2f}, "
              f"in-place baseline {np.nanmedian(base):.2f} ===")
        print(table(["cell", "nodes", "t_um", "d_um", "partner", "pair", "next_best", "fly pair"],
                    rows, ["<", ">", ">", ">", "<", ">", ">", "<"]))
        if note:
            print("  " + note)
        if conf:
            print("  " + conf)
    pd.DataFrame(all_rows, columns=["group", "cell", "terminal_nodes", "t_um", "d_um", "partner",
                                    "pair_score", "next_best", "proposed_fly_pair"]).to_csv(
        os.path.join(OUT, "faab_pr_groups_flynames_terminal.tsv"), sep="\t", index=False)
    print(f"\nDONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
