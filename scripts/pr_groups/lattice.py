"""Step 1 of the automatic photoreceptor-group finder: surface + lattice fit to the known cartridges.

Known right-side cartridges: David's 8 groups (tmp/pr_groups_pasted.tsv, skeletons cached) and LRS's SeaTable
cartridges (aedes_main rows typed R1-6 / R7-8, side R, notes "LRS: cartridge N"; roots via supervoxels, skeletons
fetched once). Per cartridge: pooled PCA axis of the R1-R6 skeletons oriented retina -> lamina; the lamina point is
the median of the R1-R6 presynaptic sites (figures/pr_metu_down_synapses.tsv; R1-R6 output synapses are in the
lamina, and the R7 presynaptic sites lie ~20 um further along +axis, in the medulla, which confirms the sign); the
retina point = centroid of the first 35 um at the other end. Without synapses a low-x rule is used as fallback, but
the x component of these axons is small (they run mostly along -z/+y), so that fallback is unreliable (it inverted
the axes of 11/13 cartridges on the first run). Fit set = largest 30-um single-linkage cluster of lamina points.
Fits: plane (PCA) and quadric w = f(u, v) to the lamina points; hexagonal lattice (spacing s, rotation theta,
offset) fitted to the in-plane coordinates by grid search; predicted = empty lattice sites inside the known hull
plus RINGS rings beyond it, mapped back to 3D through the quadric, with the axis interpolated from the known
cartridges (inverse-distance weights).
Outputs: figures/pr_lattice_known.tsv, pr_lattice_predicted.tsv, pr_lattice.png, pr_lattice_ng.txt (EM, brain
mesh, known PRs, known lamina points green, predicted sites yellow). Read-only, throttled.
"""
from faab.config import REPO
import datetime
import itertools
import json
import os
import pickle
import re
import time
import urllib.parse

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from scipy.spatial import ConvexHull, Delaunay
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX
import faab.nglink as nglink

HERE = REPO
OUT = os.path.join(HERE, "figures")
TMP = os.path.join(HERE, "..", "tmp")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
ZONE_UM = 35.0
RINGS = 2
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def nodes_um(sk):
    return sk.nodes[["x", "y", "z"]].values.astype(float) / 1000.0


def cartridge_geometry(skels_r16, syn_um=None):
    """pooled axis (unit, retina -> lamina), lamina point, retina point (all um) from the R1-R6 skeletons.
    Orientation: the lamina is where the R1-R6 cells make their output synapses (syn_um = presynaptic positions
    of these cells, from the bridge search); without synapses fall back to the low-x end rule."""
    pts = np.vstack([nodes_um(s) for s in skels_r16])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    s = (pts - c) @ axis
    if syn_um is not None and len(syn_um) >= 5:
        if (np.median(syn_um, axis=0) - c) @ axis < 0:
            axis, s = -axis, -s
        lam = np.median(syn_um, axis=0)
        how = f"{len(syn_um)} presynaptic sites"
    else:
        x_hi = pts[s > np.percentile(s, 80), 0].mean()
        x_lo = pts[s < np.percentile(s, 20), 0].mean()
        if x_hi < x_lo:
            axis, s = -axis, -s
        lam = pts[s > s.max() - ZONE_UM].mean(0)
        how = "low-x terminal zone"
    ret = pts[s < s.min() + ZONE_UM].mean(0)
    return axis, lam, ret, how


def main():
    t0 = time.time()
    sk = pickle.load(open(SCACHE, "rb"))
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())

    # ---- David's groups ----
    g = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    g["class"] = g["class"].str.replace("–", "-")
    carts = {}
    for grp, sub in g.groupby("group"):
        carts[f"g{grp}"] = dict(r16=[int(r) for r in sub[sub["class"] == "R1-R6"].root_id_curr],
                                r78=[int(r) for r in sub[sub["class"].isin(["R7", "R8"])].root_id_curr], source="David")

    # ---- LRS cartridges from SeaTable (right side) ----
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    typ = meta["type"].astype(object).map(_s)
    pr = meta[typ.isin(["R1-6", "R7-8"]) & (meta["side"].astype(object).map(_s) == "R")].copy()
    pr["cart"] = pr["notes"].astype(object).map(_s).str.extract(r"cartridge\s*(\d+)")[0]
    pr["type"] = typ[pr.index]
    sv = [int(_s(v)) for v in pr["supervoxel_id"]]
    t = throttle(t)
    roots = [int(x) for x in np.asarray(client.chunkedgraph.get_roots(sv))]
    pr["root"] = roots
    print(f"SeaTable right-side PRs: {len(pr)}; cartridges {sorted(pr.cart.dropna().unique(), key=int)}", flush=True)
    pr[["root", "type", "cart", "notes"]].assign(notes=pr["notes"].astype(object).map(_s)).to_csv(
        os.path.join(OUT, "pr_lattice_seatable_prs.tsv"), sep="\t", index=False)  # for the probe: already-typed PRs
    for cnum, sub in pr.dropna(subset=["cart"]).groupby("cart"):
        carts[f"c{cnum}"] = dict(r16=[int(r) for r in sub[sub.type == "R1-6"].root],
                                 r78=[int(r) for r in sub[sub.type == "R7-8"].root], source="LRS")
    need = sorted({r for c in carts.values() for r in c["r16"] + c["r78"] if r not in sk})
    print(f"skeletons to fetch: {len(need)}", flush=True)
    if need:
        t = throttle(t)
        for s in flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
            sk[int(s.id)] = s
        pickle.dump(sk, open(SCACHE, "wb"))

    # ---- presynaptic sites of the known PRs (bridge search output): orientation + lamina point ----
    syn_by_root = {}
    syn_path = os.path.join(OUT, "pr_metu_down_synapses.tsv")
    if os.path.exists(syn_path):
        d = pd.read_csv(syn_path, sep="\t", dtype={"pre_pt_root_id": str})
        for rid, sub in d.groupby("pre_pt_root_id"):
            syn_by_root[int(rid)] = np.array([[float(v) for v in str(s).strip("[]").split()] for s in sub.pre_pt_position]) / 1000.0

    # ---- per-cartridge geometry ----
    rows = []
    for name, c in carts.items():
        skels = [sk[r] for r in c["r16"] if r in sk]
        if len(skels) < 2:
            print(f"  {name}: only {len(skels)} R1-R6 skeletons, skipped", flush=True)
            continue
        syn = [syn_by_root[r] for r in c["r16"] if r in syn_by_root]
        syn_um = np.vstack(syn) if syn else None
        axis, lam, ret, how = cartridge_geometry(skels, syn_um)
        rows.append(dict(cartridge=name, source=c["source"], n_r16=len(skels), n_r78=len(c["r78"]), lamina_from=how,
                         lam_x=lam[0], lam_y=lam[1], lam_z=lam[2], ret_x=ret[0], ret_y=ret[1], ret_z=ret[2],
                         ax_x=axis[0], ax_y=axis[1], ax_z=axis[2], length_um=float(np.linalg.norm(lam - ret)),
                         r16_ids=",".join(map(str, c["r16"])), r78_ids=",".join(map(str, c["r78"]))))
    known = pd.DataFrame(rows)
    Lall = known[["lam_x", "lam_y", "lam_z"]].values
    # fit set = largest spatially connected cluster of cartridges (single linkage, LINK_UM); the rest are kept
    # as known cells but are too far across the curved eye for one surface/lattice
    LINK_UM = 30.0
    Dm = np.linalg.norm(Lall[:, None] - Lall[None], axis=2)
    comp = -np.ones(len(known), int)
    for i in range(len(known)):
        if comp[i] >= 0:
            continue
        stack, comp[i] = [i], i
        while stack:
            k = stack.pop()
            for j in np.where((Dm[k] < LINK_UM) & (comp < 0))[0]:
                comp[j] = i
                stack.append(j)
    big = pd.Series(comp).value_counts().idxmax()
    known["in_fit"] = comp == big
    print(f"{len(known)} cartridges with geometry; {int(known.in_fit.sum())} in the connected fit cluster, outliers: "
          f"{known[~known.in_fit].cartridge.tolist()}", flush=True)
    print(known[["cartridge", "lamina_from", "length_um"]].to_string(index=False), flush=True)
    fit = known[known.in_fit].reset_index(drop=True)
    L = fit[["lam_x", "lam_y", "lam_z"]].values
    A = fit[["ax_x", "ax_y", "ax_z"]].values
    print(f"fit cluster: retina->lamina length median {fit.length_um.median():.0f} um; axis spread (deg from mean): "
          f"{np.degrees(np.arccos(np.clip(A @ (A.mean(0) / np.linalg.norm(A.mean(0))), -1, 1))).round(0).tolist()}; "
          f"nearest-neighbour lamina distances (um): {np.sort(np.where(np.eye(len(L), dtype=bool), np.inf, np.linalg.norm(L[:, None] - L[None], axis=2)).min(1)).round(1).tolist()}", flush=True)

    # ---- plane and quadric through the lamina points ----
    c0 = L.mean(0)
    U, S, Vt = np.linalg.svd(L - c0, full_matrices=False)
    e1, e2, n = Vt[0], Vt[1], Vt[2]
    if n @ A.mean(0) < 0:  # plane normal roughly along the bundle axis, pointing lamina-ward
        n = -n
    uv = (L - c0) @ np.vstack([e1, e2]).T
    w = (L - c0) @ n
    print(f"plane fit: RMS residual {np.sqrt(np.mean(w ** 2)):.1f} um; angle between plane normal and mean axis "
          f"{np.degrees(np.arccos(abs(n @ A.mean(0)) / np.linalg.norm(A.mean(0)))):.0f} deg", flush=True)
    Q = np.column_stack([np.ones(len(uv)), uv[:, 0], uv[:, 1], uv[:, 0] ** 2, uv[:, 0] * uv[:, 1], uv[:, 1] ** 2])
    coef, *_ = np.linalg.lstsq(Q, w, rcond=None)
    wq = Q @ coef
    print(f"quadric fit: RMS residual {np.sqrt(np.mean((w - wq) ** 2)):.1f} um (coefficients {coef.round(4).tolist()})", flush=True)

    # ---- hexagonal lattice in (u, v): grid search over spacing and rotation, offset from the best point ----
    def lattice_residual(s, th, uv):
        a1 = s * np.array([np.cos(th), np.sin(th)])
        a2 = s * np.array([np.cos(th + np.pi / 3), np.sin(th + np.pi / 3)])
        B = np.column_stack([a1, a2])
        Binv = np.linalg.inv(B)
        best = None
        for k in range(len(uv)):  # offset candidates: each known point sits on a lattice site
            rel = uv - uv[k]
            idx = np.round(rel @ Binv.T)
            res = rel - idx @ B.T
            r = np.sqrt((res ** 2).sum(1)).mean()
            n_collide = len(idx) - len({tuple(i) for i in idx})  # two known cartridges on one site is impossible
            r += n_collide * s
            if best is None or r < best[0]:
                best = (r, uv[k], idx)
        return best

    # physical lower bound on the spacing: two cartridges cannot sit closer than one lattice spacing, so the smallest
    # nearest-neighbour distance among the known lamina points bounds s from below (15 % slack for noise). Without
    # this bound the relative-residual criterion drifts to sub-harmonic lattices (s / sqrt(3), s / 2) that fit any
    # point set and predict sites between adjacent known cartridges.
    nn_min = float(np.where(np.eye(len(uv), dtype=bool), np.inf, np.linalg.norm(uv[:, None] - uv[None], axis=2)).min())
    s_lo = max(3.0, 0.85 * nn_min)
    grid = []
    for s in np.arange(s_lo, 26.0, 0.25):
        for th in np.radians(np.arange(0, 60, 1.0)):
            r, off, idx = lattice_residual(s, th, uv)
            grid.append((r / s, r, s, th, off))
    grid.sort(key=lambda x: x[0])
    rel_res, res, s_best, th_best, off = grid[0]
    print(f"lattice fit (spacing searched from {s_lo:.2f} um = 0.85 x smallest in-plane neighbour distance {nn_min:.2f}): "
          f"spacing {s_best:.2f} um, rotation {np.degrees(th_best):.0f} deg, mean residual {res:.2f} um ({rel_res:.2f} of the "
          f"spacing); runners-up (spacing, rotation, relative residual): "
          f"{[(round(x[2], 2), int(round(np.degrees(x[3]))), round(x[0], 3)) for x in grid[1:6]]}", flush=True)
    a1 = s_best * np.array([np.cos(th_best), np.sin(th_best)])
    a2 = s_best * np.array([np.cos(th_best + np.pi / 3), np.sin(th_best + np.pi / 3)])
    B = np.column_stack([a1, a2])
    known_idx = np.round((uv - off) @ np.linalg.inv(B).T).astype(int)
    for col in ("lat_i", "lat_j", "lat_res_um"):
        known[col] = np.nan
    known.loc[known.in_fit, "lat_i"] = known_idx[:, 0]
    known.loc[known.in_fit, "lat_j"] = known_idx[:, 1]
    known.loc[known.in_fit, "lat_res_um"] = np.sqrt((((uv - off) - known_idx @ B.T) ** 2).sum(1)).round(2)
    fit = known[known.in_fit].reset_index(drop=True)
    occupied = {tuple(k) for k in known_idx}
    if len(occupied) < len(known_idx):
        print("WARNING: lattice collision remains among known cartridges", flush=True)

    # ---- predicted sites: inside the hull of known sites + RINGS rings ----
    hull_pts = uv
    try:
        hull = Delaunay(hull_pts)
    except Exception:
        hull = None
    cand = set()
    for (i, j) in occupied:
        for di in range(-RINGS, RINGS + 1):
            for dj in range(-RINGS, RINGS + 1):
                if max(abs(di), abs(dj), abs(di + dj)) <= RINGS:  # hexagonal ring distance
                    cand.add((i + di, j + dj))
    cand -= occupied
    pred = []
    for (i, j) in sorted(cand):
        p_uv = off + np.array([i, j]) @ B.T
        q = np.array([1, p_uv[0], p_uv[1], p_uv[0] ** 2, p_uv[0] * p_uv[1], p_uv[1] ** 2])
        p3 = c0 + p_uv[0] * e1 + p_uv[1] * e2 + float(q @ coef) * n
        d = np.linalg.norm(L - p3, axis=1)
        wgt = 1.0 / np.maximum(d, 1e-3) ** 2
        ax = (A * wgt[:, None]).sum(0)
        ax /= np.linalg.norm(ax)
        inside = bool(hull is not None and hull.find_simplex(p_uv) >= 0)
        pred.append(dict(lat_i=i, lat_j=j, inside_hull=inside, nearest_known_um=round(float(d.min()), 1),
                         lam_x=p3[0], lam_y=p3[1], lam_z=p3[2], ax_x=ax[0], ax_y=ax[1], ax_z=ax[2]))
    pred = pd.DataFrame(pred)
    for df_ in (known, pred):
        for a, v in zip("xyz", VOX):
            df_[f"lam_vox_{a}"] = (df_[f"lam_{a}"] * 1000.0 / v).round(0).astype(int)
    pred["ring"] = [min(max(abs(i - ki), abs(j - kj), abs((i + j) - (ki + kj))) for ki, kj in occupied) for i, j in zip(pred.lat_i, pred.lat_j)]
    pred = pred.sort_values(["inside_hull", "ring", "nearest_known_um"], ascending=[False, True, True])
    pred.to_csv(os.path.join(OUT, "pr_lattice_predicted.tsv"), sep="\t", index=False)
    known.to_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t", index=False)
    print(f"predicted sites: {len(pred)} ({int(pred.inside_hull.sum())} inside the known hull, "
          f"{int((pred.ring == 1).sum())} in ring 1, {int((pred.ring == 2).sum())} in ring 2)", flush=True)
    pd.set_option("display.width", 220)
    print(known[["cartridge", "source", "n_r16", "n_r78", "lam_vox_x", "lam_vox_y", "lam_vox_z", "length_um", "lat_i", "lat_j", "lat_res_um"]].to_string(index=False))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        puv = (pred[["lam_x", "lam_y", "lam_z"]].values - c0) @ np.vstack([e1, e2]).T
        ax.scatter(puv[:, 0], puv[:, 1], s=60, facecolors="none", edgecolors="goldenrod", label="predicted site")
        ax.scatter(uv[:, 0], uv[:, 1], s=70, c=["tab:green" if s_ == "David" else "tab:blue" for s_ in fit.source], label="known (green David, blue LRS)")
        for k, r in fit.iterrows():
            ax.annotate(r.cartridge, (uv[k, 0], uv[k, 1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xlabel("u (um, in lamina plane)")
        ax.set_ylabel("v (um)")
        ax.set_title(f"lamina points of known cartridges + fitted hexagonal lattice (spacing {s_best:.1f} um)")
        ax.legend(loc="best", fontsize=8)
        fig.savefig(os.path.join(OUT, "pr_lattice.png"), dpi=130, bbox_inches="tight")
    except Exception as e:
        print("figure failed:", str(e)[:100])

    # ---- link: EM, brain mesh, known PR segments, known lamina points, predicted sites ----
    all_known_ids = sorted({str(r) for c in carts.values() for r in c["r16"] + c["r78"]})
    spec = {"meshes": [{"kind": "brain", "alpha": 0.15}],
            "neurons": [{"name": "known PRs (8 groups + LRS cartridges)", "color": "#ff9f1c", "ids": all_known_ids}],
            "view": "anterior", "layout": "3d", "image": True, "scale": 12000,
            "position": [int(v) for v in (c0 * 1000.0 / VOX)]}
    state = nglink.build(spec)
    state["layers"][0]["name"] = "EM"
    ann_known = [{"type": "point", "id": f"k_{r.cartridge}", "point": [int(r.lam_vox_x), int(r.lam_vox_y), int(r.lam_vox_z)],
                  "description": f"{r.cartridge} lamina point ({r.source})"} for r in known.itertuples()]
    ann_pred = [{"type": "point", "id": f"p_{r.lat_i}_{r.lat_j}", "point": [int(r.lam_vox_x), int(r.lam_vox_y), int(r.lam_vox_z)],
                 "description": f"predicted cartridge site ({r.lat_i},{r.lat_j}), ring {r.ring}"} for r in pred.itertuples()]
    state["layers"] += [{"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                         "annotations": ann_known, "name": "known cartridge lamina points", "annotationColor": "#2ecc71"},
                        {"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                         "annotations": ann_pred, "name": "predicted cartridge sites", "annotationColor": "#ffe119"}]
    url = nglink.to_url(state)
    open(os.path.join(OUT, "pr_lattice_ng.txt"), "w").write(url + "\n")
    json.dump(state, open(os.path.join(OUT, "pr_lattice_state.json"), "w"), indent=1)
    print(f"link {len(url)} chars saved; DONE in {time.time() - t0:.0f} s")
    print(url)


if __name__ == "__main__":
    main()
