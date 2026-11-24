"""T4/T5 subtypes and where inputs land on their dendrites (David 2026-11-24, analysis approach 1 of the motion-
literature plan). Cells: the T4T5s tab (right optic lobe, putative T4 / T5). Read-only: local synapse snapshot
(faab.synsnap, newest checkpoint + one CAVE catch-up), cached L2 skeletons (figures/t4t5/t4t5_tab_skels.pkl, keyed
by root; rows whose current root has no cached skeleton are left out of the dendrite steps), aedes_ol_meshes.

Literature this follows: T4/T5 subtypes a-d = four lobula-plate layers = four directions (Maisak 2013,
10.1038/nature12320); axon terminals sort into four depths (Fischbach & Dittrich 1989, 10.1007/BF00218858); each
dendrite points one of four ways matching its layer (Takemura 2013, 10.1038/nature12450; Zhao 2025,
10.1038/s41586-025-09276-5); inputs sit at different positions along the dendrite's long axis (Takemura 2017,
10.7554/eLife.24394; Shinomiya 2019, 10.7554/eLife.40025).

Steps (each writes figures/t4t5/<stamp>_t4t5ld_<step>.* and reads the newest output of the step before):
  gather BOOK   cells from BOOK's T4T5s tab; every output synapse (presynaptic position) and input synapse
                (postsynaptic position) of them now, in um, with the neuropil it lies in (LA/ME/LO/LOP_R or other)
  layers        lobula-plate depth: local planes fitted to all T4/T5 output synapses in LOP_R (RADIUS), normals
                oriented toward the lobula (more LO_R volume on that side), depth = signed distance to the nearest
                plane, normalized per plane to its own 5-95% range (0 = far face, 1 = lobula side); per cell the
                median over its LOP outputs; Gaussian mixtures k = 1..7 (BIC) for T4, T5 and both; an independent check
                by output partners (cosine similarity of per-cell partner fractions, agglomerative clusters)
  dendrites     per cell with a skeleton: dendrite = its input synapses in ME_R (T4) or LO_R (T5); base = the dendrite
                skeleton node nearest the axon along the skeleton (geodesic from the node nearest the LOP output
                centroid); local tangent plane of the dendrite layer (planes fitted to all T4 resp. T5 dendrite
                inputs); dendrite vector = input centroid - base, in that plane; angle in a local frame (e1 = the
                dataset y axis projected into the plane, e2 = normal x e1); elongation and asymmetry
  inputs        position of every dendrite input along the cell's own dendrite axis: s = (p - base).u / L (u = unit
                dendrite vector, L = the 95th percentile of s over the cell's inputs; 0 = base, 1 = tips); per
                (cell, input cell) mean s; segregation per cell (eta^2 of s by input cell vs label shuffles); input
                cells seen on several dendrites: consistency of their position; named anchors (CT1)
Usage: t4t5_layers_dendrites.py gather BOOK.xlsx | layers | dendrites | inputs"""
import datetime
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.config import VOX, fig
from faab.partners.sheet import cell_text as s, utc_stamp

warnings.filterwarnings("ignore")
OUT = fig("t4t5")
SKELS = fig("t4t5_tab_skels.pkl")
RADIUS = 25.0      # um: local plane fits
NEAR_NODE = 3.0    # um: skeleton nodes this close to a dendrite input count as dendrite
MIN_LOP = 5        # LOP outputs needed for a depth
MIN_DEN = 20       # dendrite inputs needed for a dendrite vector
REGIONS = ["LA_R", "ME_R", "LO_R", "LOP_R"]
CT1 = {"648518347622660228": "CT1", "648518347494887381": "CT1", "648518347599884452": "CT1-like",
       "648518347672557281": "CT1-like", "648518347531730229": "LO ME (CT1-like)"}  # 1st two: CT1 now / before 11-24


def newest(tag):
    f = sorted(glob.glob(os.path.join(OUT, f"*_t4t5ld_{tag}")))
    if not f:
        sys.exit(f"no *_t4t5ld_{tag} in {OUT}; run the step before")
    return f[-1]


def cells_of(book):
    import openpyxl
    ws = openpyxl.load_workbook(book, read_only=True)["T4T5s"]
    cols, out = None, []
    for row in ws.iter_rows(values_only=True):
        v = [s(x) for x in row]
        if "root_id_curr" in v and "xyz" in v:
            cols = {h: i for i, h in enumerate(v) if h}
            continue
        if cols and v[cols["root_id_orig"]].isdigit():
            g = lambda k: v[cols[k]] if cols[k] < len(v) else ""  # noqa: E731
            out.append(dict(root=g("root_id_curr") or g("root_id_orig"), orig=g("root_id_orig"),
                            type=g("putative_type"), confidence=g("confidence"), status=g("status")))
    return pd.DataFrame(out).drop_duplicates("root")


def cmd_gather(book):
    from faab.clients.cave import client as cave_client
    stamp = utc_stamp()
    C = cells_of(book)
    roots = [int(r) for r in C.root]
    synsnap.refresh()
    snap, client = synsnap.Snapshot(), cave_client()
    T = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    print(f"{len(C)} cells ({C.type.value_counts().to_dict()}); synapses at {T:%Y-%m-%d %H:%M} UTC", flush=True)
    out = snap.synapses(pre=roots, when=T, client=client)
    inp = snap.synapses(post=roots, when=T, client=client)
    out, inp = out[out.pre_root != out.post_root], inp[inp.pre_root != inp.post_root]
    st = snap.static(np.unique(np.r_[out.id, inp.id].astype(np.int64)),
                     columns=["pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z"])
    k = np.array(VOX) / 1000.0
    for df, side in ((out, "pre"), (inp, "post")):
        m = df.merge(st, on="id", how="left")
        xyz = m[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(float) * k
        df[["x", "y", "z"]] = xyz
        df["region"] = ol_meshes.assign(xyz, REGIONS)
    out.to_parquet(os.path.join(OUT, f"{stamp}_t4t5ld_out.parquet"))
    inp.to_parquet(os.path.join(OUT, f"{stamp}_t4t5ld_in.parquet"))
    C.to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_cells.tsv"), sep="\t", index=False)
    print(f"outputs {len(out)} ({out.region.value_counts().to_dict()}); inputs {len(inp)} "
          f"({inp.region.value_counts().to_dict()}); written {stamp}_t4t5ld_{{out,in}}.parquet, _cells.tsv", flush=True)


def planes(points, radius=RADIUS, n_anchor=6000, seed=0):
    """local planes (centers C, unit normals N) of a sheet of points: anchors spread over the sheet (farthest-point
    style: a random subsample thinned so anchors are >= radius/3 apart), each from the points within `radius`"""
    tree = cKDTree(points)
    rng = np.random.default_rng(seed)
    cand = points[rng.choice(len(points), min(n_anchor * 5, len(points)), replace=False)]
    keep, at = [], cKDTree(cand)
    taken = np.zeros(len(cand), bool)
    for i in range(len(cand)):
        if taken[i]:
            continue
        keep.append(i)
        taken[at.query_ball_point(cand[i], radius / 3)] = True
    Cs, Ns = [], []
    for a in cand[keep]:
        q = points[tree.query_ball_point(a, radius)]
        if len(q) < 50:
            continue
        c = q.mean(0)
        w, v = np.linalg.eigh(np.cov((q - c).T))
        if w[0] > 0.5 * w[1]:
            continue
        Cs.append(c)
        Ns.append(v[:, 0])
    return np.array(Cs), np.array(Ns)


def orient(C, N, toward):
    """flip normals so each points toward the nearest vertex of the `toward` neuropil mesh (2026-11-24: probing for
    volume along the normal left most lobula-plate planes unoriented - the lobula is not straight ahead everywhere)"""
    v, _ = ol_meshes.mesh(toward)
    v = np.asarray(v, float) / 1000.0
    _, j = cKDTree(v).query(C)
    sg = np.sign(np.einsum("ij,ij->i", v[j] - C, N))
    sg[sg == 0] = 1
    return C, N * sg[:, None]


def signed_depth(x, C, N, radius=RADIUS):
    tree = cKDTree(C)
    _, j = tree.query(x)
    r = x - C[j]
    d = np.einsum("ij,ij->i", r, N[j])
    side = np.sqrt(np.maximum((r ** 2).sum(1) - d ** 2, 0))
    d[side > 1.5 * radius] = np.nan
    return d, j


def gmm_bic(x, kmax=7, seed=0):
    from sklearn.mixture import GaussianMixture
    x = np.asarray(x, float)[:, None]
    res = []
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, n_init=5, random_state=seed).fit(x)
        res.append((k, g.bic(x), g))
    return res


def cmd_layers():
    stamp = utc_stamp()
    out = pd.read_parquet(newest("out.parquet"))
    C0 = pd.read_csv(newest("cells.tsv"), sep="\t", dtype={"root": str})
    typ = dict(zip(C0.root, C0.type))
    lop = out[out.region == "LOP_R"].copy()
    P = lop[["x", "y", "z"]].to_numpy(float)
    Cp, Np = planes(P)
    Cp, Np = orient(Cp, Np, "LO_R")
    d, j = signed_depth(P, Cp, Np)
    lop["d"], lop["plane"] = d, j
    # per-plane 5/95% range of the depths of the outputs that use it -> normalized depth (0 far face, 1 lobula side)
    rng_ = lop.groupby("plane").d.quantile([0.05, 0.95]).unstack()
    lo5, hi95 = lop.plane.map(rng_[0.05]), lop.plane.map(rng_[0.95])
    lop["dn"] = (lop.d - lo5) / (hi95 - lo5)
    thick = (rng_[0.95] - rng_[0.05])
    print(f"{len(lop)} LOP outputs of {lop.pre_root.nunique()} cells; {len(Cp)} oriented planes; layer of T4/T5 "
          f"outputs 5-95% thickness median {thick.median():.1f} um (IQR {thick.quantile(.25):.1f}-{thick.quantile(.75):.1f})")
    per = lop.groupby("pre_root").agg(n_lop=("dn", "size"), depth_um=("d", "median"), depth=("dn", "median"),
                                      depth_iqr=("dn", lambda v: np.nanpercentile(v, 75) - np.nanpercentile(v, 25)))
    per = per[per.n_lop >= MIN_LOP].reset_index().rename(columns={"pre_root": "root"})
    per["root"] = per.root.astype(str)
    per["type"] = per.root.map(typ)
    per = per.dropna(subset=["depth"])
    print(f"cells with >= {MIN_LOP} LOP outputs and a depth: {len(per)} ({per.type.value_counts().to_dict()}); "
          f"per-cell spread (IQR of normalized depth) median {per.depth_iqr.median():.2f}")
    best = {}
    for name, sub in (("T4", per[per.type == "T4"]), ("T5", per[per.type == "T5"]), ("both", per)):
        res = gmm_bic(sub.depth)
        bics = {k: round(b) for k, b, _ in res}
        kb = min(res, key=lambda t: t[1])
        best[name] = kb
        mu = np.sort(kb[2].means_.ravel())
        print(f"  {name}: BIC by k {bics}; best k={kb[0]}, means {np.round(mu, 2).tolist()}")
    # histogram text (20 bins) for T4 and T5
    bins = np.linspace(-0.3, 1.3, 33)
    for name in ("T4", "T5"):
        h, _ = np.histogram(per[per.type == name].depth, bins)
        print(f"  {name} depth histogram ({bins[0]:.2f}..{bins[-1]:.2f}, step {bins[1]-bins[0]:.2f}): {h.tolist()}")
    # layer labels from the 'both' mixture, numbered from the lobula side (1 = deepest toward the lobula)
    g = best["both"][2]
    order = np.argsort(-g.means_.ravel())
    rank = {c: i + 1 for i, c in enumerate(order)}
    per["layer"] = [rank[c] for c in g.predict(per[["depth"]].to_numpy())]
    per["layer_p"] = g.predict_proba(per[["depth"]].to_numpy()).max(1)
    # output-partner check: per-cell fractions onto partners that get LOP input from >= 20 of these cells
    lp = lop[lop.pre_root.astype(str).isin(per.root)]
    e = lp.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    fan = e.groupby("post_root").pre_root.nunique()
    keep_p = fan[fan >= 20].index
    e = e[e.post_root.isin(keep_p)]
    M = e.pivot_table(index="pre_root", columns="post_root", values="n", fill_value=0)
    M = M[M.sum(1) >= 10]
    X = M.to_numpy(float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    k = best["both"][0]
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
    lay = per.set_index("root").loc[M.index.astype(str), "layer"].to_numpy()
    ari = adjusted_rand_score(lay, lab)
    ct = pd.crosstab(pd.Series(lay, name="depth layer"), pd.Series(lab, name="partner cluster"))
    print(f"output-partner check: {X.shape[0]} cells x {X.shape[1]} partners (>= 20 of these cells each); "
          f"{k} agglomerative clusters vs depth layers: ARI {ari:.2f}\n{ct}")
    # partner layer specificity: depth spread of each big partner's T4/T5 inputs
    dep = per.set_index("root").depth
    pl = lp.assign(dc=lp.pre_root.astype(str).map(dep)).groupby("post_root").agg(
        n_cells=("pre_root", "nunique"), depth_med=("dc", "median"),
        depth_iqr=("dc", lambda v: np.nanpercentile(v, 75) - np.nanpercentile(v, 25)))
    pl = pl[pl.n_cells >= 20].sort_values("n_cells", ascending=False)
    print(f"partners with >= 20 T4/T5 inputs: {len(pl)}; IQR of their inputs' depth: median {pl.depth_iqr.median():.2f} "
          f"(all cells' IQR {np.nanpercentile(per.depth, 75) - np.nanpercentile(per.depth, 25):.2f}); "
          f"top 10:\n{pl.head(10).round(2).to_string()}")
    per.to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_layers.tsv"), sep="\t", index=False)
    pl.reset_index().to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_layer_partners.tsv"), sep="\t", index=False)
    np.savez(os.path.join(OUT, f"{stamp}_t4t5ld_lop_planes.npz"), C=Cp, N=Np)
    print(f"written {stamp}_t4t5ld_layers.tsv, _layer_partners.tsv, _lop_planes.npz")


def geodesic_from(sk, start_idx):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    nd = sk.nodes
    idx = {n: i for i, n in enumerate(nd.node_id)}
    xyz = nd[["x", "y", "z"]].to_numpy(float) / 1000.0
    ch = [(idx[c], idx[p]) for c, p in zip(nd.node_id, nd.parent_id) if p in idx]
    if not ch:
        return np.zeros(len(nd)), xyz
    a, b = np.array(ch).T
    w = np.linalg.norm(xyz[a] - xyz[b], axis=1) + 1e-6
    G = csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(nd), len(nd)))
    return dijkstra(G, indices=start_idx), xyz


def branch_vector(S, xyz, den, bi):
    """length-weighted sum of the skeleton edges pointing away from the base inside the dendrite (BFS from the base
    through dendrite nodes): the arbor's growth direction (Takemura 2013). Fly check (fly_direction_measures.py,
    2026-11-24): separates T4a-d / T5a-d into two opposite pairs ~90 deg apart and survives 3 um node spacing; the
    centroid - base vector does not (it mostly follows where the stem enters)"""
    from collections import deque
    nd = S.nodes
    idx = {n: i for i, n in enumerate(nd.node_id)}
    adj = [[] for _ in range(len(nd))]
    for c, p in zip(nd.node_id, nd.parent_id):
        if p in idx:
            adj[idx[c]].append(idx[p])
            adj[idx[p]].append(idx[c])
    inden, seen, q, vsum = set(int(i) for i in den), {int(bi)}, deque([int(bi)]), np.zeros(3)
    while q:
        a = q.popleft()
        for b in adj[a]:
            if b not in seen and b in inden:
                seen.add(b)
                vsum += xyz[b] - xyz[a]
                q.append(b)
    return vsum, len(seen)


def cmd_dendrites():
    stamp = utc_stamp()
    out = pd.read_parquet(newest("out.parquet"))
    inp = pd.read_parquet(newest("in.parquet"))
    C0 = pd.read_csv(newest("cells.tsv"), sep="\t", dtype={"root": str})
    sk = pickle.load(open(SKELS, "rb"))
    reg = {"T4": "ME_R", "T5": "LO_R"}
    frames = {}
    for t, r in reg.items():
        ids = set(int(x) for x in C0[C0.type == t].root)
        pts = inp[(inp.region == r) & inp.post_root.isin(ids)][["x", "y", "z"]].to_numpy(float)
        Cd, Nd = planes(pts)
        Cd, Nd = orient(Cd, Nd, "LOP_R")  # normals toward the lobula plate side, for a consistent sign
        frames[t] = (Cd, Nd, cKDTree(Cd))
        print(f"{t} dendrite layer ({r}): {len(pts)} inputs, {len(Cd)} oriented planes", flush=True)
    lop_c = out[out.region == "LOP_R"].groupby("pre_root")[["x", "y", "z"]].mean()
    rows = []
    for _, c in C0.iterrows():
        t, root = c.type, c.root
        S = sk.get(root)
        den = inp[(inp.post_root == int(root)) & (inp.region == reg.get(t, ""))]
        rec = dict(root=root, type=t, n_den=len(den), skel=S is not None)
        if S is None or len(den) < MIN_DEN or int(root) not in lop_c.index or S.n_nodes < 5:
            rows.append(rec)
            continue
        P = den[["x", "y", "z"]].to_numpy(float)
        xyz = S.nodes[["x", "y", "z"]].to_numpy(float) / 1000.0
        ax = lop_c.loc[int(root)].to_numpy(float)
        start = int(np.argmin(np.linalg.norm(xyz - ax, axis=1)))
        g, xyz = geodesic_from(S, start)
        near = cKDTree(P).query(xyz)[0] <= NEAR_NODE
        if not near.any() or not np.isfinite(g[near]).any():
            rows.append(rec)
            continue
        bi = np.where(near)[0][np.nanargmin(np.where(np.isfinite(g[near]), g[near], np.inf))]
        base = xyz[bi]
        Cd, Nd, tr = frames[t]
        cen = P.mean(0)
        n = Nd[tr.query(cen)[1]]
        e1 = np.array([0.0, 1.0, 0.0]) - n[1] * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        v = cen - base
        v2 = np.array([v @ e1, v @ e2])
        Q = np.c_[(P - cen) @ e1, (P - cen) @ e2]
        w, U = np.linalg.eigh(np.cov(Q.T))
        rec.update(base_x=base[0], base_y=base[1], base_z=base[2], cen_x=cen[0], cen_y=cen[1], cen_z=cen[2],
                   n_x=n[0], n_y=n[1], n_z=n[2], v1=v2[0], v2=v2[1], vlen=float(np.hypot(*v2)),
                   angle=float(np.degrees(np.arctan2(v2[1], v2[0]))), base_off=float(abs(v @ n)),
                   sd_major=float(np.sqrt(w[1])), sd_minor=float(np.sqrt(max(w[0], 0))),
                   axis_angle=float(np.degrees(np.arctan2(U[1, 1], U[0, 1]))) % 180,
                   geo_base=float(g[bi]))
        rec["asym"] = rec["vlen"] / rec["sd_major"] if rec["sd_major"] > 0 else np.nan
        bv, n_reach = branch_vector(S, xyz, np.where(near)[0], bi)
        b2 = np.array([bv @ e1, bv @ e2])
        rec.update(b1=b2[0], b2=b2[1], blen=float(np.hypot(*b2)), n_den_nodes=n_reach,
                   bangle=float(np.degrees(np.arctan2(b2[1], b2[0]))) if np.hypot(*b2) > 0 else np.nan)
        rows.append(rec)
    D = pd.DataFrame(rows)
    ok = D.vlen.notna()
    print(f"dendrite vectors: {ok.sum()} of {len(D)} cells ({D[ok].type.value_counts().to_dict()}); vector length "
          f"median {D[ok].vlen.median():.1f} um, major-axis SD {D[ok].sd_major.median():.1f} um, asymmetry median "
          f"{D[ok].asym.median():.2f}")
    L = pd.read_csv(newest("layers.tsv"), sep="\t", dtype={"root": str})
    D = D.merge(L[["root", "depth", "layer", "layer_p"]], on="root", how="left")
    for t in ("T4", "T5"):
        sub = D[ok & (D.type == t) & D.layer.notna()]
        print(f"{t}: dendrite angle by lobula-plate layer (circular mean, resultant length R, n):")
        for lay, g in sub.groupby("layer"):
            a = np.radians(g.angle.to_numpy())
            m = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
            R = np.hypot(np.sin(a).mean(), np.cos(a).mean())
            print(f"   layer {int(lay)}: mean {m:6.1f} deg, R {R:.2f}, n {len(g)}; asymmetry median {g.asym.median():.2f}")
    D.to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_dendrites.tsv"), sep="\t", index=False)
    print(f"written {stamp}_t4t5ld_dendrites.tsv")


MIN_ASYM = 1.0     # dendrites at least this one-sided (vector length / major-axis SD) for the position step
MIN_PART = 3       # synapses of one input cell onto one dendrite to place it
N_SHUF = 200


def partner_profiles(roots, stamp):
    """neuropil profile of each input cell (all its synapses, both sides): fractions in LA/ME/LO/LOP_R; cached"""
    from faab.clients.cave import client as cave_client
    old = sorted(glob.glob(os.path.join(OUT, "*_t4t5ld_partner_profiles.tsv")))
    if old:
        P = pd.read_csv(old[-1], sep="\t", dtype={"root": str})
        if set(map(str, roots)) <= set(P.root):
            return P
    snap, client = synsnap.Snapshot(), cave_client()
    T = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    ids = [int(r) for r in roots]
    a = snap.synapses(pre=ids, when=T, client=client)
    b = snap.synapses(post=ids, when=T, client=client)
    st = snap.static(np.unique(np.r_[a.id, b.id].astype(np.int64)), columns=["pre_x", "pre_y", "pre_z",
                                                                              "post_x", "post_y", "post_z"])
    k = np.array(VOX) / 1000.0
    rows = []
    for df, side, rcol in ((a, "pre", "pre_root"), (b, "post", "post_root")):
        m = df.merge(st, on="id", how="left")
        xyz = m[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(float) * k
        rows.append(pd.DataFrame({"root": m[rcol].astype(str), "side": side,
                                  "region": ol_meshes.assign(xyz, REGIONS)}))
    R = pd.concat(rows)
    P = pd.crosstab([R.root], [R.region]).reindex(columns=REGIONS + ["none"], fill_value=0)
    n = P.sum(1)
    P = P.div(n, axis=0).add_prefix("f_")
    P["n_syn"] = n
    P = P.reset_index()
    P.to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_partner_profiles.tsv"), sep="\t", index=False)
    return P


def profile_class(r):
    if r.root in CT1:
        return CT1[r.root]
    if r.f_LA_R >= 0.15 and r.f_ME_R >= 0.3:
        return "LA+ME (C2/C3/L-like)"
    if r.f_ME_R >= 0.8 and r.f_LO_R < 0.05:
        return "ME only (Mi-like)"
    if r.f_ME_R >= 0.2 and r.f_LO_R >= 0.1:
        return "ME+LO (Tm-like)"
    if r.f_LO_R >= 0.7:
        return "LO only"
    if r.f_LOP_R >= 0.3:
        return "LOP-heavy"
    return "other"


def cmd_inputs():
    stamp = utc_stamp()
    inp = pd.read_parquet(newest("in.parquet"))
    D = pd.read_csv(newest("dendrites.tsv"), sep="\t", dtype={"root": str})
    C0 = pd.read_csv(newest("cells.tsv"), sep="\t", dtype={"root": str})
    pool = set(C0.root)
    # axis = the branch direction (2026-11-24 fly check, fly_input_positions.py: along it the fly's input types sort as
    # published - Mi9 0.65, Mi1 0.23, Mi4 -0.06, CT1 -0.02 on T4; Tm9 0.59, Tm1 0.31, CT1 0.02 on T5 - more cleanly
    # than along the centroid - base vector, which the first version used with an asymmetry cut)
    D = D[D.blen.notna() & (D.blen > 0)]
    reg = {"T4": "ME_R", "T5": "LO_R"}
    recs, syn_rows = [], []
    for _, d in D.iterrows():
        n = np.array([d.n_x, d.n_y, d.n_z])
        e1 = np.array([0.0, 1.0, 0.0]) - n[1] * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        u = (d.b1 * e1 + d.b2 * e2) / d.blen
        w = np.cross(n, u)
        base = np.array([d.base_x, d.base_y, d.base_z])
        g = inp[(inp.post_root == int(d.root)) & (inp.region == reg[d.type])]
        X = g[["x", "y", "z"]].to_numpy(float) - base
        s_raw, t_raw = X @ u, X @ w
        L = np.percentile(s_raw, 95)
        if L <= 0:
            continue
        syn_rows.append(pd.DataFrame({"cell": d.root, "type": d.type, "pre": g.pre_root.astype(str).to_numpy(),
                                      "s": s_raw / L, "t": t_raw / L}))
    S = pd.concat(syn_rows)
    print(f"{S.cell.nunique()} dendrites ({S.drop_duplicates('cell').type.value_counts().to_dict()}), {len(S)} input "
          f"synapses placed along the branch axis", flush=True)
    # per cell: do its input cells sit at different positions? (eta^2 of s by input cell, vs label shuffles)
    rng = np.random.default_rng(0)
    seg = []
    for c, g in S.groupby("cell"):
        cnt = g.pre.value_counts()
        keep = cnt[cnt >= MIN_PART].index
        h = g[g.pre.isin(keep)]
        if len(keep) < 3:
            continue
        lab = pd.factorize(h.pre)[0]
        x = h.s.to_numpy()

        def eta2(lb):
            m = np.bincount(lb, x) / np.bincount(lb)
            return (np.bincount(lb) * (m - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        e = eta2(lab)
        null = np.array([eta2(rng.permutation(lab)) for _ in range(N_SHUF)])
        seg.append(dict(cell=c, type=g.type.iloc[0], n_inputs=len(keep), n_syn=len(h), eta2=e,
                        null_med=np.median(null), p=(1 + (null >= e).sum()) / (1 + N_SHUF)))
    G = pd.DataFrame(seg)
    for t in ("T4", "T5"):
        q = G[G.type == t]
        print(f"{t}: {len(q)} dendrites with >= 3 input cells of >= {MIN_PART} synapses: eta^2 median {q.eta2.median():.2f} "
              f"vs shuffled {q.null_med.median():.2f}; p < 0.05 in {(q.p < 0.05).mean():.0%}", flush=True)
    # per (cell, input cell) mean position; input cells seen on several dendrites
    E = S.groupby(["type", "cell", "pre"]).agg(n=("s", "size"), s=("s", "mean"), t=("t", "mean")).reset_index()
    E = E[E.n >= MIN_PART]
    prof = partner_profiles(sorted(set(E.pre)), stamp)
    prof["cls"] = prof.apply(profile_class, axis=1)
    cls = dict(zip(prof.root, prof.cls))
    E["cls"] = E.pre.map(cls).fillna("other")
    E.loc[E.pre.isin(pool), "cls"] = "T4/T5 (pool)"
    for t in ("T4", "T5"):
        q = E[E.type == t]
        print(f"\n{t}: input cells by class - pairs (dendrite, input cell), mean position s (0 base, 1 tips) "
              f"median [IQR], lateral |t| median:")
        for k, g in q.groupby("cls"):
            if len(g) < (3 if k in CT1.values() else 10):
                continue
            print(f"   {k:24s} pairs {len(g):5d}, cells {g.pre.nunique():5d}: s {g.s.median():.2f} "
                  f"[{g.s.quantile(.25):.2f}-{g.s.quantile(.75):.2f}], |t| {g.t.abs().median():.2f}")
        rep = q.groupby("pre").filter(lambda x: len(x) >= 3)
        if len(rep):
            lab = pd.factorize(rep.pre)[0]
            x = rep.s.to_numpy()

            def eta2b(lb):
                m = np.bincount(lb, x) / np.bincount(lb)
                return (np.bincount(lb) * (m - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum()
            e = eta2b(lab)
            null = np.array([eta2b(rng.permutation(lab)) for _ in range(N_SHUF)])
            print(f"   consistency: {rep.pre.nunique()} input cells on >= 3 dendrites: eta^2 of position by input cell "
                  f"{e:.2f} vs shuffled {np.median(null):.2f} (p {(1 + (null >= e).sum()) / (1 + N_SHUF):.3f})")
    S.to_parquet(os.path.join(OUT, f"{stamp}_t4t5ld_input_positions.parquet"))
    E.to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_input_pairs.tsv"), sep="\t", index=False)
    G.to_csv(os.path.join(OUT, f"{stamp}_t4t5ld_segregation.tsv"), sep="\t", index=False)
    print(f"written {stamp}_t4t5ld_input_positions.parquet, _input_pairs.tsv, _segregation.tsv")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gather":
        cmd_gather(sys.argv[2])
    elif cmd == "layers":
        cmd_layers()
    elif cmd == "dendrites":
        cmd_dendrites()
    elif cmd == "inputs":
        cmd_inputs()
    else:
        sys.exit(__doc__)
