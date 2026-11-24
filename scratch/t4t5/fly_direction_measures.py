"""Which dendrite-direction measure separates fly T4/T5 subtypes (male optic lobe, known types), and does it survive
skeletons as coarse as the Aedes L2 skeletons (~3 um between nodes)? Network-free: reuses the newest
figures/t4t5/*_fly_t4t5_raw.pkl and *_fly_t4t5_check.tsv (depth) of fly_t4t5_dendrite_check.py.
Measures, all in the local dendrite-layer plane (frame as in t4t5_layers_dendrites.py):
  centroid   input-synapse centroid - base (the current Aedes measure)
  branches   length-weighted sum of skeleton edges pointing away from the base inside the dendrite (Takemura 2013:
             the dominant orientation of the arbor's nodes)
  tips       centroid of the dendrite nodes in the farthest 30% (geodesic from the base) - base
each on the full skeleton and on one resampled to RESAMPLE um. Score: within spatial patches, leave-one-out
nearest-class assignment of subtype from (angle, depth) - the share of cells whose own subtype's mean (from the
others) is the nearest - and angle-only. Usage: fly_direction_measures.py"""
import glob
import importlib.util
import os
import pickle
from collections import deque

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from faab.config import fig

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ld", os.path.join(HERE, "t4t5_layers_dendrites.py"))
LD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LD)
spec2 = importlib.util.spec_from_file_location("fc", os.path.join(HERE, "fly_t4t5_dendrite_check.py"))
FC = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(FC)
OUT = fig("t4t5")
NM = 8.0
RESAMPLE = 3.0


def resample(df, step_um):
    """skeleton table (rowId, link, x, y, z in 8 nm voxels) resampled to about step_um between nodes (navis)"""
    import navis
    nodes = pd.DataFrame({"node_id": df.rowId, "parent_id": df.link, "x": df.x * NM, "y": df.y * NM, "z": df.z * NM,
                          "radius": 0.0})
    n = navis.TreeNeuron(nodes, units="nm")
    r = navis.resample_skeleton(n, resample_to=step_um * 1000)
    nd = r.nodes
    return pd.DataFrame({"rowId": nd.node_id, "link": nd.parent_id, "x": nd.x / NM, "y": nd.y / NM, "z": nd.z / NM})


def adjacency(nd):
    idx = {n: i for i, n in enumerate(nd.node_id)}
    adj = [[] for _ in range(len(nd))]
    for c, p in zip(nd.node_id, nd.parent_id):
        if p in idx:
            adj[idx[c]].append(idx[p])
            adj[idx[p]].append(idx[c])
    return adj


def measures(skdf, Pd, lop_c, frame):
    S = FC.Sk(skdf)
    xyz0 = S.nodes[["x", "y", "z"]].to_numpy(float) / 1000.0
    start = int(np.argmin(np.linalg.norm(xyz0 - lop_c, axis=1)))
    g, xyz = LD.geodesic_from(S, start)
    near = cKDTree(Pd).query(xyz)[0] <= LD.NEAR_NODE
    if near.sum() < 3:
        return None
    den = np.where(near)[0]
    bi = den[np.nanargmin(np.where(np.isfinite(g[den]), g[den], np.inf))]
    base, cen = xyz[bi], Pd.mean(0)
    Cd, Nd, tr = frame
    nn = Nd[tr.query(cen)[1]]
    e1 = np.array([0.0, 1.0, 0.0]) - nn[1] * nn
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nn, e1)
    ang = lambda v: float(np.degrees(np.arctan2(v @ e2, v @ e1)))  # noqa: E731
    out = {"centroid": ang(cen - base)}
    # branches: BFS from the base through dendrite nodes; edges parent -> child (away from the base)
    adj, inden = adjacency(S.nodes), set(den.tolist())
    seen, q, vsum = {bi}, deque([bi]), np.zeros(3)
    dist = {bi: 0.0}
    while q:
        a = q.popleft()
        for b in adj[a]:
            if b in seen or b not in inden:
                continue
            seen.add(b)
            vsum += xyz[b] - xyz[a]
            dist[b] = dist[a] + np.linalg.norm(xyz[b] - xyz[a])
            q.append(b)
    out["branches"] = ang(vsum) if np.linalg.norm(vsum) > 0 else np.nan
    if len(dist) >= 4:
        dv = np.array(list(dist.values()))
        far = [k for k, v in dist.items() if v >= np.percentile(dv, 70)]
        out["tips"] = ang(xyz[far].mean(0) - base)
    else:
        out["tips"] = np.nan
    return out


def circ(a):
    a = np.radians(np.asarray(a, float))
    return np.arctan2(np.sin(a).mean(), np.cos(a).mean()), np.hypot(np.sin(a).mean(), np.cos(a).mean())


def loo_score(df, col, use_depth):
    """within each patch, leave-one-out: is the cell's own subtype the nearest class (angle, + depth)?"""
    ok, n = 0, 0
    for _, h in df.groupby("patch"):
        h = h.dropna(subset=[col, "depth"])
        for i, r in h.iterrows():
            rest = h.drop(i)
            best, bd = None, np.inf
            for st, q in rest.groupby("type"):
                if len(q) < 3:
                    continue
                m, _ = circ(q[col])
                da = abs(np.angle(np.exp(1j * (np.radians(r[col]) - m)))) / np.pi  # 0..1
                dd = abs(r.depth - q.depth.median()) / 0.5 if use_depth else 0.0
                d = da + dd
                if d < bd:
                    best, bd = st, d
            ok += best == r.type
            n += 1
    return ok / max(n, 1)


def main():
    raw = pickle.load(open(sorted(glob.glob(os.path.join(OUT, "*_fly_t4t5_raw.pkl")))[-1], "rb"))
    chk = pd.read_csv(sorted(glob.glob(os.path.join(OUT, "*_fly_t4t5_check.tsv")))[-1], sep="\t")
    dep = dict(zip(chk.cell, chk.depth))
    typ = dict(raw["cells"])
    inp, out = raw["inp"], raw["out"]
    inp = inp[inp.bodyId_post.isin(typ)]
    out = out[out.bodyId_pre.isin(typ)]
    um = lambda d, sfx: d[[f"x_{sfx}", f"y_{sfx}", f"z_{sfx}"]].to_numpy(float) * NM / 1000.0  # noqa: E731
    I, O = um(inp, "post"), um(out, "pre")
    lop = (out.roi_pre == "LOP(R)").to_numpy()
    P = O[lop]
    lop_c = pd.DataFrame(P).groupby(out.bodyId_pre.to_numpy()[lop]).mean()
    reg = {"T4": "ME(R)", "T5": "LO(R)"}
    frames = {}
    for t, r in reg.items():
        m = (inp.roi_post == r).to_numpy() & inp.bodyId_post.map(lambda x: typ[x][:2] == t).to_numpy()
        Cd, Nd = LD.planes(I[m])
        Nd = FC.orient_to_points(Cd, Nd, P)
        frames[t] = (Cd, Nd, cKDTree(Cd))
    rows = []
    for cid, t in raw["cells"]:
        m = ((inp.bodyId_post == cid) & (inp.roi_post == reg[t[:2]])).to_numpy()
        skdf = raw["sk"].get(cid)
        if skdf is None or m.sum() < LD.MIN_DEN or cid not in lop_c.index:
            continue
        Pd = I[m]
        rec = dict(cell=cid, type=t, depth=dep.get(cid, np.nan), cen_x=Pd[:, 0].mean(), cen_y=Pd[:, 1].mean(),
                   cen_z=Pd[:, 2].mean())
        full = measures(skdf, Pd, lop_c.loc[cid].to_numpy(float), frames[t[:2]])
        coarse = measures(resample(skdf, RESAMPLE), Pd, lop_c.loc[cid].to_numpy(float), frames[t[:2]])
        if full is None or coarse is None:
            continue
        rec.update({f"{k}_full": v for k, v in full.items()})
        rec.update({f"{k}_coarse": v for k, v in coarse.items()})
        rows.append(rec)
    D = pd.DataFrame(rows)
    print(f"{len(D)} fly cells with all measures")
    cols = [f"{k}_{s}" for k in ("centroid", "branches", "tips") for s in ("full", "coarse")]
    for t in ("T4", "T5"):
        g = D[D.type.str.startswith(t)].copy()
        g["patch"] = KMeans(3, n_init=5, random_state=0).fit_predict(g[["cen_x", "cen_y", "cen_z"]])
        print(f"{t} (n {len(g)}; chance 0.25): leave-one-out subtype from angle (+ depth) within 3 patches")
        print(f"   depth only: {loo_score(g.assign(zero=0.0), 'zero', True):.2f}")
        for c in cols:
            print(f"   {c:16s} angle {loo_score(g, c, False):.2f} | angle + depth {loo_score(g, c, True):.2f}")
        for c in ("branches_full", "branches_coarse"):
            parts = []
            for p, h in g.groupby("patch"):
                for st, q in h.groupby("type"):
                    m, R = circ(q[c].dropna())
                    parts.append(f"p{p} {st} {np.degrees(m):4.0f}/{R:.2f}")
            print(f"   {c} mean deg / R: " + "; ".join(parts))
    D.to_csv(os.path.join(OUT, "fly_direction_measures.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
