"""Fly check of the dendrite / lobula-plate measures of t4t5_layers_dendrites.py on cells whose subtype is known
(male optic lobe, neuPrint optic-lobe:v1.1, read-only; same specimen as male-cns). For N cells of each of T4a-d and
T5a-d: input and output synapses (positions), skeletons; then exactly the Aedes measures - lobula-plate depth of the
outputs (local planes, normalized 0-1), dendrite base (skeleton node nearest the axon along the cable), dendrite
vector (input centroid - base, in the local dendrite-layer plane), its angle in the local frame (dataset y projected),
main axis, asymmetry - and how well each separates the eight subtypes. Serial queries only.
Writes figures/t4t5/<stamp>_fly_t4t5_check.tsv (+ the raw synapses / skeletons in a pickle for reruns).
Usage: fly_t4t5_dendrite_check.py [N]   (default 60; reuses the newest pickle if one exists with the same N)"""
import glob
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab.config import fig
from faab.partners.sheet import utc_stamp

spec = importlib.util.spec_from_file_location("ld", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "t4t5_layers_dendrites.py"))
LD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LD)
OUT = fig("t4t5")
NM = 8.0  # neuPrint voxel (nm)
TYPES = [f"T4{x}" for x in "abcd"] + [f"T5{x}" for x in "abcd"]


def fetch(n):
    from neuprint import NeuronCriteria as NC, fetch_synapse_connections
    from faab.clients.neuprint import OPTIC_LOBE, client
    c = client(OPTIC_LOBE)
    cells = []
    for t in TYPES:
        ids = c.fetch_custom(f"MATCH (n:Neuron {{type:'{t}'}}) RETURN n.bodyId AS id ORDER BY id").id.to_numpy()
        pick = np.random.default_rng(len(t) * 7 + ord(t[-1])).choice(ids, min(n, len(ids)), replace=False)
        cells += [(int(i), t) for i in pick]
    ids = [i for i, _ in cells]
    inp, out = [], []
    for b in range(0, len(ids), 40):
        inp.append(fetch_synapse_connections(None, NC(bodyId=ids[b:b + 40]), client=c))
        out.append(fetch_synapse_connections(NC(bodyId=ids[b:b + 40]), None, client=c))
        print(f"  synapses {min(b + 40, len(ids))}/{len(ids)}", flush=True)
    sk = {}
    for k, i in enumerate(ids):
        try:
            sk[i] = c.fetch_skeleton(i, format="pandas")
        except Exception as e:
            print(f"  skeleton {i} failed: {str(e)[:80]}", flush=True)
        if k % 80 == 0:
            print(f"  skeletons {k + 1}/{len(ids)}", flush=True)
    return dict(n=n, cells=cells, inp=pd.concat(inp), out=pd.concat(out), sk=sk)


class Sk:  # the navis-like bits geodesic_from() uses
    def __init__(self, df):
        self.nodes = pd.DataFrame({"node_id": df.rowId, "parent_id": df.link, "x": df.x * NM, "y": df.y * NM,
                                   "z": df.z * NM})
        self.n_nodes = len(df)


def orient_to_points(C, N, pts):
    _, j = cKDTree(pts).query(C)
    sg = np.sign(np.einsum("ij,ij->i", pts[j] - C, N))
    sg[sg == 0] = 1
    return N * sg[:, None]


def circ(a):
    a = np.radians(np.asarray(a, float))
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())), np.hypot(np.sin(a).mean(), np.cos(a).mean())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    old = [f for f in sorted(glob.glob(os.path.join(OUT, "*_fly_t4t5_raw.pkl")))]
    raw = pickle.load(open(old[-1], "rb")) if old else None
    stamp = utc_stamp()
    if raw is None or raw["n"] != n:
        raw = fetch(n)
        pickle.dump(raw, open(os.path.join(OUT, f"{stamp}_fly_t4t5_raw.pkl"), "wb"))
    typ = dict(raw["cells"])
    inp, out = raw["inp"].copy(), raw["out"].copy()
    inp["post"] = inp.bodyId_post.astype(np.int64)
    out["pre"] = out.bodyId_pre.astype(np.int64)
    inp = inp[inp.post.isin(typ)]
    out = out[out.pre.isin(typ)]
    um = lambda d, sfx: d[[f"x_{sfx}", f"y_{sfx}", f"z_{sfx}"]].to_numpy(float) * NM / 1000.0  # noqa: E731
    I, O = um(inp, "post"), um(out, "pre")
    # lobula-plate depth of the outputs
    lop = (out.roi_pre == "LOP(R)").to_numpy()
    P = O[lop]
    Cp, Np = LD.planes(P)
    lo_pts = I[(inp.roi_post == "LO(R)").to_numpy()]
    Np = orient_to_points(Cp, Np, lo_pts)
    d, j = LD.signed_depth(P, Cp, Np)
    L = pd.DataFrame({"cell": out.pre.to_numpy()[lop], "d": d, "plane": j})
    q = L.groupby("plane").d.quantile([0.05, 0.95]).unstack()
    L["dn"] = (L.d - L.plane.map(q[0.05])) / (L.plane.map(q[0.95]) - L.plane.map(q[0.05]))
    dep = L.groupby("cell").dn.median()
    print(f"fly: {len(typ)} cells; LOP outputs {lop.sum()}, {len(Cp)} planes; 5-95% thickness median "
          f"{(q[0.95] - q[0.05]).median():.1f} um")
    # dendrite vectors
    reg = {"T4": "ME(R)", "T5": "LO(R)"}
    frames = {}
    for t, r in reg.items():
        m = (inp.roi_post == r).to_numpy() & inp.post.map(lambda x: typ[x][:2] == t).to_numpy()
        Cd, Nd = LD.planes(I[m])
        Nd = orient_to_points(Cd, Nd, P)
        frames[t] = (Cd, Nd, cKDTree(Cd))
    lop_c = pd.DataFrame(P).groupby(out.pre.to_numpy()[lop]).mean()
    rows = []
    for cid, t in raw["cells"]:
        rec = dict(cell=cid, type=t, depth=dep.get(cid, np.nan))
        m = ((inp.post == cid) & (inp.roi_post == reg[t[:2]])).to_numpy()
        S = raw["sk"].get(cid)
        if S is None or m.sum() < LD.MIN_DEN or cid not in lop_c.index:
            rows.append(rec)
            continue
        Pd = I[m]
        S = Sk(S)
        xyz0 = S.nodes[["x", "y", "z"]].to_numpy(float) / 1000.0
        start = int(np.argmin(np.linalg.norm(xyz0 - lop_c.loc[cid].to_numpy(float), axis=1)))
        g, xyz = LD.geodesic_from(S, start)
        near = cKDTree(Pd).query(xyz)[0] <= LD.NEAR_NODE
        if not near.any():
            rows.append(rec)
            continue
        bi = np.where(near)[0][np.nanargmin(np.where(np.isfinite(g[near]), g[near], np.inf))]
        base, cen = xyz[bi], Pd.mean(0)
        Cd, Nd, tr = frames[t[:2]]
        nn = Nd[tr.query(cen)[1]]
        e1 = np.array([0.0, 1.0, 0.0]) - nn[1] * nn
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nn, e1)
        v = cen - base
        v2 = np.array([v @ e1, v @ e2])
        Q = np.c_[(Pd - cen) @ e1, (Pd - cen) @ e2]
        w, U = np.linalg.eigh(np.cov(Q.T))
        rec.update(angle=float(np.degrees(np.arctan2(v2[1], v2[0]))), vlen=float(np.hypot(*v2)),
                   sd_major=float(np.sqrt(w[1])), axis_angle=float(np.degrees(np.arctan2(U[1, 1], U[0, 1]))) % 180,
                   cen_x=cen[0], cen_y=cen[1], cen_z=cen[2])
        rec["asym"] = rec["vlen"] / rec["sd_major"]
        rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, f"{stamp}_fly_t4t5_check.tsv"), sep="\t", index=False)
    print("per subtype: lobula-plate depth median (IQR) | dendrite angle circular mean, R | axis 0-180 median | "
          "asymmetry median | n with a vector")
    for t in TYPES:
        g = D[D.type == t]
        gv = g[g.angle.notna()]
        mu, R = circ(gv.angle) if len(gv) else (np.nan, np.nan)
        print(f"  {t}: depth {g.depth.median():.2f} ({g.depth.quantile(.25):.2f}-{g.depth.quantile(.75):.2f}) | "
              f"angle {mu:7.1f}, R {R:.2f} | axis {gv.axis_angle.median():5.1f} | asym {gv.asym.median():.2f} | {len(gv)}")
    print(f"written {stamp}_fly_t4t5_check.tsv")


if __name__ == "__main__":
    main()
