"""Fly check of the input-position measure (male optic lobe, known input types): where each input type's synapses lie
along a T4 / T5 dendrite, measured exactly as for Aedes but along two candidate axes - the branch direction
(fly_direction_measures 'branches', 3 um skeletons) and the centroid - base vector. Literature to recover: on T4,
Mi9 toward the distal tips, Mi1 / Tm3 in the middle, Mi4 / C3 / CT1 toward the base (Takemura 2017,
10.7554/eLife.24394); on T5, Tm9 distal, Tm1 / Tm2 / Tm4 middle, CT1 proximal (Shinomiya 2019, 10.7554/eLife.40025).
Reads the newest *_fly_t4t5_raw.pkl; one neuPrint lookup (batched, serial) of the input cells' types, cached in
figures/t4t5/fly_input_types.tsv. Usage: fly_input_positions.py"""
import glob
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab.config import fig

HERE = os.path.dirname(os.path.abspath(__file__))
_s = lambda n: importlib.util.spec_from_file_location(n, os.path.join(HERE, f"{n}.py"))  # noqa: E731
LD = importlib.util.module_from_spec(_s("t4t5_layers_dendrites"))
_s("t4t5_layers_dendrites").loader.exec_module(LD)
FC = importlib.util.module_from_spec(_s("fly_t4t5_dendrite_check"))
_s("fly_t4t5_dendrite_check").loader.exec_module(FC)
FM = importlib.util.module_from_spec(_s("fly_direction_measures"))
_s("fly_direction_measures").loader.exec_module(FM)
OUT = fig("t4t5")
NM = 8.0
TYPES_TSV = os.path.join(OUT, "fly_input_types.tsv")


def input_types(ids):
    if os.path.exists(TYPES_TSV):
        T = pd.read_csv(TYPES_TSV, sep="\t")
        if set(ids) <= set(T.id):
            return dict(zip(T.id, T.type.fillna("")))
    from faab.clients.neuprint import OPTIC_LOBE, client
    c = client(OPTIC_LOBE)
    ids = sorted(set(int(i) for i in ids))
    rows = []
    for b in range(0, len(ids), 2000):
        q = f"MATCH (n:Neuron) WHERE n.bodyId IN {ids[b:b + 2000]} RETURN n.bodyId AS id, n.type AS type"
        rows.append(c.fetch_custom(q))
    T = pd.concat(rows)
    T.to_csv(TYPES_TSV, sep="\t", index=False)
    return dict(zip(T.id, T.type.fillna("")))


def main():
    raw = pickle.load(open(sorted(glob.glob(os.path.join(OUT, "*_fly_t4t5_raw.pkl")))[-1], "rb"))
    typ = dict(raw["cells"])
    inp, out = raw["inp"], raw["out"]
    inp = inp[inp.bodyId_post.isin(typ)].copy()
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
    pre_t = input_types(inp.bodyId_pre.unique().tolist())
    inp["pre_type"] = inp.bodyId_pre.map(pre_t).fillna("")
    rows = []
    for cid, t in raw["cells"]:
        m = ((inp.bodyId_post == cid) & (inp.roi_post == reg[t[:2]])).to_numpy()
        skdf = raw["sk"].get(cid)
        if skdf is None or m.sum() < LD.MIN_DEN or cid not in lop_c.index:
            continue
        Pd = I[m]
        S = FC.Sk(FM.resample(skdf, FM.RESAMPLE))
        xyz0 = S.nodes[["x", "y", "z"]].to_numpy(float) / 1000.0
        start = int(np.argmin(np.linalg.norm(xyz0 - lop_c.loc[cid].to_numpy(float), axis=1)))
        g, xyz = LD.geodesic_from(S, start)
        near = cKDTree(Pd).query(xyz)[0] <= LD.NEAR_NODE
        if near.sum() < 3:
            continue
        den = np.where(near)[0]
        bi = den[np.nanargmin(np.where(np.isfinite(g[den]), g[den], np.inf))]
        base, cen = xyz[bi], Pd.mean(0)
        Cd, Nd, tr = frames[t[:2]]
        nn = Nd[tr.query(cen)[1]]
        bv, _ = LD.branch_vector(S, xyz, den, bi)
        for name, v in (("branch", bv), ("centroid", cen - base)):
            v = v - (v @ nn) * nn
            if np.linalg.norm(v) == 0:
                continue
            u = v / np.linalg.norm(v)
            s_raw = (Pd - base) @ u
            L = np.percentile(s_raw, 95)
            if L <= 0:
                continue
            rows.append(pd.DataFrame({"cell": cid, "type": t, "axis": name, "pre": inp.bodyId_pre.to_numpy()[m],
                                      "pre_type": inp.pre_type.to_numpy()[m], "s": s_raw / L}))
    S = pd.concat(rows)
    S.to_parquet(os.path.join(OUT, "fly_input_positions.parquet"))
    for t, want in (("T4", ["Mi9", "Mi1", "Tm3", "Mi4", "C3", "CT1", "TmY15", "T4"]),
                    ("T5", ["Tm9", "Tm1", "Tm2", "Tm4", "CT1", "T5", "LT33", "Tm23"])):
        for ax in ("branch", "centroid"):
            q = S[S.type.str.startswith(t) & (S.axis == ax)]
            q = q.assign(pt=q.pre_type.where(~q.pre_type.str.match(rf"^{t}[a-d]$"), t))
            med = q.groupby("pt").s.agg(["median", "size"])
            parts = [f"{k} {med.loc[k, 'median']:.2f} (n {int(med.loc[k, 'size'])})" for k in want if k in med.index]
            print(f"{t} along {ax:8s} (0 base, 1 tips): " + " | ".join(parts))


if __name__ == "__main__":
    main()
