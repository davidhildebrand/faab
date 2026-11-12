"""Every cell with output synapses in the right lobula plate, from the local synapse snapshot (network-free apart from
the snapshot's own refresh; no CAVE calls). For the T4/T5 tab (David 2026-11-12, expected ~2,700 right T4/T5).
Fly calibration (tmp/fly_lop_criterion.py, male-CNS v1.0): '>= 50% of outputs in LOP(R)' AND 'more inputs in ME(R) +
LO(R) than in LOP(R)' passes all 6,793 right T4/T5 and 3 other cells (precision 100%) - no CT1 needed.
Per root at the newest checkpoint: n_out, out_LOP_R; n_in, in_LA_R / in_ME_R / in_LO_R / in_LOP_R (self-synapses and
root 0 dropped). Writes figures/t4t5/<stamp>_t4t5_lop_scan.parquet and prints the pass counts by size.
Usage: t4t5_lop_scan.py"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from faab import ol_meshes, synsnap
from faab.config import FIGURES
from faab.partners.sheet import utc_stamp

VOX = np.array([0.016, 0.016, 0.045])  # um per voxel
LABS = ["LA_R", "ME_R", "LO_R", "LOP_R"]


def bbox_vox(label, pad_um=2.0):
    v, _ = ol_meshes.mesh(label)
    v = np.asarray(v, float) / 1000.0
    return (v.min(0) - pad_um) / VOX, (v.max(0) + pad_um) / VOX


def in_boxes(X, labels):
    m = np.zeros(len(X), bool)
    for lab in labels:
        lo, hi = bbox_vox(lab)
        m |= np.all((X >= lo) & (X <= hi), axis=1)
    return m


def main():
    t0 = time.time()
    stamp = utc_stamp()
    snap = synsnap.Snapshot()
    tag = snap.at("latest")
    A = snap.load(tag)["pre"]
    pre, post, sid = A["pre_root"], A["post_root"], A["id"]
    keep = (pre != post) & (pre != 0) & (post != 0)
    pre, post, sid = pre[keep], post[keep], sid[keep]
    print(f"checkpoint {tag}: {len(pre):,} synapses ({time.time() - t0:.0f} s)", flush=True)
    st = pq.read_table(os.path.join(snap.root, "static.parquet"),
                       columns=["id", "pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z"])
    sids = st["id"].to_numpy()
    idx = np.searchsorted(sids, sid)
    assert np.array_equal(sids[idx], sid), "synapse ids missing from static.parquet"

    def xyz(side):
        return np.stack([st[f"{side}_{k}"].to_numpy()[idx] for k in "xyz"], axis=1)

    out = {}
    # outputs in LOP_R
    P = xyz("pre")
    m = in_boxes(P, ["LOP_R"])
    lab = ol_meshes.assign(P[m] * VOX, ["LOP_R"])
    r, n = np.unique(pre[m][lab == "LOP_R"], return_counts=True)
    out["out_LOP_R"] = pd.Series(n, index=r)
    del P
    print(f"outputs in LOP_R: {n.sum():,} synapses, {len(r):,} roots ({time.time() - t0:.0f} s)", flush=True)
    # inputs per optic-lobe neuropil, only for roots that have LOP_R output (keeps the point count down)
    cand = np.isin(post, r)
    Q = xyz("post")[cand]
    pc = post[cand]
    m = in_boxes(Q, LABS)
    lab = ol_meshes.assign(Q[m] * VOX, LABS)
    for L in LABS:
        rr, nn = np.unique(pc[m][lab == L], return_counts=True)
        out[f"in_{L}"] = pd.Series(nn, index=rr)
    print(f"inputs of those roots: {len(pc):,} synapses ({time.time() - t0:.0f} s)", flush=True)
    ro, no = np.unique(pre, return_counts=True)
    ri, ni = np.unique(post, return_counts=True)
    d = pd.DataFrame(index=pd.Index(r, name="root"))
    d["n_out"] = pd.Series(no, index=ro).reindex(d.index).fillna(0).astype(int)
    d["n_in"] = pd.Series(ni, index=ri).reindex(d.index).fillna(0).astype(int)
    for k, s in out.items():
        d[k] = s.reindex(d.index).fillna(0).astype(int)
    d["share_out_LOP"] = d.out_LOP_R / d.n_out
    d["rule"] = (d.share_out_LOP >= 0.5) & (d.in_ME_R + d.in_LO_R > d.in_LOP_R)
    d["checkpoint"] = tag
    path = os.path.join(FIGURES, "t4t5", f"{stamp}_t4t5_lop_scan.parquet")
    d.reset_index().to_parquet(path, index=False)
    print(f"roots with LOP_R output {len(d):,}; passing the fly rule {int(d.rule.sum()):,}")
    for k in (1, 5, 10, 20, 40):
        p = d[d.rule & (d.n_out >= k)]
        print(f"  rule + >= {k} outputs: {len(p):,} (T4-like {(p.in_ME_R >= p.in_LO_R).sum():,}, T5-like {(p.in_ME_R < p.in_LO_R).sum():,})")
    print(f"written {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
