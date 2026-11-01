"""NBLAST scores among the LMC cells (lmc_data.py), two ways, for the L1-L5 typing (David, 2026-10-31):
  align     navis.nblast_align, rigid CPD without scaling (scaling would erase the depth differences between types),
            both directions averaged
  centroid  translation only (centroids superposed), which keeps size and orientation; it worked for T4/T5 where
            rigid alignment did not (faab-type-match-fragment-search)
Dotprops from the L2 skeletons resampled to 1 um (k=5). Left cells are mirrored into the right frame (x -> 2*mid - x,
mid between the side medians). Network-free; resumable per query chunk. Writes figures/lmc/lmc_nblast.pkl and a
progress log tmp/lmc_nblast.log. Usage: lmc_nblast.py [--cores 6] [--chunk 14]"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from faab.config import TMP, fig

LOG = os.path.join(TMP, "lmc_nblast.log")


def log(msg):
    print(msg, flush=True)
    open(LOG, "a").write(msg + "\n")


def main():
    import navis
    navis.set_pbars(hide=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    cells = pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).fillna("")
    sk = pickle.load(open(fig("lmc_skeletons.pkl"), "rb"))
    cells = cells[cells.root.isin(sk)]
    pts = {r: navis.resample_skeleton(sk[r], resample_to=1).nodes[["x", "y", "z"]].to_numpy(float) for r in cells.root}
    cx = {r: p[:, 0].mean() for r, p in pts.items()}
    side = dict(zip(cells.root, cells.side))
    ml = np.median([cx[r] for r in cx if side[r] == "L"])
    mr = np.median([cx[r] for r in cx if side[r] == "R"])
    mid = (ml + mr) / 2
    log(f"{len(pts)} cells; midline x {mid:.1f} um (L median {ml:.1f}, R {mr:.1f}); left cells mirrored")
    for r, p in pts.items():
        if side[r] == "L":
            p[:, 0] = 2 * mid - p[:, 0]

    def dp(p, rid):
        d = navis.make_dotprops(p, k=5)
        d.units, d.id, d.name = "um", rid, rid
        return d

    ids = list(cells.root)
    path = fig("lmc_nblast.pkl")
    store = pickle.load(open(path, "rb")) if os.path.exists(path) else {"fwd": {}, "rev": {}}
    old = store.get("ids", [])
    for d in ("fwd", "rev"):  # rows stored as arrays over the old id list -> {target: score}
        store[d] = {q: (dict(zip(old, v)) if isinstance(v, np.ndarray) else v) for q, v in store[d].items()}
    store.update(ids=ids, mid=mid)
    # centroid: one all-by-all call (fast, always redone)
    C = navis.NeuronList([dp(pts[r] - pts[r].mean(0), r) for r in ids])
    m = navis.nblast_allbyall(C, n_cores=a.cores, progress=False).loc[ids, ids].to_numpy()
    store["centroid"] = (m + m.T) / 2
    pickle.dump(store, open(path, "wb"))
    log(f"centroid NBLAST done at {time.time() - t0:.0f} s")
    # align: rigid CPD per pair, both directions, chunked; new cells also get their columns in the cached rows
    kw = dict(align_method="rigid", two_way_align=False, n_cores=a.cores, progress=False, align_kwargs={"scale": False})
    nl = lambda rs: navis.NeuronList([dp(pts[r], r) for r in rs])
    todo = [r for r in ids if r not in store["fwd"]]
    new_t = [t for t in ids if any(t not in store["fwd"][q] for q in store["fwd"])]
    log(f"align: {len(ids) - len(todo)} cached, {len(todo)} to score; {len(new_t)} new targets for the cached rows")
    if new_t and store["fwd"]:
        qs = [q for q in store["fwd"] if q in pts]
        f = navis.nblast_align(nl(qs), nl(new_t), **kw).values
        rv = navis.nblast_align(nl(new_t), nl(qs), **kw).values.T
        for j, q in enumerate(qs):
            store["fwd"][q].update(zip(new_t, f[j]))
            store["rev"][q].update(zip(new_t, rv[j]))
        pickle.dump(store, open(path, "wb"))
        log(f"  new targets scored at {time.time() - t0:.0f} s")
    T = nl(ids)
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        f = navis.nblast_align(nl(part), T, **kw).values
        rv = navis.nblast_align(T, nl(part), **kw).values.T
        for j, r in enumerate(part):
            store["fwd"][r], store["rev"][r] = dict(zip(ids, f[j])), dict(zip(ids, rv[j]))
        pickle.dump(store, open(path, "wb"))
        log(f"  align {min(i + a.chunk, len(todo))}/{len(todo)} at {time.time() - t0:.0f} s")
    F = np.array([[store["fwd"][q][t] for t in ids] for q in ids])
    R = np.array([[store["rev"][q][t] for t in ids] for q in ids])
    store["align"] = (F + R) / 2
    pickle.dump(store, open(path, "wb"))
    log(f"done at {time.time() - t0:.0f} s; written {path}")


if __name__ == "__main__":
    main()
