"""Stage 1 of a new T4/T5 candidate tab (David 2026-11-11: "a new tab with additional T4/T5 candidates ... use the
criteria that we developed in the past"). Read-only (snapshot caught up to now).
Past criteria (2026-10-24..11-09, memories faab-lplc2-search / faab-t4t5-inputs-depth): T4/T5 are reciprocal partners
of the CT1 cells (>= 3 synapses each way with one CT1 cell of the side); a T4/T5 has most of its OUTPUTS in the lobula
plate (T4: inputs in the medulla, T5: inputs in the lobula); T4/T5 are small (labeled T4/T5 nucleus rank median 345,
cut <= 600 in the LPLC2 search; very large cells out); morphology check = centroid-aligned NBLAST kNN (t4t5_knn.py),
a later stage. Right side only: CT1 648518347494887381 and the CT1-like LO ME cell 648518347672557281 (SeaTable labels
their sides the other way round; by x they are in the right optic lobe with PR group 1).
Per cell: synapses from / onto each CT1, input and output counts, input and output shares per neuropil (LA/ME/LO/LOP
right), T4-like (ME inputs > LO inputs) or T5-like, and whether it is on David's 411 list or in PRg1toT4T5.
Writes figures/t4t5/<stamp>_t4t5_candidates_pool.tsv and prints the counts.
Usage: t4t5_candidates.py BOOK.xlsx"""
import os
import re
import sys

import numpy as np
import openpyxl
import pandas as pd

from faab import ol_meshes, synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import FIGURES, TMP
from faab.partners.sheet import cell_text as s, utc_stamp

CT1 = {648518347494887381: "CT1", 648518347672557281: "CT1-like (LO ME)"}
VOX = np.array([0.016, 0.016, 0.045])
LABS = ["LA_R", "ME_R", "LO_R", "LOP_R"]
IDS = re.compile(r"\d{15,}")
ONE_WAY = "--one-way" in sys.argv  # refinement test: >= 3 synapses from OR onto one CT1 cell


def main():
    stamp = utc_stamp()
    c = cave_client()
    last = throttle(0)
    ok = np.asarray(c.chunkedgraph.is_latest_roots(list(CT1))).astype(bool)
    assert ok.all(), f"a CT1 root changed: {dict(zip(CT1, ok))}"
    snap = synsnap.Snapshot(log=lambda *_: None)
    out_ = snap.synapses(pre=list(CT1), when="now", client=c)
    in_ = snap.synapses(post=list(CT1), when="now", client=c)
    f = out_.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    t = in_.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    rows = []
    for ct in CT1:
        a = f[f.pre_root == ct].set_index("post_root").n
        b = t[t.post_root == ct].set_index("pre_root").n
        both = a.index.union(b.index)
        rows.append(pd.DataFrame({"cell": both, "ct1": ct, "from_ct1": a.reindex(both).fillna(0).astype(int).to_numpy(),
                                  "onto_ct1": b.reindex(both).fillna(0).astype(int).to_numpy()}))
    r = pd.concat(rows)
    keep = ((r.from_ct1 >= 3) | (r.onto_ct1 >= 3)) if ONE_WAY else ((r.from_ct1 >= 3) & (r.onto_ct1 >= 3))
    r = r[keep & ~r.cell.isin(list(CT1))]
    r = r.sort_values("from_ct1", ascending=False).drop_duplicates("cell")  # reciprocal with either cell
    cells = sorted(r.cell.astype(np.int64))
    print(f"CT1 {'contact (>= 3 either way' if ONE_WAY else '-reciprocal (>= 3 each way'} with one CT1 cell): {len(cells)} cells "
          f"({r.ct1.map(CT1).value_counts().to_dict()})", flush=True)

    def shares(side):
        df = snap.synapses(pre=cells if side == "pre" else None, post=cells if side == "post" else None, when="now",
                           positions=True, client=c)
        df = df[df.pre_root != df.post_root]
        k = "pre" if side == "pre" else "post"
        P = df[[f"{k}_x", f"{k}_y", f"{k}_z"]].to_numpy(float) * VOX
        df = df.assign(np=ol_meshes.assign(P, LABS))
        g = pd.crosstab(df[f"{side}_root"], df.np)
        for L in LABS + ["none"]:
            if L not in g:
                g[L] = 0
        n = g.sum(axis=1)
        return n, g[LABS + ["none"]].div(n.clip(lower=1), axis=0)

    n_out, s_out = shares("pre")
    n_in, s_in = shares("post")
    q = pd.DataFrame({"cell": cells}).set_index("cell")
    q["n_out"], q["n_in"] = n_out.reindex(q.index).fillna(0).astype(int), n_in.reindex(q.index).fillna(0).astype(int)
    for L in LABS:
        q[f"out_{L}"] = s_out[L].reindex(q.index).fillna(0).round(2)
        q[f"in_{L}"] = s_in[L].reindex(q.index).fillna(0).round(2)
    q["kind"] = np.where(q.in_ME_R >= q.in_LO_R, "T4-like", "T5-like")
    q = q.join(r.set_index("cell")[["ct1", "from_ct1", "onto_ct1"]])
    q["ct1"] = q.ct1.map(CT1)

    # already known: David's 411 list (current roots of tmp/t4t5_query_survey.tsv 'now') and PRg1toT4T5 rows
    sv = pd.read_csv(os.path.join(TMP, "t4t5_query_survey.tsv"), sep="\t", dtype=str)
    old = [int(x) for x in sv.now]
    ok = np.asarray(c.chunkedgraph.is_latest_roots(old)).astype(bool)
    known = {x for x, g in zip(old, ok) if g}
    for x in [x for x, g in zip(old, ok) if not g]:
        last = throttle(last)
        known |= {int(y) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(x))}
    ws = openpyxl.load_workbook(sys.argv[1], read_only=True)["PRg1toT4T5"]
    intab = {int(s(row[6])) for row in ws.iter_rows(values_only=True) if len(row) > 6 and IDS.fullmatch(s(row[6]))}
    q["on_411_list"] = q.index.isin(known)
    q["in_PRg1toT4T5"] = q.index.isin(intab)
    out = os.path.join(FIGURES, "t4t5", f"{stamp}_t4t5_candidates_pool{'_oneway' if ONE_WAY else ''}.tsv")
    q.reset_index().to_csv(out, sep="\t", index=False)

    lop = q.out_LOP_R >= 0.5
    print(f"outputs >= 50% in the lobula plate: {int(lop.sum())} ({q[lop].kind.value_counts().to_dict()})")
    for k in (0.3, 0.5, 0.7, 0.9):
        print(f"  LOP output share >= {k}: {int((q.out_LOP_R >= k).sum())}")
    print(f"on David's 411 list: {int(q.on_411_list.sum())} (of those LOP >= 50%: {int((q.on_411_list & lop).sum())}); "
          f"in PRg1toT4T5: {int(q.in_PRg1toT4T5.sum())}")
    new = q[lop & ~q.on_411_list & ~q.in_PRg1toT4T5]
    print(f"NEW candidates (LOP >= 50%, not listed): {len(new)}; outputs per cell median {int(new.n_out.median())}, "
          f"90% {int(new.n_out.quantile(0.9))}; kinds {new.kind.value_counts().to_dict()}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
