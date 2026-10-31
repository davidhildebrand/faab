"""Larger FAFB/FlyWire photoreceptor sample (tighter ranges) + convergent output types.

Fetches synapses in throttled chunks, saves per-cell metrics, and aggregates which
TARGET CELL TYPES the photoreceptors converge onto. Read-only.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
from fafbseg import flywire
from faab.clients.cave import throttle
from faab.config import fig

OUT = os.path.join(REPO, "figures")
NSAMPLE = 150
CHUNK = 50
SEED = 0


def main():
    rng = np.random.RandomState(SEED)
    # full annotation table (cached locally from github) -> root_id => cell_type
    ann_all = flywire.get_hierarchical_annotations(materialization=783, dataset="public")
    ann_all = ann_all[ann_all.root_id.notna()]
    type_of = dict(zip(ann_all.root_id.astype("int64"), ann_all.cell_type.astype(str)))
    print(f"annotation map: {len(type_of)} typed root_ids", flush=True)

    per_cell, conv = [], []
    t = throttle(time.time())
    for ctype in ("R1-6", "R7", "R8"):
        ann = ann_all[(ann_all.cell_type.astype(str) == ctype)
                      & (ann_all.cell_class.astype(str) == "visual")]
        ids = [int(x) for x in ann.root_id.unique() if int(x) != 0]
        pick = list(rng.choice(ids, size=min(NSAMPLE, len(ids)), replace=False))
        print(f"\n{ctype}: {len(ids)} annotated -> sampling {len(pick)}", flush=True)

        syn = []
        for k in range(0, len(pick), CHUNK):
            t = throttle(t)
            batch = [int(x) for x in pick[k:k + CHUNK]]
            s = flywire.get_synapses(batch, dataset="public", pre=True, post=True,
                                     min_score=50, transmitters=False)
            syn.append(s)
            print(f"  chunk {k // CHUNK + 1}/{-(-len(pick) // CHUNK)}: {len(s)} synapses",
                  flush=True)
        syn = pd.concat(syn, ignore_index=True)
        syn = syn[(syn.pre != syn.post) & (syn.pre != 0) & (syn.post != 0)]

        for b in pick:
            o = syn[syn.pre == b]
            i = syn[syn.post == b]
            if not len(o):
                continue
            op = o.groupby("post").size()
            ip = i.groupby("pre").size()
            per_cell.append({"root": b, "class": ctype, "out_syn": int(op.sum()),
                             "out_partners": int(op.shape[0]),
                             "in_syn": int(ip.sum()) if len(ip) else 0,
                             "in_partners": int(ip.shape[0]) if len(ip) else 0,
                             "top_partner": int(op.max()),
                             "top3_pct": int(round(op.nlargest(3).sum() / op.sum() * 100)),
                             "singletons": int((op == 1).sum())})

        # convergence: which target TYPES do these photoreceptors project to?
        out = syn[syn.pre.isin(pick)].copy()
        out["post_type"] = out.post.map(lambda r: type_of.get(int(r), "(untyped fragment)"))
        g = out.groupby("post_type").agg(n_syn=("post", "size"),
                                         n_sources=("pre", "nunique"),
                                         n_targets=("post", "nunique")).reset_index()
        g["pct_of_output"] = (g.n_syn / g.n_syn.sum() * 100).round(1)
        g["src_class"] = ctype
        conv.append(g.sort_values("n_syn", ascending=False))

    pc = pd.DataFrame(per_cell)
    pc.to_csv(fig("fafb_photoreceptor_percell.csv"), index=False)
    cv = pd.concat(conv, ignore_index=True)
    cv.to_csv(fig("fafb_convergence.csv"), index=False)

    pd.set_option("display.width", 170)
    print("\n=== FAFB per-cell summary, median (min-max), ALL sampled cells ===")
    for cls, d in pc.groupby("class"):
        print(f"  {cls}: n={len(d)}  out_syn {int(d.out_syn.median())} "
              f"({int(d.out_syn.min())}-{int(d.out_syn.max())})  "
              f"top_prtnr {int(d.top_partner.median())}  top3% {int(d.top3_pct.median())}")
    print("\n=== FAFB top convergent OUTPUT TYPES ===")
    for cls, d in cv.groupby("src_class"):
        print(f"\n  from {cls} (n sources sampled):")
        print(d.head(10)[["post_type", "n_syn", "pct_of_output", "n_sources", "n_targets"]]
              .to_string(index=False))
    print("\nsaved fafb_photoreceptor_percell.csv + fafb_convergence.csv\nDONE")


if __name__ == "__main__":
    main()
