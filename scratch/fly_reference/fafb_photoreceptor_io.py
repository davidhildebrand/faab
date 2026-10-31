"""Compute the same photoreceptor connectivity metrics in FAFB/FlyWire (public,
materialization 783) as for aedes and male-CNS, so all three tables share columns.

FlyWire is fully proofread, which makes it the strongest reference for judging
whether the aedes counts reflect biology or incomplete reconstruction.
Read-only, throttled.
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
NSAMPLE = 40
SEED = 0


def main():
    rng = np.random.RandomState(SEED)
    t = throttle(time.time())
    recs = []
    per_cell_all = []
    for ctype in ("R1-6", "R7", "R8"):
        t = throttle(t)
        ann = flywire.search_annotations(ctype, dataset="public")
        ann = ann[(ann["cell_type"].astype(str) == ctype)
                  & (ann["cell_class"].astype(str) == "visual")]
        ids = [int(x) for x in ann["root_id"].unique() if int(x) != 0]
        pick = list(rng.choice(ids, size=min(NSAMPLE, len(ids)), replace=False))
        print(f"{ctype}: {len(ids)} annotated -> sampling {len(pick)}", flush=True)

        t = throttle(t)
        syn = flywire.get_synapses(pick, dataset="public", pre=True, post=True,
                                   min_score=50, transmitters=False)
        syn = syn[(syn.pre != syn.post) & (syn.pre != 0) & (syn.post != 0)]
        rows = []
        for b in pick:
            o = syn[syn.pre == b]
            i = syn[syn.post == b]
            if not len(o):
                continue
            op = o.groupby("post").size()
            ip = i.groupby("pre").size()
            rows.append({"root": b, "out_syn": int(op.sum()),
                         "out_partners": int(op.shape[0]),
                         "in_syn": int(ip.sum()) if len(ip) else 0,
                         "in_partners": int(ip.shape[0]) if len(ip) else 0,
                         "top_partner": int(op.max()),
                         "top3_pct": int(round(op.nlargest(3).sum() / op.sum() * 100)),
                         "singletons": int((op == 1).sum())})
        m = pd.DataFrame(rows)
        m["class"] = ctype
        per_cell_all.append(m)
        # representative subset: cells within the IQR of output synapses
        q1, q3 = m.out_syn.quantile([0.25, 0.75])
        mid = m[(m.out_syn >= q1) & (m.out_syn <= q3)]
        print(f"  {len(m)} cells with output; IQR {int(q1)}-{int(q3)}; "
              f"{len(mid)} in IQR", flush=True)
        recs.append({"source": "FAFB", "class": ctype, "n": len(mid),
                     **{k: int(round(mid[k].median())) for k in
                        ("out_syn", "out_partners", "in_syn", "in_partners",
                         "top_partner", "top3_pct", "singletons")}})

    df = pd.DataFrame(recs)
    print("\n=== FAFB/FlyWire photoreceptor reference (median of IQR cells) ===")
    pd.set_option("display.width", 170)
    print(df.to_string(index=False))
    df.to_csv(fig("fafb_photoreceptor_ref.csv"), index=False)
    pd.concat(per_cell_all).to_csv(fig("fafb_photoreceptor_percell.csv"),
                                   index=False)
    print("\nsaved fafb_photoreceptor_ref.csv + fafb_photoreceptor_percell.csv\nDONE")


if __name__ == "__main__":
    main()
