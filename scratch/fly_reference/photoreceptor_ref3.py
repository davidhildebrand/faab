"""Recompute ALL Drosophila photoreceptor metrics from ONE consistent, representative
sample per group, so every column matches the FAAB per-cell table (Table 1).

Sampling: cells whose output-synapse count lies within the group IQR (excludes
truncated/cropped bodies and outliers), then N per group with a fixed seed.
"""
from faab.config import REPO
import os
import numpy as np
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.sources.neuprint import client as npclient, MALE_CNS, OPTIC_LOBE

OUT = os.path.join(REPO, "figures")
N = 25
SEED = 0


def group_of(t):
    t = str(t)
    if t.startswith("R1-R6") or t in list("R1 R2 R3 R4 R5 R6".split()):
        return "R1-R6"
    if t.startswith("R7"):
        return "R7"
    if t.startswith("R8"):
        return "R8"
    return None


def metrics(c, bodies):
    # label='Segment' includes EVERY partner body incl. small fragments, matching how
    # CAVE synapses_v2 counts. The default label='Neuron' drops ~46% of output synapses.
    dn = fetch_simple_connections(NC(bodyId=bodies), NC(label="Segment"), client=c)
    up = fetch_simple_connections(NC(label="Segment"), NC(bodyId=bodies), client=c)
    rows = []
    for b in bodies:
        d = dn[dn.bodyId_pre == b]
        u = up[up.bodyId_post == b]
        if not len(d):
            continue
        rows.append({"out_syn": int(d.weight.sum()),
                     "out_partners": int(d.bodyId_post.nunique()),
                     "in_syn": int(u.weight.sum()) if len(u) else 0,
                     "in_partners": int(u.bodyId_pre.nunique()) if len(u) else 0,
                     "top_partner": int(d.weight.max()),
                     "singletons": int((d.weight == 1).sum()),
                     "top3_share_pct": int(round(d.weight.nlargest(3).sum()
                                                 / d.weight.sum() * 100))})
    return pd.DataFrame(rows)


def main():
    recs = []
    for DS, tag in [(MALE_CNS, "Dmel mCNS"), (OPTIC_LOBE, "Dmel OL")]:
        c = npclient(DS)
        ndf, _ = fetch_neurons(NC(type="R[1-8].*", regex=True), client=c)
        ndf["grp"] = ndf["type"].map(group_of)
        ndf = ndf[ndf.grp.notna() & (ndf["downstream"] > 0)]
        for grp, d in ndf.groupby("grp"):
            q1, q3 = d["downstream"].quantile([0.25, 0.75])
            mid = d[(d["downstream"] >= q1) & (d["downstream"] <= q3)]
            bodies = mid.sample(min(N, len(mid)), random_state=SEED)["bodyId"].tolist()
            m = metrics(c, bodies)
            print(f"{tag:10} {grp:6} sampled {len(m)} cells (IQR {int(q1)}-{int(q3)} out-syn)",
                  flush=True)
            recs.append({"source": tag, "class": grp, "n": len(m),
                         **{k: int(round(m[k].median())) for k in m.columns}})
    df = pd.DataFrame(recs)
    print("\n=== Drosophila reference, all metrics from one consistent sample ===")
    pd.set_option("display.width", 170)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, "photoreceptor_ref_consistent.csv"), index=False)
    print("\nsaved photoreceptor_ref_consistent.csv\nDONE")


if __name__ == "__main__":
    main()
