"""male-CNS photoreceptor metrics with NO completeness filtering: sample N cells at
random per type, compute per-cell metrics, save them. Medians and ranges are derived
downstream so nothing is silently excluded."""
from faab.config import REPO
import os
import numpy as np
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.sources.neuprint import client as npclient, MALE_CNS

OUT = os.path.join(REPO, "figures")
NSAMPLE = 40
SEED = 0


def group_of(t):
    t = str(t)
    if t.startswith("R1-R6") or t in "R1 R2 R3 R4 R5 R6".split():
        return "R1-6"
    if t.startswith("R7"):
        return "R7"
    if t.startswith("R8"):
        return "R8"
    return None


def main():
    c = npclient(MALE_CNS)
    ndf, _ = fetch_neurons(NC(type="R[1-8].*", regex=True), client=c)
    ndf["grp"] = ndf["type"].map(group_of)
    ndf = ndf[ndf.grp.notna()]
    rng = np.random.RandomState(SEED)
    out = []
    for grp, d in ndf.groupby("grp"):
        ids = d.bodyId.tolist()
        pick = list(rng.choice(ids, size=min(NSAMPLE, len(ids)), replace=False))
        print(f"{grp}: {len(ids)} annotated -> random sample of {len(pick)} (no filtering)",
              flush=True)
        dn = fetch_simple_connections(NC(bodyId=pick), NC(label="Segment"), client=c)
        up = fetch_simple_connections(NC(label="Segment"), NC(bodyId=pick), client=c)
        for b in pick:
            o = dn[dn.bodyId_pre == b]
            i = up[up.bodyId_post == b]
            if not len(o):
                out.append({"root": b, "class": grp, "out_syn": 0, "out_partners": 0,
                            "in_syn": int(i.weight.sum()) if len(i) else 0,
                            "in_partners": int(i.bodyId_pre.nunique()) if len(i) else 0,
                            "top_partner": 0, "top3_pct": 0, "singletons": 0})
                continue
            out.append({"root": b, "class": grp,
                        "out_syn": int(o.weight.sum()),
                        "out_partners": int(o.bodyId_post.nunique()),
                        "in_syn": int(i.weight.sum()) if len(i) else 0,
                        "in_partners": int(i.bodyId_pre.nunique()) if len(i) else 0,
                        "top_partner": int(o.weight.max()),
                        "top3_pct": int(round(o.weight.nlargest(3).sum() / o.weight.sum() * 100)),
                        "singletons": int((o.weight == 1).sum())})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "mcns_photoreceptor_percell.csv"), index=False)
    pd.set_option("display.width", 170)
    print("\nmedian (min-max) per class, ALL sampled cells:")
    for cls, d in df.groupby("class"):
        print(f"  {cls}: n={len(d)}  out_syn {int(d.out_syn.median())} "
              f"({int(d.out_syn.min())}-{int(d.out_syn.max())})")
    print("\nsaved mcns_photoreceptor_percell.csv\nDONE")


if __name__ == "__main__":
    main()
