"""Does fetch_simple_connections exclude small (Segment) partners? If so, fly partner
counts are NOT comparable to the aedes counts, which include every fragment.

Compare, for the same fly cells: the `downstream` neuron property (ALL output synapses)
vs the summed weights returned with label='Neuron' (default) and label='Segment'.
"""
import numpy as np
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.sources.neuprint import client as npclient, MALE_CNS


def main():
    c = npclient(MALE_CNS)
    ndf, _ = fetch_neurons(NC(type="R1-R6"), client=c)
    ndf = ndf[ndf["downstream"] > 0]
    q1, q3 = ndf["downstream"].quantile([0.25, 0.75])
    mid = ndf[(ndf["downstream"] >= q1) & (ndf["downstream"] <= q3)]
    bodies = mid.sample(10, random_state=0)["bodyId"].tolist()
    prop = dict(zip(ndf.bodyId, ndf.downstream))

    for lab in ("Neuron", "Segment"):
        try:
            dn = fetch_simple_connections(NC(bodyId=bodies), NC(label=lab), client=c)
        except Exception as e:
            print(f"label={lab}: query failed: {str(e)[:90]}")
            continue
        rows = []
        for b in bodies:
            d = dn[dn.bodyId_pre == b]
            rows.append({"body": b, "prop_downstream": prop[b],
                         "conn_syn": int(d.weight.sum()) if len(d) else 0,
                         "partners": int(d.bodyId_post.nunique()) if len(d) else 0,
                         "singletons": int((d.weight == 1).sum()) if len(d) else 0})
        r = pd.DataFrame(rows)
        r["captured_%"] = (r.conn_syn / r.prop_downstream * 100).round(0)
        print(f"\n=== label='{lab}' ===")
        print(r.to_string(index=False))
        print(f"  median partners {r.partners.median():.0f}, "
              f"median captured {r['captured_%'].median():.0f}% of downstream synapses, "
              f"median singletons {r.singletons.median():.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
