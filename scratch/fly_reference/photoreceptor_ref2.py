"""How CONCENTRATED is Drosophila photoreceptor output? Distinct downstream partner
bodies per cell, and top-partner share — to compare against the aedes distribution."""
import numpy as np
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.sources.neuprint import client as npclient, MALE_CNS


def main():
    c = npclient(MALE_CNS)
    for grp, crit in [("R1-R6", NC(type="R1-R6")), ("R7", NC(type="R7y")), ("R8", NC(type="R8y"))]:
        ndf, _ = fetch_neurons(crit, client=c)
        ndf = ndf[ndf["downstream"] > 0]
        sample = ndf.sort_values("downstream", ascending=False)["bodyId"].head(12).tolist()
        dn = fetch_simple_connections(NC(bodyId=sample), None, client=c)
        rows = []
        for b in sample:
            d = dn[dn.bodyId_pre == b]
            if not len(d):
                continue
            tot = d.weight.sum()
            rows.append({"partners": d.bodyId_post.nunique(),
                         "syn": int(tot),
                         "top_partner_syn": int(d.weight.max()),
                         "top3_share": float(d.weight.nlargest(3).sum() / tot),
                         "singletons": int((d.weight == 1).sum())})
        r = pd.DataFrame(rows)
        print(f"\n=== male-CNS {grp} (n={len(r)} well-traced cells) ===")
        print(f"  output synapses      : median {r.syn.median():.0f}")
        print(f"  distinct partners    : median {r.partners.median():.0f} "
              f"(range {r.partners.min()}-{r.partners.max()})")
        print(f"  syn per partner      : median {r.syn.median()/r.partners.median():.1f}")
        print(f"  top partner synapses : median {r.top_partner_syn.median():.0f}")
        print(f"  top-3 partner share  : median {r.top3_share.median()*100:.0f}% of all output")
        print(f"  singleton partners   : median {r.singletons.median():.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
