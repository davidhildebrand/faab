"""Convergent output TYPES of male-CNS photoreceptors: which target cell types do
R1-6 / R7 / R8 send their output to, and how many source cells converge on each.
Uses label='Segment' so untyped fragments are counted too (comparable to CAVE)."""
from faab.config import REPO
import os
import numpy as np
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.clients.neuprint import client as npclient, MALE_CNS

OUT = os.path.join(REPO, "figures")
NSAMPLE = 60
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
    allg = []
    for grp, d in ndf.groupby("grp"):
        pick = list(rng.choice(d.bodyId.tolist(),
                               size=min(NSAMPLE, len(d)), replace=False))
        dn = fetch_simple_connections(NC(bodyId=pick), NC(label="Segment"), client=c)
        dn["post_type"] = dn["type_post"].fillna("(untyped fragment)").astype(str)
        g = (dn.groupby("post_type")
             .agg(n_syn=("weight", "sum"), n_sources=("bodyId_pre", "nunique"),
                  n_targets=("bodyId_post", "nunique")).reset_index())
        g["pct_of_output"] = (g.n_syn / g.n_syn.sum() * 100).round(1)
        g["src_class"] = grp
        g["n_sampled"] = len(pick)
        allg.append(g.sort_values("n_syn", ascending=False))
        print(f"{grp}: sampled {len(pick)} cells, {int(g.n_syn.sum())} output synapses",
              flush=True)
    cv = pd.concat(allg, ignore_index=True)
    cv.to_csv(os.path.join(OUT, "mcns_convergence.csv"), index=False)
    pd.set_option("display.width", 170)
    print("\n=== male-CNS top convergent OUTPUT TYPES ===")
    for cls, d in cv.groupby("src_class"):
        n = int(d.n_sampled.iloc[0])
        print(f"\n  from {cls} ({n} source cells sampled):")
        print(d.head(10)[["post_type", "n_syn", "pct_of_output", "n_sources", "n_targets"]]
              .to_string(index=False))
    print("\nsaved mcns_convergence.csv\nDONE")


if __name__ == "__main__":
    main()
