"""Fresh Drosophila reference numbers for photoreceptor input/output synapse counts,
to compare against the aedes (FAAB) cells. Datasets: male-CNS:v1.0, optic-lobe:v1.1.

Note on comparability: CAVE `synapses_v2` counts individual synapses, so the matching
neuprint columns are `downstream` (output synapses) and `upstream` (input synapses).
`pre`/`post` are T-bars / PSDs and are listed for context.
"""
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons
from faab.sources.neuprint import client as npclient, MALE_CNS, OPTIC_LOBE


def group_of(t):
    t = str(t)
    if t.startswith("R1-R6") or t in ("R1", "R2", "R3", "R4", "R5", "R6"):
        return "R1-R6"
    if t.startswith("R7"):
        return "R7"
    if t.startswith("R8"):
        return "R8"
    return None


def main():
    for DS, name in [(MALE_CNS, "male-CNS:v1.0"), (OPTIC_LOBE, "optic-lobe:v1.1")]:
        c = npclient(DS)
        ndf, _ = fetch_neurons(NC(type="R[1-8].*", regex=True), client=c)
        ndf["grp"] = ndf["type"].map(group_of)
        ndf = ndf[ndf.grp.notna()]
        # drop the untraced 'unclear' bodies with zero synapses
        ndf = ndf[(ndf["downstream"] > 0) | (ndf["upstream"] > 0)]
        g = ndf.groupby("grp").agg(
            n=("bodyId", "size"),
            out_syn_median=("downstream", "median"),
            out_syn_q1=("downstream", lambda s: s.quantile(0.25)),
            out_syn_q3=("downstream", lambda s: s.quantile(0.75)),
            in_syn_median=("upstream", "median"),
            in_syn_q1=("upstream", lambda s: s.quantile(0.25)),
            in_syn_q3=("upstream", lambda s: s.quantile(0.75)),
            tbars_median=("pre", "median"),
            psd_median=("post", "median"))
        print(f"\n=== {name} ===")
        pd.set_option("display.width", 170)
        print(g.round(0).to_string())
    print("\nDONE")


if __name__ == "__main__":
    main()
