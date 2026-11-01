"""Where do L5s get their input (2026-11-01, LMC search: L5 gets almost no photoreceptor synapses, so a second net is
needed)? For the SeaTable cells typed L5 (roots at their SeaTable point, as fetched): their presynaptic partners from
the local snapshot, sorted by synapses, labeled by SeaTable type or by the candidate table (photoreceptor target with
medulla output = LMC-like). --out: their postsynaptic partners instead.
Network-free apart from the snapshot catch-up. Usage: lmc_l5_inputs.py SEATABLE.tsv [--out]"""
import sys

import pandas as pd
import caveclient

from faab import synsnap
from faab.clients.cave import DATASTACK
from faab.config import fig

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
lab = S[["class", "subclass", "subsubclass", "type"]].agg(" ".join, axis=1).str.split().str.join(" ")
st = dict(zip(S.root_now, lab))
T = pd.read_csv(fig("lmc_candidates.tsv"), sep="\t", dtype={"root": str}).fillna("")
lmc_like = set(T[(T.n_in_LA >= 20) & (T.n_out_ME >= 20)].root)
l5 = S[(S["class"] == "LMC") & lab.str.contains(r"\bL5\b")].root_now.tolist()
out = "--out" in sys.argv
me, other = ("pre", "post") if out else ("post", "pre")  # the L5 side and the partner side of each edge
e = synsnap.Snapshot(log=lambda *_: None).edges(**{me: [int(r) for r in l5]}, when="now",
                                                   client=caveclient.CAVEclient(DATASTACK))
e[other] = e[other].astype(str)
e["who"] = [st.get(p) or ("LMC-like candidate" if p in lmc_like else "-") for p in e[other]]
tot = e.groupby(me).n.sum()
print(f"{len(l5)} L5s; {'outputs' if out else 'inputs'} per L5 p50 {int(tot.median())}")
w = e.groupby("who").agg(synapses=("n", "sum"), partners=(other, "nunique"), l5s_reached=(me, "nunique"))
w["share"] = (w.synapses / e.n.sum()).round(3)
print(w.sort_values("synapses", ascending=False).head(15).to_string())
strong = e[e.n >= 5].groupby("who")[me].nunique().sort_values(ascending=False)
print("\nL5s with >= 5 synapses from a partner of this kind:", strong.head(8).to_dict())
