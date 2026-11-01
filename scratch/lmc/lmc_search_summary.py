"""Summary of the LMC search (2026-11-01): L1-L4 calls (lmc_cand_typing.tsv; the photoreceptor net's 'L5' calls do not
match the L5 profile and count as 'type unclear') and the L5 search (lmc_l5_candidates.tsv, L5 profile; with a nucleus
by the lamina = likely, without a nucleus = soma unchecked), per side, type and confidence; overlap between the two.
Network-free. Usage: lmc_search_summary.py"""
import pandas as pd

from faab.config import fig

T = pd.read_csv(fig("lmc_cand_typing.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": ""})
L5 = pd.read_csv(fig("lmc_l5_candidates.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": ""})
L5 = L5[L5.l5_profile & ~L5.known_l5]
T["type"] = T.call.where(T.call != "L5", "unclear")
T["level"] = T.confidence.where(T.soma_by != "ME", "soma by the medulla")
print("L1-L4 search: type x confidence, per side")
print(pd.crosstab([T.side, T.type], T.level, margins=True).to_string())
L5["level"] = L5.soma_by.map({"LA": "soma by the lamina"}).fillna("no nucleus on the root")
print("\nL5 search: per side")
print(pd.crosstab(L5.side, L5.level, margins=True).to_string())
both = set(L5.root) & set(T.root)
print(f"\nin both searches: {len(both)} (their L1-L4 search call: {T[T.root.isin(both)].call.value_counts().to_dict()})")
